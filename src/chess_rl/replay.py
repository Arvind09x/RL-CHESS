"""R02 completed-game replay, audited through R01 and the shared adapter.

SQLite transactions are the admission boundary. No weights are loaded, no
search is rerun, and no unfinished position acquires an outcome target.
"""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3

from chess_rl.adapter import encode_position, legal_mask, make_state, terminal_result
from chess_rl.contracts import AdapterConfig, MOVE_COUNT
from chess_rl.evaluation.matches import digest
from chess_rl.selfplay import audit, _read, read_game_record
from chess_rl.training.checkpoints import ModelConfig


MANIFEST = dict(schema="r02-replay-v1", contracts=asdict(AdapterConfig()),
    duplicate_policy="first-admitted-completed-trajectory-v1",
    game_identity="sha256([initial_fen, full actual moves_uci])",
    source_identity="sha256(canonical R01 configuration and game snapshot)",
    sample_identity="sha256([r02-sample-v1, game_id, configuration_sha256, ply])",
    target_type="search-visit-distribution/exact-final-root-side-WDL",
    storage="SQLite; one complete source/game/sample transaction; synchronous=FULL")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(value, fields, label):
    _require(type(value) is dict and set(value) == set(fields), f"{label} schema mismatch")


def read_source(directory):
    """Read checksummed R01 files without a runtime match or checkpoint inference."""
    directory = Path(directory)
    snapshot = dict(configuration=_read(directory / "configuration.json"),
                    game=read_game_record(directory))
    finalize(snapshot)
    return snapshot


def finalize(snapshot):
    """Validate all statuses; return played-root targets only after completion.

    Full actual history is authoritative. R01 audit validates the execution
    record; additional strict primitive checks close JSON bool/int equivalence.
    Replayed opening positions and the final terminal board are never sampled.
    """
    try:
        snapshot = json.loads(_json(snapshot))  # owned, finite JSON only
        _fields(snapshot, ("configuration", "game"), "source")
        configuration, game = snapshot["configuration"], snapshot["game"]
        _fields(configuration, ("schema", "checkpoint", "contracts", "runtime", "settings",
            "seed", "opening_selection_seed", "opening", "rng", "execution", "boundary"), "R01 configuration")
        _fields(game, ("schema", "configuration_sha256", "positions", "interruptions", "rng_state",
            "initial_fen", "moves_uci", "history_complete", "final_fen", "final_promoted",
            "winner", "outcome", "status", "ending_reason"), "R01 game")
        _require(_json(configuration["contracts"]) == _json(asdict(AdapterConfig())), "incompatible contracts")
        checkpoint = configuration["checkpoint"]
        _fields(checkpoint, ("path", "sha256", "bytes", "checkpoint_id", "model_config",
                            "contracts", "provenance", "role"), "checkpoint identity")
        _require(type(checkpoint["sha256"]) is str and len(checkpoint["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in checkpoint["sha256"]), "invalid checkpoint SHA256")
        _require(type(checkpoint["bytes"]) is int and checkpoint["bytes"] > 0, "invalid checkpoint size")
        for field in ("checkpoint_id", "path", "role"):
            _require(type(checkpoint[field]) is str and bool(checkpoint[field].strip()), "invalid checkpoint identity")
        _require(type(checkpoint["provenance"]) is dict, "invalid checkpoint provenance")
        _require(_json(checkpoint["contracts"]) == _json(configuration["contracts"]), "checkpoint contracts mismatch")
        _require(_json(asdict(ModelConfig(**checkpoint["model_config"]))) == _json(checkpoint["model_config"]),
                 "incomplete checkpoint model configuration")
        _require(game["history_complete"] is True, "complete game history required")
        _require(type(game["moves_uci"]) is list and type(game["positions"]) is list
                 and type(game["interruptions"]) is list, "invalid history arrays")
        _require(type(game["final_promoted"]) is int, "invalid promoted bits")
        for position in game["positions"]:
            _fields(position, ("ply", "root_fen", "root_promoted", "side_to_move", "legal_mask",
                "root_visits", "visit_policy", "completed_simulations", "move_id", "move_uci",
                "exploration", "rng_before", "rng_after"), "R01 position")
            for name in ("ply", "root_promoted", "completed_simulations", "move_id"):
                _require(type(position[name]) is int and position[name] >= 0, f"invalid {name}")
            _require(type(position["legal_mask"]) is list and len(position["legal_mask"]) == MOVE_COUNT
                     and all(type(v) is bool for v in position["legal_mask"]), "invalid legal mask")
            policy = position["visit_policy"]
            _require(type(policy) is list and len(policy) == MOVE_COUNT and all(
                type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in policy), "invalid visit policy")
        final_state = audit(configuration, game)
        game_id = digest([game["initial_fen"], game["moves_uci"]])
        source_id = digest(snapshot)
        samples = []
        if game["status"] == "completed":
            terminal = terminal_result(final_state)
            _require(terminal is not None, "completed game must have a rule outcome")
            for position in game["positions"]:
                ply = position["ply"]
                state = make_state(game["initial_fen"], game["moves_uci"][:ply], history_complete=True)
                _require(terminal_result(state) is None, "terminal played root")
                _require(position["legal_mask"] == legal_mask(state).tolist(), "root mask mismatch")
                encoded = encode_position(state)
                wdl = ([0, 1, 0] if terminal.winner is None else
                       [1, 0, 0] if terminal.winner == state.board.turn else [0, 0, 1])
                samples.append(dict(schema="r02-sample-v1", sample_id=digest([
                    "r02-sample-v1", game_id, game["configuration_sha256"], ply]),
                    game_id=game_id, source_id=source_id,
                    configuration_sha256=game["configuration_sha256"],
                    contracts=configuration["contracts"], checkpoint=checkpoint,
                    opening=configuration["opening"], search_settings=configuration["settings"],
                    source="completed-R01-self-play", policy_target_type="root-visit-distribution",
                    wdl_target_type="exact-final-root-side-WDL", initial_fen=game["initial_fen"],
                    moves_uci=game["moves_uci"][:ply], history_complete=True,
                    available_snapshots=encoded.available_snapshots, position=position,
                    wdl=wdl, outcome=game["outcome"], ending_reason=game["ending_reason"]))
        return dict(source_id=source_id, game_id=game_id, status=game["status"],
                    snapshot=snapshot, samples=samples)
    except (KeyError, TypeError, OverflowError, IndexError) as exc:
        raise ValueError("malformed R01 replay source") from exc


class ReplayBuffer:
    """Bounded local buffer; opening or external changes reaudit stored targets.

    First committed completed trajectory wins, even if later records have other
    checkpoint/search/opening identities or root coverage. Every distinct audited
    source is retained as a receipt. Repeated identical ingestion is a no-op.
    An unfinished snapshot does not reserve a trajectory; a resumed completion
    is a new source and can be admitted. No capacity eviction or training here.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._validated_signature = None
        self._connection = None
        _require(self.path.is_file(), "replay buffer missing; create it explicitly")
        connection = self._connect()
        try:
            with connection:
                connection.execute("BEGIN")
                self._validate_if_changed(connection)
        except Exception:
            self.close()
            raise

    @classmethod
    def create(cls, path):
        path = Path(path)
        # Exclusive file creation prevents replacing any existing artifact.
        with path.open("xb"):
            pass
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            with connection:
                connection.execute("CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)")
                connection.execute("INSERT INTO metadata VALUES (1, ?)", (_json(MANIFEST),))
                connection.execute("CREATE TABLE sources (ordinal INTEGER PRIMARY KEY, source_id TEXT NOT NULL UNIQUE, "
                    "game_id TEXT NOT NULL, disposition TEXT NOT NULL, payload TEXT NOT NULL)")
                connection.execute("CREATE TABLE games (game_id TEXT PRIMARY KEY, source_id TEXT NOT NULL UNIQUE "
                    "REFERENCES sources(source_id), sample_count INTEGER NOT NULL)")
                connection.execute("CREATE TABLE samples (sample_id TEXT PRIMARY KEY, game_id TEXT NOT NULL "
                    "REFERENCES games(game_id), payload TEXT NOT NULL)")
        finally:
            connection.close()
        return cls(path)

    def _connect(self):
        if self._connection is None:
            connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=10)
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            self._connection = connection
        return self._connection

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._validated_signature = None

    def __del__(self):
        try:
            self.close()
        except (AttributeError, sqlite3.Error):
            pass

    def _signature(self):
        # The supported store uses rollback journals. WAL has a separate live
        # file; disable this optimization if an external writer enables WAL.
        if Path(str(self.path) + "-wal").exists():
            return None
        stat = self.path.stat()
        return (self._connect().execute("PRAGMA data_version").fetchone()[0], stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def _validate_if_changed(self, connection):
        signature = self._signature()
        if signature is None or signature != self._validated_signature:
            self._validate(connection)
            self._validated_signature = signature

    @staticmethod
    def _validate(connection):
        try:
            _require(connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite corruption")
            _require(not connection.execute("PRAGMA foreign_key_check").fetchall(), "broken replay references")
            _require(connection.execute("SELECT id, payload FROM metadata").fetchall() == [(1, _json(MANIFEST))],
                     "incompatible replay manifest")
            games, samples = {}, {}
            for _, source_id, game_id, disposition, payload in connection.execute(
                    "SELECT * FROM sources ORDER BY ordinal"):
                result = finalize(json.loads(payload))
                _require(payload == _json(result["snapshot"]) and source_id == result["source_id"]
                         and game_id == result["game_id"], "source identity/content mismatch")
                expected = "excluded-" + result["status"]
                if result["status"] == "completed":
                    expected = "duplicate" if game_id in games else "admitted"
                    if expected == "admitted":
                        games[game_id] = (source_id, len(result["samples"]))
                        for sample in result["samples"]:
                            samples[sample["sample_id"]] = (game_id, _json(sample))
                _require(disposition == expected, "invalid source admission")
            stored_games = {key: (source, count) for key, source, count in connection.execute("SELECT * FROM games")}
            stored_samples = {key: (game, payload) for key, game, payload in connection.execute("SELECT * FROM samples")}
            _require(games == stored_games and samples == stored_samples, "replay target/admission corruption")
        except (sqlite3.DatabaseError, KeyError, TypeError) as exc:
            raise ValueError("corrupt replay buffer") from exc

    def ingest(self, snapshot):
        result = finalize(snapshot)
        connection = self._connect()
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            self._validate_if_changed(connection)
            version = connection.execute("PRAGMA data_version").fetchone()[0]
            prior = connection.execute("SELECT disposition FROM sources WHERE source_id=?",
                                       (result["source_id"],)).fetchone()
            if prior:
                receipt = dict(source_id=result["source_id"], disposition=prior[0], repeated=True, added_samples=0)
            else:
                disposition = "excluded-" + result["status"]
                if result["status"] == "completed":
                    disposition = "duplicate" if connection.execute("SELECT 1 FROM games WHERE game_id=?",
                        (result["game_id"],)).fetchone() else "admitted"
                connection.execute("INSERT INTO sources(source_id, game_id, disposition, payload) VALUES(?,?,?,?)",
                    (result["source_id"], result["game_id"], disposition, _json(result["snapshot"])))
                count = 0
                if disposition == "admitted":
                    count = len(result["samples"])
                    connection.execute("INSERT INTO games VALUES(?,?,?)", (result["game_id"], result["source_id"], count))
                    for sample in result["samples"]:
                        connection.execute("INSERT INTO samples VALUES(?,?,?)",
                            (sample["sample_id"], result["game_id"], _json(sample)))
                receipt = dict(source_id=result["source_id"], disposition=disposition, repeated=False, added_samples=count)
        signature = self._signature()
        # Our persistent connection's data_version excludes its own commits.
        # Retain the version checked under the lock so another writer racing
        # after our commit cannot be accidentally blessed by this cache.
        self._validated_signature = None if signature is None else (version, *signature[1:])
        return receipt

    def summary(self):
        """export()["summary"] without materializing sources or samples (large campaigns)."""
        connection = self._connect()
        with connection:
            connection.execute("BEGIN")
            self._validate_if_changed(connection)
            counts = dict(connection.execute(
                "SELECT disposition, COUNT(*) FROM sources GROUP BY disposition").fetchall())
            positions = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        return dict(source_counts={name: counts.get(name, 0) for name in
                                   ("admitted", "duplicate", "excluded-truncated", "excluded-interrupted")},
                    training_positions=positions)

    def iter_samples(self):
        """export()["samples"] one at a time, from one read snapshot; consume fully before other calls."""
        connection = self._connect()
        with connection:
            connection.execute("BEGIN")
            self._validate_if_changed(connection)
            for (payload,) in connection.execute("SELECT payload FROM samples ORDER BY sample_id"):
                yield json.loads(payload)

    def export(self):
        connection = self._connect()
        with connection:
            connection.execute("BEGIN")  # one consistent read snapshot
            self._validate_if_changed(connection)
            sources = [dict(source_id=source, game_id=game, disposition=disposition,
                snapshot=json.loads(payload)) for _, source, game, disposition, payload in connection.execute(
                    "SELECT * FROM sources ORDER BY ordinal")]
            samples = [json.loads(payload) for (payload,) in connection.execute(
                "SELECT payload FROM samples ORDER BY sample_id")]
            counts = {name: sum(s["disposition"] == name for s in sources) for name in
                      ("admitted", "duplicate", "excluded-truncated", "excluded-interrupted")}
            return dict(manifest=copy.deepcopy(MANIFEST), sources=sources, samples=samples,
                        summary=dict(source_counts=counts, training_positions=len(samples)))
