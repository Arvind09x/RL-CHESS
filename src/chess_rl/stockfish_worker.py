"""D05 bounded, sequential teacher annotation of audited replay references.

No result cache, sample refill, training, or annotation resumption is provided.
The D04 overlap identity is never used as a teacher cache key.
"""
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import tempfile

import chess
import chess.engine

from chess_rl.adapter import make_state, terminal_result, move_to_id, id_to_move
from chess_rl.canonical_games import canonical_json, load_json
from chess_rl.pgn_replay import file_identity, json_line
from chess_rl.position_sampling import (PHASES, RECORD_VERSION, SAMPLING_VERSION,
    SAMPLING_DESCRIPTION, PHASE_DESCRIPTION, phase, rule_position, validate_game,
    verify_intake)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_artifacts(manifest, manifest_path, games_path, split_audit_path, prepared):
    """Verify accepted metadata and all authoritative table/registry bytes."""
    manifest.require_annotation_ready()
    if manifest_path.read_text() != manifest.to_json():
        raise ValueError("Manifest bytes differ from canonical loaded manifest")
    split, identities = verify_intake(games_path, split_audit_path,
        manifest.preparation.split_audit_sha256, manifest)
    path = prepared / "overlap-audit.json"
    identity = file_identity(path)
    if identity["sha256"] != manifest.preparation.overlap_audit_sha256:
        raise ValueError("Overlap audit identity mismatch")
    audit = load_json(path.read_bytes())
    expected = dict(audit_version="position-overlap-audit-v1", complete=True,
        dataset_id=manifest.dataset_id, sampling_version=SAMPLING_VERSION,
        sampling_seed=manifest.sampling_seed, sampling_algorithm=SAMPLING_DESCRIPTION,
        phase_boundaries=PHASE_DESCRIPTION, overlap_identity=manifest.overlap_policy,
        frozen_registry_version="protected-heldout-v1", cross_split_overlaps_after=0,
        labels_produced=0, assignments_sha256=split["assignments_sha256"], inputs=identities)
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("Overlap audit contract mismatch")
    identities["overlap-audit.json"] = identity
    for name in ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl", "exclusions.jsonl"):
        identities[name] = file_identity(prepared / name)
        if identities[name] != audit["outputs"][name]:
            raise ValueError(f"Audited output identity mismatch: {name}")
    if identities["protected-heldout.jsonl"]["sha256"] != manifest.preparation.protected_heldout_registry_sha256:
        raise ValueError("Protected registry differs from manifest")
    identities["manifest"] = file_identity(manifest_path)
    return identities


def select_tiny(prepared, games_path, manifest, identities):
    """First two audited train references per phase/color in table order: <=12.

    Select membership before annotation; failures never trigger replacements.
    Color comes from the actual rule position and is checked again on replay.
    """
    selected, buckets = [], {}
    with (prepared / "samples.jsonl").open() as handle:
        for raw in handle:
            ref = load_json(raw)
            if ref["split"] != "train":
                continue
            bucket = (ref["phase"], ref["rule_position"].split()[1])
            if bucket[0] not in PHASES or bucket[1] not in ("w", "b"):
                raise ValueError("Invalid audited phase/color")
            if buckets.get(bucket, 0) < 2:
                selected.append(ref)
                buckets[bucket] = buckets.get(bucket, 0) + 1
            if len(selected) == 12:
                break
    if not selected:
        raise ValueError("No training references available")
    wanted = {r["game_id"] for r in selected}
    protected = {r["rule_position"] for r in selected}
    with (prepared / "protected-heldout.jsonl").open() as handle:
        if any(load_json(raw)["rule_position"] in protected for raw in handle):
            raise ValueError("Selected training position is protected")
    with (prepared / "heldout-games.jsonl").open() as handle:
        if any(load_json(raw)["game_id"] in wanted for raw in handle):
            raise ValueError("Selected game is held out")
    games = {}
    with games_path.open() as handle:
        for raw in handle:
            game = load_json(raw)
            if game["game_id"] in wanted:
                if game["game_id"] in games:
                    raise ValueError("Duplicate canonical game")
                validate_game(game, manifest)
                games[game["game_id"]] = game
    if set(games) != wanted:
        raise ValueError("Missing canonical game")
    # Guard changes during selection; inputs remain read-only.
    for name in ("samples.jsonl", "protected-heldout.jsonl", "heldout-games.jsonl"):
        if file_identity(prepared / name) != identities[name]:
            raise ValueError("Reference/registry changed during selection")
    if file_identity(games_path) != identities["games.jsonl"]:
        raise ValueError("Canonical games changed during selection")
    for ref in selected:
        replay_root(ref, games[ref["game_id"]], manifest)
    return selected, games


def replay_root(ref, game, manifest):
    validate_game(game, manifest)
    if (set(ref) != {"record_version", "game_id", "split", "sample_ply", "phase", "rule_position"}
            or ref["record_version"] != RECORD_VERSION or ref["game_id"] != game["game_id"]
            or ref["split"] != game["split"] or ref["split"] != "train"
            or type(ref["sample_ply"]) is not int
            or not 0 <= ref["sample_ply"] <= len(game["moves_uci"])):
        raise ValueError("Replay reference mismatch")
    state = make_state(game["start_fen"], game["moves_uci"][:ref["sample_ply"]], history_complete=True)
    if terminal_result(state) is not None:
        raise ValueError("Terminal annotation root")
    if rule_position(state.board) != ref["rule_position"] or phase(state.board, ref["sample_ply"]) != ref["phase"]:
        raise ValueError("Replayed root identity/phase mismatch")
    return state


def replay_identity(state):
    """Full-history identity, including counters, beyond the overlap identity."""
    payload = dict(start_fen=state.board.root().fen(en_passant="fen"),
        moves_uci=[m.uci() for m in state.board.move_stack], history_complete=state.history_complete)
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def serialize_info(value):
    """Lossless transport fields for supported python-chess INFO_ALL types."""
    if isinstance(value, chess.Move):
        return value.uci()
    if isinstance(value, chess.engine.PovWdl):
        return dict(relative=list(value.relative), turn="white" if value.turn else "black")
    if isinstance(value, chess.engine.PovScore):
        return dict(relative_cp=value.relative.score(), relative_mate=value.relative.mate(),
                    turn="white" if value.turn else "black", text=str(value.relative))
    if isinstance(value, dict):
        return {str(k): serialize_info(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_info(v) for v in value]
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    return dict(invalid_type=type(value).__name__, representation=repr(value))


def validate_response(state, info, bestmove):
    """Validate a completed principal row and separate UCI bestmove."""
    if not isinstance(info, dict) or type(info.get("multipv")) is not int or info["multipv"] != 1:
        raise ValueError("Missing/invalid principal response")
    if not isinstance(info.get("wdl"), chess.engine.PovWdl) or type(info["wdl"].turn) is not bool:
        raise ValueError("Missing/invalid WDL")
    counts = list(info["wdl"].pov(state.board.turn))
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != 1000:
        raise ValueError("Invalid WDL counts")
    pv = info.get("pv")
    if not isinstance(pv, list) or not pv:
        raise ValueError("Missing/invalid PV")
    board = state.board.copy(stack=True)
    for move in pv:
        if not isinstance(move, chess.Move) or not move or move not in board.legal_moves:
            raise ValueError("Illegal PV move")
        board.push(move)
    if not isinstance(bestmove, chess.Move) or not bestmove or bestmove not in state.board.legal_moves or bestmove != pv[0]:
        raise ValueError("Missing/invalid/mismatched bestmove")
    if not isinstance(info.get("score"), chess.engine.PovScore) or type(info["score"].turn) is not bool:
        raise ValueError("Missing/invalid score")
    score = info["score"].relative
    if (score.score() is None) == (score.mate() is None) or any(
            value is not None and type(value) is not int for value in (score.score(), score.mate())):
        raise ValueError("Invalid score/mate value")
    for key in ("depth", "seldepth", "nodes", "nps", "hashfull", "tbhits"):
        if type(info.get(key)) is not int or info[key] < 0:
            raise ValueError(f"Missing/invalid search field: {key}")
    if info["depth"] == 0 or info["nodes"] == 0 or info["tbhits"] != 0:
        raise ValueError("Invalid completed search or tablebase hits")
    if type(info.get("time")) not in (int, float) or not math.isfinite(info["time"]) or info["time"] < 0:
        raise ValueError("Missing/invalid search time")
    move_id = move_to_id(state, bestmove)
    if id_to_move(state, move_id) != bestmove:
        raise ValueError("Bestmove ID round trip mismatch")
    return dict(best_move_uci=bestmove.uci(), best_move_id=move_id,
        teacher_wdl_counts=counts, teacher_wdl=[n / 1000 for n in counts],
        returned_search=serialize_info(info), returned_bestmove_uci=bestmove.uci())


ALTERNATE_BUILD_FIELDS = ("release_tag", "source_commit", "source_archive_sha256")


def alternate_build_record(manifest, evidence, executable_sha256):
    """Provenance for an attested build of the pinned source other than the manifest's binary.

    Returns None for the manifest's reference binary. An alternate build must
    attest the same release tag, source commit and source archive; embedded
    networks are checked by the caller. Attestation establishes provenance
    only; label equivalence is demonstrated separately on overlapping references.
    """
    if executable_sha256 == manifest.engine.executable_sha256:
        return None
    attested = evidence.get("alternate_build")
    expected = dict(release_tag=manifest.engine.release_tag, source_commit=manifest.engine.source_commit,
                    source_archive_sha256=manifest.engine.package_sha256)
    if (not isinstance(attested, dict) or set(attested) < set(ALTERNATE_BUILD_FIELDS)
            or any(attested[k] != v for k, v in expected.items())):
        raise ValueError("Teacher executable differs from the manifest and carries no matching source attestation")
    return dict(reference_executable_sha256=manifest.engine.executable_sha256, **expected,
                platform=str(attested.get("platform", "")), build=str(attested.get("build", "")))


def verify_teacher_bytes(manifest, executable, build_evidence):
    if importlib.metadata.version("python-chess") != "1.999" or chess.__version__ != "1.11.2":
        raise ValueError("Pinned python-chess transport mismatch")
    identity = file_identity(executable)
    evidence = load_json(build_evidence.read_bytes())
    if evidence["executable_sha256"] != identity["sha256"]:
        raise ValueError("Teacher build evidence mismatch")
    alternate = alternate_build_record(manifest, evidence, identity["sha256"])
    binary = executable.read_bytes()
    assets = []
    for asset, saved in zip(manifest.engine.embedded_nnue, evidence["nnue"], strict=True):
        blob = binary[saved["embedded_offset"]:saved["embedded_offset"] + saved["bytes"]]
        if asset.name != saved["name"] or asset.sha256 != saved["sha256"] or hashlib.sha256(blob).hexdigest() != asset.sha256:
            raise ValueError("Embedded NNUE identity mismatch")
        assets.append(saved)
    record = dict(executable=str(executable.resolve()), executable_identity=identity,
        embedded_nnue=assets, build_evidence=file_identity(build_evidence),
        python_chess="1.999", chess=chess.__version__)
    if alternate is not None:
        record["alternate_build"] = alternate
    return record


def teacher_provenance(identities, teacher):
    """Label target provenance fields shared by annotation and the loader."""
    record = dict(manifest_sha256=identities["manifest"]["sha256"],
        games_sha256=identities["games.jsonl"]["sha256"], samples_sha256=identities["samples.jsonl"]["sha256"],
        overlap_audit_sha256=identities["overlap-audit.json"]["sha256"],
        executable_sha256=teacher["executable_identity"]["sha256"], embedded_nnue=teacher["embedded_nnue"])
    if teacher.get("alternate_build"):
        record["alternate_build"] = teacher["alternate_build"]
    return record


def configure_options(protocol, manifest):
    if protocol.id != dict(name=manifest.engine.uci_name, author=manifest.engine.uci_author):
        raise ValueError("UCI identity mismatch")
    options = dict(Threads=1, Hash=128, UCI_ShowWDL=True, UCI_LimitStrength=False, SyzygyPath="")
    for name, value in options.items():
        option = protocol.options[name]
        if option.parse(value) != value:
            raise ValueError(f"Unsupported UCI setting: {name}")
    if protocol.options["Clear Hash"].type != "button" or protocol.options["MultiPV"].parse(1) != 1:
        raise ValueError("Missing hash reset/MultiPV capability")
    for name in ("Ponder", "UCI_Chess960"):
        if protocol.options[name].parse(False) is not False:
            raise ValueError("Unsupported managed option")
    for name, asset in zip(("EvalFile", "EvalFileSmall"), manifest.engine.embedded_nnue, strict=True):
        if protocol.options[name].default != asset.name:
            raise ValueError("Embedded NNUE default mismatch")
    return options


class AttemptFailure(Exception):
    def __init__(self, message, returned=None):
        super().__init__(message)
        self.returned = returned


class EngineDiagnostics(logging.Handler):
    """Fail closed when python-chess reports malformed or unexpected UCI.

    The pinned parser can log a malformed field and keep other parsed fields.
    Capturing diagnostics prevents that partial parse becoming a valid label.
    This handler is scoped to the single sequential attempt.
    """
    def __init__(self, trace, state):
        super().__init__(logging.DEBUG)
        self.trace = trace
        self.state = state

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.trace.setdefault("transport_errors", []).append(record.getMessage())
        # Observe the pinned transport's raw stdout. Its parser can stop a PV
        # at an unrecognized token without logging an error. Validate the whole
        # Stockfish PV suffix, rather than accepting that truncated prefix.
        if record.msg == "%s: >> %s" and len(record.args) == 2:
            line = record.args[1]
            self.trace.setdefault("uci_received", []).append(line)
            tokens = line.split()
            if tokens[:1] == ["info"] and "pv" in tokens and "string" not in tokens:
                board = self.state.board.copy(stack=True)
                try:
                    for uci in tokens[tokens.index("pv") + 1:]:
                        if not chess.Move.from_uci(uci):
                            raise ValueError("Null PV move")
                        board.push_uci(uci)
                except ValueError as error:
                    self.trace.setdefault("transport_errors", []).append(f"Invalid raw PV: {line}: {error}")
        elif record.msg == "%s: << %s" and len(record.args) == 2:
            self.trace.setdefault("uci_sent", []).append(record.args[1])


async def engine_attempt(state, manifest, executable, timeout, trace):
    """Fresh engine per attempt; Clear Hash precedes every node-limited search.

    The async transport lets the deadline cover completion, and exposes the
    actual UCI bestmove. MultiPV/Ponder/Chess960 are python-chess managed.
    """
    protocol = None
    diagnostics = EngineDiagnostics(trace, state)
    logger = logging.getLogger("chess.engine")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(diagnostics)
    with tempfile.TemporaryDirectory(prefix="d05-engine-") as directory:
        try:
            _, protocol = await asyncio.wait_for(chess.engine.popen_uci(str(executable), cwd=directory), timeout)
            options = configure_options(protocol, manifest)
            trace["engine_id"] = dict(protocol.id)
            trace["option_capabilities"] = {k: asdict(v) for k, v in protocol.options.items()}
            trace["effective_options"] = {k: v.default for k, v in protocol.options.items() if v.type != "button"}
            trace["effective_options"].update(options, MultiPV=1, Ponder=False, UCI_Chess960=False)
            await asyncio.wait_for(protocol.configure(options), timeout)
            await asyncio.wait_for(protocol.configure({"Clear Hash": None}), timeout)
            await asyncio.wait_for(protocol.ping(), timeout)
            trace["hash_reset_completed"] = True
            trace["returned_rows"] = []

            async def search():
                analysis = await protocol.analysis(state.board.copy(stack=True),
                    chess.engine.Limit(nodes=manifest.teacher.nodes), multipv=1, info=chess.engine.INFO_ALL)
                final = None
                async for row in analysis:
                    trace["returned_rows"].append(serialize_info(row))
                    # Do not merge a stale WDL/PV into an incomplete final row.
                    if row.get("multipv") == 1:
                        final = row
                best = await analysis.wait()
                trace["returned_bestmove"] = serialize_info(best.move)
                trace["returned_ponder"] = serialize_info(best.ponder)
                trace["completed"] = True
                return final, best.move

            result = await asyncio.wait_for(search(), timeout)
            if trace.get("transport_errors"):
                raise AttemptFailure("Malformed/unexpected UCI response", trace["transport_errors"])
            return result
        finally:
            # Close on timeout/interruption as well; never accept stopped partials.
            try:
                if protocol is not None:
                    try:
                        if trace.get("completed"):
                            try:
                                await asyncio.wait_for(protocol.quit(), 5)
                                trace["engine_exit_code"] = protocol.returncode.result()
                            except Exception as error:
                                trace["shutdown_error"] = type(error).__name__
                    finally:
                        protocol.transport.close()
                        try:
                            await asyncio.wait_for(asyncio.shield(protocol.returncode), 5)
                        except TimeoutError:
                            trace["shutdown_wait_timeout"] = True
            finally:
                logger.removeHandler(diagnostics)
                logger.setLevel(previous_level)


class PersistentEngine:
    """Explicit reusable teacher backend for future, separately recorded runs.

    One sequential owner. Reset and readiness precede every search; incomplete,
    malformed or invalid responses discard the process before the next attempt.
    Historical fresh-process annotation and its evidence remain unchanged.
    """
    policy = "persistent-clear-hash-v1"

    def __init__(self, manifest, executable, timeout=30):
        if not 0 < timeout <= 60:
            raise ValueError("timeout must be in (0,60]")
        self.manifest, self.executable, self.timeout = manifest, Path(executable), timeout
        self.protocol = None
        self.directory = None
        self.busy = False
        self.identity = file_identity(self.executable)
        if self.identity["sha256"] != manifest.engine.executable_sha256:
            raise ValueError("teacher executable checksum mismatch")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *errors):
        await self.close()

    async def close(self):
        protocol, self.protocol = self.protocol, None
        try:
            if protocol is not None:
                try:
                    await asyncio.wait_for(protocol.quit(), 5)
                finally:
                    protocol.transport.close()
        finally:
            if self.directory is not None:
                self.directory.cleanup()
                self.directory = None

    async def attempt(self, state, trace):
        if self.busy:
            raise ValueError("persistent engine supports one sequential owner")
        self.busy = True
        logger = logging.getLogger("chess.engine")
        previous_level = logger.level
        diagnostics = EngineDiagnostics(trace, state)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(diagnostics)
        try:
            # Rehash only on startup; a stable stat identity guards each attempt.
            stat = self.executable.stat()
            current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if self.protocol is None:
                if file_identity(self.executable) != self.identity:
                    raise ValueError("teacher executable changed")
                self.stat_identity = current
                self.directory = tempfile.TemporaryDirectory(prefix="teacher-session-")
                _, self.protocol = await asyncio.wait_for(
                    chess.engine.popen_uci(str(self.executable), cwd=self.directory.name), self.timeout)
            elif current != self.stat_identity:
                raise ValueError("teacher executable changed")
            protocol = self.protocol
            options = configure_options(protocol, self.manifest)
            trace["process_policy"] = self.policy
            trace["engine_id"] = dict(protocol.id)
            trace["option_capabilities"] = {k: asdict(v) for k,v in protocol.options.items()}
            trace["effective_options"] = {k:v.default for k,v in protocol.options.items() if v.type != "button"}
            trace["effective_options"].update(options, MultiPV=1, Ponder=False, UCI_Chess960=False)
            await asyncio.wait_for(protocol.configure(options), self.timeout)
            protocol.send_line("ucinewgame")
            await asyncio.wait_for(protocol.configure({"Clear Hash": None}), self.timeout)
            await asyncio.wait_for(protocol.ping(), self.timeout)
            trace["hash_reset_completed"] = True
            trace["returned_rows"] = []
            async def search():
                analysis = await protocol.analysis(state.board.copy(stack=True),
                    chess.engine.Limit(nodes=self.manifest.teacher.nodes), multipv=1,
                    info=chess.engine.INFO_ALL, game=object())
                final = None
                async for row in analysis:
                    trace["returned_rows"].append(serialize_info(row))
                    if row.get("multipv") == 1:
                        final = row
                best = await analysis.wait()
                validate_response(state, final, best.move)
                trace["returned_bestmove"] = serialize_info(best.move)
                trace["returned_ponder"] = serialize_info(best.ponder)
                return final, best.move
            result = await asyncio.wait_for(search(), self.timeout)
            if trace.get("transport_errors"):
                raise AttemptFailure("Malformed/unexpected UCI response", trace["transport_errors"])
            await asyncio.wait_for(protocol.ping(), self.timeout)
            trace.update(completed=True, process_healthy_after_search=True)
            return result
        except BaseException:
            await self.close()
            raise
        finally:
            logger.removeHandler(diagnostics)
            logger.setLevel(previous_level)
            self.busy = False


class AnnotationWorker:
    """Single-run accounting; sink receives every attempt before a label.

    An interruption is recorded, then propagated. Callers must mark the run
    interrupted; they may not reinterpret surviving partial output as success.
    """
    def __init__(self, manifest, provenance, attempt, sink, *, max_attempts=2, root_replayer=replay_root):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be 1..3")
        manifest.require_annotation_ready()
        self.manifest, self.provenance = manifest, provenance
        self.attempt, self.sink, self.max_attempts = attempt, sink, max_attempts
        # D08 supplies an audited three-split replayer; D05 stays train-only.
        if not callable(root_replayer):
            raise ValueError("root_replayer must be callable")
        self.root_replayer = root_replayer
        self.seen = set()

    async def annotate(self, ref, game):
        key = (ref["game_id"], ref["sample_ply"])
        if key in self.seen:
            raise ValueError("Duplicate reference in run")
        self.seen.add(key)
        for number in range(1, self.max_attempts + 1):
            record = dict(reference=ref, attempt=number, started_at_utc=utc_now(),
                requested_settings=asdict(self.manifest.teacher), returned={}, status="failed")
            try:
                state = self.root_replayer(ref, game, self.manifest)
                before = replay_identity(state)
                info, best = await self.attempt(state, record["returned"])
                record["returned"]["principal_response"] = serialize_info(info)
                record["returned"]["bestmove"] = serialize_info(best)
                if replay_identity(state) != before:
                    raise ValueError("Engine attempt mutated authoritative replay")
                if (record["returned"].get("completed") is not True
                        or record["returned"].get("hash_reset_completed") is not True):
                    raise ValueError("Incomplete analysis or hash reset")
                target = validate_response(state, info, best)
                label = dict(record_version="stockfish-label-v1", **{k: v for k, v in ref.items() if k != "record_version"},
                    root_fen=state.board.fen(en_passant="fen"), history_complete=True,
                    available_snapshots=min(8, ref["sample_ply"] + 1), replay_sha256=before,
                    dataset_id=self.manifest.dataset_id, encoding_version=self.manifest.encoding_version,
                    move_vocab_version=self.manifest.move_vocab_version, ending_policy=self.manifest.ending_policy,
                    wdl_order=list(self.manifest.wdl_order), teacher_config_id=self.manifest.teacher.config_id,
                    requested_settings=record["requested_settings"], successful_attempt=number,
                    target_provenance=dict(kind="stockfish-root-wdl-and-pv", **self.provenance), **target)
                record["status"] = "success"
            except BaseException as error:
                record["error"] = dict(type=type(error).__name__, message=str(error))
                if isinstance(error, AttemptFailure):
                    record["returned"]["failure_response"] = serialize_info(error.returned)
                interrupted = not isinstance(error, Exception)
                record["status"] = "interrupted" if interrupted else "failed"
                record["finished_at_utc"] = utc_now()
                self.sink(record)
                if interrupted:
                    raise
            else:
                record["finished_at_utc"] = utc_now()
                self.sink(record)
                return label
        return None
