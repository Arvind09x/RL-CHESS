"""D02 source mainlines only: strict PGN intake and shared-adapter replay.

No game identity, consolidation, splitting, sampling or teacher targets live here.
Byte offsets refer to uncompressed member bytes; ordinals count framed sources,
including rejected ones. A frame with lost boundaries is never split by guessing.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import chess
import chess.pgn

from chess_rl.adapter import make_state
from chess_rl.contracts import AdapterConfig
from chess_rl.data_manifest import DatasetManifest

REPLAY_VERSION = "pgn-mainline-presplit-v1"
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
TAG = re.compile(r'\[([A-Za-z0-9][A-Za-z0-9_+#=:-]*)\s+"((?:[^"\\\r\n]|\\["\\])*)"\]\s*')
TOKEN = re.compile(
    r"(?P<result>1/2-1/2|1-0|0-1|\*)|(?P<number>\d+\.(?:\.\.)?)|"
    r"(?P<nag>\$\d+|[!?]{1,2})|(?P<paren>[()])|"
    r"(?P<move>(?:[NBKRQ]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[NBRQ])?"
    r"|O-O(?:-O)?|0-0(?:-0)?|--|Z0|0000|@@@@)[+#]?)"
)


class SourceError(ValueError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


@dataclass(frozen=True)
class SourceFrame:
    raw: bytes
    byte_start: int
    byte_end: int
    line_start: int
    line_end: int


def iter_frames(handle):
    """Stream binary PGN frames; blank lines end movetext outside brace comments.

    Headers beginning after movetext also delimit sources. Unclosed brace
    comments consume the remaining member as one rejected span: subsequent
    source-game count is unknown, rather than an invented ordinal sequence.
    """
    lines = []
    offset = 0
    in_comment = False
    body = False
    start = start_line = 0
    last_line = 0
    for line_no, raw in enumerate(handle, 1):
        stripped = raw.strip().lstrip(b"\xef\xbb\xbf")
        if lines and body and not in_comment and (not stripped or stripped.startswith(b"[")):
            yield SourceFrame(b"".join(lines), start, offset, start_line, last_line)
            lines, body = [], False
        if not lines and not stripped:
            offset += len(raw)
            continue
        if not lines:
            start, start_line = offset, line_no
        lines.append(raw)
        last_line = line_no
        # Headers may contain literal braces. Semicolon/percent comments do not.
        if in_comment or not stripped.startswith((b"[", b"%")):
            if stripped and not stripped.startswith((b";", b"%")):
                body = True
            for fragment in re.finditer(rb"[{};]", raw):
                char = fragment.group()
                if char == b";" and not in_comment:
                    break
                if char == b"{":
                    in_comment = True
                elif char == b"}":
                    in_comment = False
        offset += len(raw)
    if lines:
        yield SourceFrame(b"".join(lines), start, offset, start_line, last_line)


def _headers_and_body(text):
    headers = {}
    lines = text.lstrip("\ufeff").splitlines(keepends=True)
    consumed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or line.startswith((";", "%")):
            consumed += len(line)
            continue
        if not stripped.startswith("["):
            break
        match = TAG.fullmatch(stripped)
        if not match or match[1] in headers:
            raise SourceError("headers", "Malformed or duplicate PGN tag")
        headers[match[1]] = re.sub(r'\\(["\\])', r'\1', match[2])
        consumed += len(line)
    return headers, text.lstrip("\ufeff")[consumed:]


def _check_movetext(body):
    """Account for every byte/token the forgiving python-chess lexer might skip."""
    pos = depth = 0
    result = None
    mainline = []
    numbers = []
    context_moves = [0]
    while pos < len(body):
        char = body[pos]
        if char.isspace():
            pos += 1
            continue
        if char == "{" or char == ";" or (char == "%" and (pos == 0 or body[pos-1] == "\n")):
            end = body.find("}" if char == "{" else "\n", pos + 1)
            if end < 0:
                if char == "{":
                    raise SourceError("unclosed_comment", "Unclosed brace comment; remaining source boundaries unknown")
                break
            pos = end + 1
            continue
        match = TOKEN.match(body, pos)
        if not match:
            raise SourceError("syntax", f"Unrecognized movetext at character {pos}: {body[pos:pos+40]!r}")
        kind, token = match.lastgroup, match.group()
        if result is not None:
            raise SourceError("syntax", "Movetext after the mainline result marker")
        if kind == "paren":
            if token == "(":
                if not context_moves[-1]:
                    raise SourceError("syntax", "Variation without a preceding move")
                context_moves.append(0)
            elif depth > 0:
                if not context_moves.pop():
                    raise SourceError("syntax", "Empty variation")
            depth += 1 if token == "(" else -1
            if depth < 0:
                raise SourceError("syntax", "Unmatched variation close")
        elif kind == "result":
            if depth:
                raise SourceError("syntax", "Result marker inside variation")
            result = token
        elif kind == "move" and token in {"--", "Z0", "0000", "@@@@"}:
            raise SourceError("null_move", "Null moves are not standard source moves")
        elif kind == "move":
            context_moves[-1] += 1
            if not depth:
                mainline.append(token)
        elif kind == "nag" and not context_moves[-1]:
            raise SourceError("syntax", "Annotation glyph without a preceding move")
        elif kind == "number" and not depth:
            numbers.append((len(mainline), int(token.split(".")[0]), token.endswith("...")))
        pos = match.end()
    if depth:
        raise SourceError("syntax", "Unclosed variation")
    if result is None:
        raise SourceError("missing_result_marker", "Missing explicit source termination marker; no result or moves inferred")
    if any(ply == len(mainline) for ply, _, _ in numbers):
        raise SourceError("syntax", "Move number without a subsequent mainline move")
    return result, mainline, numbers


class _QuietBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error):
        # Preserve errors without python-chess logging entire partial games.
        self.game.errors.append(error)


def parse_mainline(text):
    """Return genuine root, full UCI mainline, original headers and result marker.

    Legal unfinished sources ending '*' are retained with explicit metadata.
    A declared setup is complete only from that genuine source root; no earlier
    history is inferred. Result metadata never supplies an adapter/teacher label.
    """
    headers, body = _headers_and_body(text)
    variant = headers.get("Variant", "Standard")
    if variant.lower() not in {"standard", "chess", "normal"}:
        raise SourceError("variant", f"Unsupported Variant: {variant!r}")
    if headers.get("SetUp", "0") not in {"0", "1"}:
        raise SourceError("setup", "SetUp must be 0 or 1")
    setup = headers.get("SetUp") == "1"
    if setup != ("FEN" in headers):
        raise SourceError("setup", "A declared setup requires both SetUp=1 and FEN")
    start = headers.get("FEN", chess.STARTING_FEN)
    if len(start.split()) != 6:
        raise SourceError("setup", "Starting FEN must have all six fields")
    placement, turn, rights, ep, halfmove, fullmove = start.split()
    if (not re.fullmatch(r"[pnbrqkPNBRQK1-8/]+", placement)
            or turn not in {"w", "b"}
            or not re.fullmatch(r"-|[a-h][36]", ep)
            or not re.fullmatch(r"\d+", halfmove)
            or not re.fullmatch(r"[1-9]\d*", fullmove)):
        raise SourceError("setup", "Malformed standard six-field FEN")
    try:
        root = chess.Board(start)
        if not root.is_valid() or root.chess960:
            raise ValueError(f"invalid board status={root.status()}")
        # X-FEN/Shredder castling aliases are not declarations of standard chess.
        if not re.fullmatch(r"-|K?Q?k?q?", rights):
            raise ValueError("nonstandard castling field")
    except ValueError as exc:
        raise SourceError("setup", str(exc)) from exc
    marker, tokens, numbers = _check_movetext(body)
    for ply, number, black in numbers:
        offset = ply + int(root.turn == chess.BLACK)
        if number != root.fullmove_number + offset // 2 or black != bool(offset % 2):
            raise SourceError("move_number", f"Move number/side disagrees with source root at ply {ply+1}")
    if "Result" in headers and headers["Result"] not in RESULTS:
        raise SourceError("result", "Invalid declared Result")
    try:
        game = chess.pgn.read_game(io.StringIO(text), Visitor=_QuietBuilder)
    except (ValueError, IndexError) as exc:
        raise SourceError("parser_error", str(exc)) from exc
    if game is None or game.errors:
        raise SourceError("parser_error", "; ".join(str(e) for e in game.errors) if game else "No parsed game")
    moves = tuple(move.uci() for move in game.mainline_moves())
    if len(moves) != len(tokens):
        raise SourceError("parser_error", "Parsed mainline length differs from fully accounted source move tokens")
    try:
        state = make_state(root.fen(en_passant="fen"), moves, history_complete=True)
    except ValueError as exc:
        raise SourceError("replay_error", str(exc)) from exc
    if (type(game.board()) is not chess.Board or game.board().chess960
            or game.board().fen(en_passant="fen") != state.initial_fen):
        raise SourceError("setup", "Parser root differs from validated standard source root")
    return state, headers, marker


def source_record(frame, *, archive_sha256, member, member_index, game_ordinal):
    return {
        "archive_sha256": archive_sha256, "member": member,
        "member_index": member_index, "game_ordinal": game_ordinal,
        "byte_start": frame.byte_start, "byte_end": frame.byte_end,
        "line_start": frame.line_start, "line_end": frame.line_end,
        "source_sha256": hashlib.sha256(frame.raw).hexdigest(),
    }


def replay_record(frame, source, manifest):
    try:
        text = frame.raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceError("encoding", str(exc)) from exc
    state, headers, marker = parse_mainline(text)
    return {
        "record_version": REPLAY_VERSION, "dataset_id": manifest.dataset_id,
        "encoding_version": manifest.encoding_version,
        "move_vocab_version": manifest.move_vocab_version,
        "ending_policy": manifest.ending_policy, "wdl_order": list(manifest.wdl_order),
        "start_fen": state.initial_fen,
        "moves_uci": [m.uci() for m in state.board.move_stack],
        "history_complete": True, "history_origin": "declared-pgn-start",
        "declared_setup": headers.get("SetUp") == "1",
        "source": source, "pgn_headers": headers,
        "pgn_result": {
            "header": headers.get("Result"), "movetext": marker,
            "unfinished": marker == "*",
            "discrepancy": "Result" in headers and headers["Result"] != marker,
        },
    }


def replay_state(record):
    """Reconstruct an accepted pre-split record through the public adapter."""
    if (record.get("record_version") != REPLAY_VERSION
            or record.get("history_complete") is not True
            or record.get("history_origin") != "declared-pgn-start"):
        raise ValueError("Unsupported replay version/history provenance")
    config = AdapterConfig()
    for key in ("encoding_version", "move_vocab_version", "ending_policy", "wdl_order"):
        expected = getattr(config, key)
        if record.get(key) != (list(expected) if key == "wdl_order" else expected):
            raise ValueError(f"Unsupported {key}")
    return make_state(record["start_fen"], record["moves_uci"], history_complete=True)


def json_line(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def file_identity(path):
    with Path(path).open("rb") as handle:
        return {"sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                "bytes": Path(path).stat().st_size}


def prepare_archive(archive, manifest, output_dir, *, progress=None):
    """Validate exact archive bytes, stream all PGN members, save source accounting.

    Outputs are exclusive-created in a new directory. ZIP/decompression errors
    abort the run (partial outputs are not completion); never silently skip a
    member. Non-PGN members are identified, hashed and reported separately.
    """
    identity = file_identity(archive)
    if (identity["sha256"] != manifest.archive.downloaded_sha256
            or identity["bytes"] != manifest.archive.downloaded_bytes):
        raise ValueError("Archive bytes differ from manifest retrieval provenance")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    counts = Counter()
    reasons = Counter()
    members = []
    with zipfile.ZipFile(archive) as zipped, (out / "replays.jsonl").open("w", encoding="utf-8", newline="\n") as accepted, (out / "rejections.jsonl").open("w", encoding="utf-8", newline="\n") as rejected:
        for index, info in enumerate(zipped.infolist(), 1):
            member = {"member_index": index, "name": info.filename,
                      "bytes": info.file_size, "compressed_bytes": info.compress_size,
                      "crc32": f"{info.CRC:08x}", "compression": info.compress_type,
                      "zip_timestamp": list(info.date_time), "header_offset": info.header_offset,
                      "pgn": not info.is_dir() and info.filename.lower().endswith(".pgn")}
            digest = hashlib.sha256()
            member_counts = Counter()
            with zipped.open(info) as raw:
                if member["pgn"]:
                    # Hash all uncompressed bytes, including inter-game whitespace.
                    def hashed_lines():
                        for line in raw:
                            digest.update(line)
                            yield line
                    for ordinal, frame in enumerate(iter_frames(hashed_lines()), 1):
                        source = source_record(frame, archive_sha256=identity["sha256"],
                                               member=info.filename, member_index=index, game_ordinal=ordinal)
                        member_counts["games"] += 1
                        try:
                            record = replay_record(frame, source, manifest)
                        except SourceError as exc:
                            member_counts["rejected"] += 1
                            reasons[exc.reason] += 1
                            rejected.write(json_line({"source": source, "reason": exc.reason, "detail": str(exc),
                                                       "remaining_boundaries_unknown": exc.reason == "unclosed_comment"}))
                        else:
                            member_counts["accepted"] += 1
                            member_counts["plies"] += len(record["moves_uci"])
                            member_counts["unfinished"] += record["pgn_result"]["unfinished"]
                            member_counts["result_discrepancies"] += record["pgn_result"]["discrepancy"]
                            member_counts["declared_setups"] += record["declared_setup"]
                            accepted.write(json_line(record))
                        if progress and ordinal % 10000 == 0:
                            progress(info.filename, ordinal, dict(member_counts))
                else:
                    for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                        digest.update(chunk)
            member["sha256"] = digest.hexdigest()
            member["counts"] = {name: member_counts[name] for name in
                                ("games", "accepted", "rejected", "plies", "unfinished", "result_discrepancies", "declared_setups")}
            members.append(member)
            counts.update(member_counts)
    report = {
        "record_version": REPLAY_VERSION, "dataset_id": manifest.dataset_id,
        "archive": identity, "archive_count": 1, "member_count": len(members),
        "pgn_member_count": sum(m["pgn"] for m in members), "members": members,
        "counts": {name: counts[name] for name in
                   ("games", "accepted", "rejected", "plies", "unfinished", "result_discrepancies", "declared_setups")},
        "rejection_reasons": dict(sorted(reasons.items())),
        "source_game_count_exact": not reasons["unclosed_comment"],
        "outputs": {name: file_identity(out / name) for name in ("replays.jsonl", "rejections.jsonl")},
        "annotation_blockers": list(manifest.annotation_blockers()),
        "measurement_scope": "source frames and legal mainline replays before D03 duplicate consolidation or split assignment",
        "complete": True,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
