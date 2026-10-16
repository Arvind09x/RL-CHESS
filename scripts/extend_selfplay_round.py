"""Extend a recorded two-half self-play round by further halves, then record the training buffer.

choose records once which extension halves count (a prefix, all-or-nothing by one deadline).
pointer records once the replay buffer training reads, falling back to the core choice.
finished exits 0 if every game of a campaign finished by the deadline, else 1.
disk exits 0 if the projected space for the next half and its buffer fits the quota, 10 if not
(including when disk use could not be measured completely).
See chess_rl.selfplay_rounds for the rules.
"""
import argparse
import json
from pathlib import Path
import sys

from chess_rl import selfplay_rounds as rounds
from chess_rl.selfplay_campaign import finished_by

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_selfplay_campaign import epoch  # noqa: E402

SHORT = 10


def skipped(text):
    half, separator, reason = text.partition("=")
    if not separator or not reason:
        raise argparse.ArgumentTypeError("use DIRECTORY=REASON")
    return Path(half), reason


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("choose", help="record once which extension halves count")
    choose.add_argument("--output", type=Path, required=True)
    choose.add_argument("--core", type=Path, required=True, help="the core round's choice record")
    choose.add_argument("--half", type=Path, action="append", default=[], help="extension half, in order")
    choose.add_argument("--buffer", type=Path, action="append", default=[],
                        help="buffer for the core plus halves up to the matching --half")
    choose.add_argument("--skipped", type=skipped, action="append", default=[], help="DIRECTORY=REASON")
    choose.add_argument("--deadline", type=epoch, required=True)
    pointer = commands.add_parser("pointer", help="record once the buffer training reads")
    pointer.add_argument("--output", type=Path, required=True)
    pointer.add_argument("--core", type=Path, required=True)
    pointer.add_argument("--extension", type=Path, help="extension choice record, if one was made")
    pointer.add_argument("--note", help="why the extension did not run or did not load, if known")
    finished = commands.add_parser("finished", help="exit 0 if every game finished by the deadline")
    finished.add_argument("--directory", type=Path, required=True)
    finished.add_argument("--deadline", type=epoch, required=True)
    disk = commands.add_parser("disk", help="exit 0 if the next half and its buffer fit, 10 if not")
    disk.add_argument("--core", type=Path, required=True)
    disk.add_argument("--play", type=Path, action="append", required=True, help="half still to play (or resume)")
    disk.add_argument("--buffer", type=Path, required=True, help="buffer still to build")
    disk.add_argument("--buffer-halves", type=int, required=True)
    disk.add_argument("--limit-bytes", type=int, required=True, help="quota limit for the du fallback")
    args = parser.parse_args()
    code = 0
    if args.command == "choose":
        result = rounds.choose_prefix(args.output, core=args.core, halves=args.half, buffers=args.buffer,
                                      deadline=args.deadline, skipped=args.skipped)
    elif args.command == "pointer":
        result = rounds.write_training_pointer(args.output, core=args.core, extension=args.extension, note=args.note)
    elif args.command == "finished":
        result = dict(directory=str(args.directory), finished_by_deadline=finished_by(args.directory, args.deadline))
        code = 0 if result["finished_by_deadline"] else 1
    else:
        core = json.loads(args.core.read_text())
        try:
            need = rounds.project_need(core_halves=core["halves"], core_buffer=core["buffer"], play=args.play,
                                       buffer=args.buffer, buffer_halves=args.buffer_halves)
            free = rounds.free_quota(args.limit_bytes)
            result = dict(need=need, quota=free, fits=free["free_bytes"] >= need["need_bytes"])
        except rounds.DiskUnmeasured as error:  # fail closed: an incomplete scan cannot approve a half
            result = dict(fits=False, unmeasured=str(error))
        code = 0 if result["fits"] else SHORT
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
