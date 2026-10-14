"""Per-update timing from a supervised run's metrics.jsonl.

The runner stamps an update's elapsed_seconds before its validation and checkpoint run, so the
interval into the following update carries that cost. Only intervals between consecutive update
records are used; the baseline record is stamped after its own validation. Steady-state update time uses only intervals whose
previous record has no validation; the validation+checkpoint overhead is estimated as the excess of
the other intervals over the median update and is labelled an estimate. Explicit timers in the
runner would be needed for an exact split.
"""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory containing metrics.jsonl")
    p.add_argument("--project-steps", type=int, help="project wall time for this many updates")
    p.add_argument("--project-validation-every", type=int, default=500)
    args = p.parse_args()
    records = [json.loads(line) for line in (args.run / "metrics.jsonl").read_text().splitlines() if line]
    steady, after_validation = [], []
    for previous, current in zip(records, records[1:]):
        # The baseline record is stamped after its validation, so its interval is not comparable.
        if previous["event"] != "update" or current["event"] != "update":
            continue
        dt = current["elapsed_seconds"] - previous["elapsed_seconds"]
        (after_validation if "validation" in previous else steady).append(dt)
    if len(steady) < 2:
        raise SystemExit("not enough steady-state intervals")
    steady.sort()
    median = statistics.median(steady)
    report = dict(run=str(args.run), updates=sum(r["event"] == "update" for r in records),
        steady_intervals=len(steady), update_seconds=dict(median=median, mean=statistics.fmean(steady),
            p10=steady[len(steady) // 10], p90=steady[(len(steady) * 9) // 10], min=steady[0], max=steady[-1]),
        validation_checkpoint_seconds_estimate=dict(
            count=len(after_validation),
            mean=statistics.fmean(dt - median for dt in after_validation) if after_validation else None,
            note="interval after a validated step minus median update; includes checkpoint write"))
    if args.project_steps:
        overhead = report["validation_checkpoint_seconds_estimate"]["mean"] or 0.0
        validations = args.project_steps // args.project_validation_every
        report["projection"] = dict(steps=args.project_steps, validation_every=args.project_validation_every,
            hours=(args.project_steps * median + validations * overhead) / 3600)
    (args.run / "step-timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
