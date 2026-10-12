"""Dependency-free exports of the supervised runner's durable metric log."""
import csv
import json
from pathlib import Path


def write_learning_curves(directory):
    """Export this run segment, including its explicitly labelled initial baseline."""
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    fields = ["step", "epoch", "event", "elapsed_seconds"]
    fields += [f"train_{key}" for key in ("batch_size", "total_loss", "policy_loss", "wdl_loss", "gradient_norm",
                                          "clipped", "learning_rate")]
    fields += [f"validation_{key}" for key in ("total_loss", "policy_loss", "wdl_loss", "policy_accuracy", "positions")]
    with (directory / "learning-curves.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flattened = {key: row[key] for key in fields if key in row}
            for source in ("train", "validation"):
                flattened.update({f"{source}_{key}": value for key, value in row.get(source, {}).items()})
            writer.writerow(flattened)

    panels = [("Total loss", "total_loss", ("train", "validation")),
              ("Policy loss", "policy_loss", ("train", "validation")),
              ("WDL loss", "wdl_loss", ("train", "validation")),
              ("Validation policy accuracy", "policy_accuracy", ("validation",)),
              ("Global gradient L2 norm (before clipping)", "gradient_norm", ("train",)),
              ("Learning rate", "learning_rate", ("train",))]
    colors = {"train": "#2563eb", "validation": "#c2410c"}
    xmin, xmax = min(row["step"] for row in rows), max(row["step"] for row in rows)
    baseline = rows[0]
    label = "Untrained baseline" if baseline["event"] == "untrained_baseline" else "Resumed baseline"
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1300" viewBox="0 0 1000 1300">',
           '<rect width="1000" height="1300" fill="white"/>',
           '<g font-family="sans-serif" fill="#172033">',
           '<text x="80" y="30" font-size="20">Supervised learning curves — this run segment</text>',
           f'<text x="80" y="54" font-size="13">{label} at step {baseline["step"]}; training losses are per-update batches, validation uses the full split.</text>']
    for i, (title, key, sources) in enumerate(panels):
        top, height, left, width = 95 + 200 * i, 140, 80, 700
        series = {source: [(row["step"], row[source][key]) for row in rows if key in row.get(source, {})]
                  for source in sources}
        ymax = 1.0 if key == "policy_accuracy" else max((value for points in series.values() for _, value in points), default=1.0) * 1.05
        ymax = max(ymax, 1e-12)
        svg.append(f'<text x="{left}" y="{top - 12}" font-size="15">{title}</text>')
        for tick in range(3):
            y = top + height * (1 - tick / 2)
            svg.append(f'<path d="M {left} {y} H {left + width}" stroke="#dce1e8"/>')
            svg.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" font-size="11">{ymax * tick / 2:.3g}</text>')
        for step in sorted({xmin, (xmin + xmax) // 2, xmax}):
            x = left + width * (step - xmin) / max(1, xmax - xmin)
            svg.append(f'<text x="{x}" y="{top + height + 18}" text-anchor="middle" font-size="11">{step}</text>')
        svg.append(f'<text x="{left + width / 2}" y="{top + height + 34}" text-anchor="middle" font-size="11">Optimizer step</text>')
        for j, (source, points) in enumerate(series.items()):
            coordinates = [(left + width * (step - xmin) / max(1, xmax - xmin),
                            top + height * (1 - value / ymax)) for step, value in points]
            path = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
            color = colors[source]
            svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="1.5"/>')
            for x, y in coordinates:
                svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="{color}"/>')
            svg.append(f'<text x="810" y="{top + 16 + j * 22}" fill="{color}" font-size="13">{source}</text>')
    svg.extend(["</g>", "</svg>"])
    (directory / "learning-curves.svg").write_text("\n".join(svg) + "\n")
