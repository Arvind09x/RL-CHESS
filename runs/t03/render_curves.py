"""Render all saved T03 metric rows; no training or checkpoint mutation."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
root = Path(__file__).resolve().parents[2]
run = root / 'runs/t03'
rows = [json.loads(line) for line in (run / 'learning-curves.jsonl').read_text().splitlines()]
result = json.loads((run / 'result.json').read_text())
# A dependency-free static review plot; all data remain in lossless JSONL/CSV.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="760" viewBox="0 0 1100 760">',
    '<rect width="1100" height="760" fill="white"/>',
    '<style>text{font:14px sans-serif;fill:#263442}.title{font-size:22px;font-weight:600}</style>',
    f'<text x="55" y="35" class="title">T03 · eight fixed D05 training labels · {result["status"]} at step {result["step"]}</text>',
    '<text x="55" y="59">Dropout 0 · Adam 0.0003 · seed 20261009 · cap 1000 · plumbing checkpoint</text>']
panels = [ ('Legal best-move accuracy', ['legal_best_move_accuracy'], 1.05, [(1., 'gate 100%')]),
    ('Teacher-WDL KL (nats)', ['teacher_wdl_kl_mean', 'teacher_wdl_kl_max'], max(r['metrics']['teacher_wdl_kl_max'] for r in rows) * 1.1, [(.01, 'mean gate .01'), (.03, 'max gate .03')]),
    ('WDL cross-entropy vs teacher entropy (nats)', ['wdl_cross_entropy', 'teacher_entropy_floor'], 1.3, []),
    ('Policy cross-entropy (nats)', ['policy_ce'], 3.5, []) ]
colors = ['#1762a3', '#d16020']
for index, (title, fields, ymax, thresholds) in enumerate(panels):
    x, y = 65 + (index % 2) * 535, 120 + (index // 2) * 310
    w, h = 440, 200
    svg.append(f'<text x="{x}" y="{y-20}">{title}</text>')
    for tick in range(5):
        value = ymax * tick / 4
        yy = y + h * (1 - value / ymax)
        svg.append(f'<path d="M{x},{yy} H{x+w}" stroke="#e4e8ec"/><text x="{x-40}" y="{yy+5}">{value:.2f}</text>')
    for step in sorted({round(result['step'] * i / 5) for i in range(6)}):
        xx = x + w * step / result['step']
        svg.append(f'<text x="{xx-7}" y="{y+h+22}">{step}</text>')
    svg.append(f'<text x="{x+w/2-20}" y="{y+h+44}">Update</text>')
    for value, label in thresholds:
        yy = y + h * (1 - value / ymax)
        svg.append(f'<path d="M{x},{yy} H{x+w}" stroke="#6b7280" stroke-dasharray="4 4"/>')
    for number, field in enumerate(fields):
        points = ' '.join(f"{x+w*r['step']/result['step']:.2f},{y+h*(1-r['metrics'][field]/ymax):.2f}" for r in rows)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[number]}" stroke-width="2.5"/>')
        label = field.replace('teacher_wdl_kl_', '').replace('_', ' ')
        svg.append(f'<text x="{x+number*190}" y="{y+h+68}" style="fill:{colors[number]}">{label}</text>')
svg.append('<text x="55" y="741">Every completed update is shown. KL gates: mean ≤ 0.01, maximum ≤ 0.03 nats. Raw per-position KL is saved.</text></svg>')
with (run / 'learning-curves.svg').open('w') as handle: handle.write('\n'.join(svg) + '\n')
ET.parse(run / 'learning-curves.svg')
