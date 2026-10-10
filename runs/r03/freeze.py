"""Freeze bounded R03 choices before any mixed optimizer execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path('/Users/arvindsrinivass/Desktop/RL-Chess')

def identity(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd).decode()

def main():
    from chess_rl.replay import ReplayBuffer
    from chess_rl.evaluation.matches import digest
    plan_path = ROOT / 'runs/r03/frozen-plan.json'
    if plan_path.exists():
        raise ValueError('frozen plan already exists')
    replay = ROOT / 'runs/r02-integration/fresh-replay.sqlite'
    export = ReplayBuffer(replay).export()
    inputs = [ROOT / 'data/manifests/elite-2025-01-d04-v1.json',
        ROOT / 'runs/d01/stockfish-provenance.json', replay,
        Path('/private/tmp/rl-chess-d03-prepared/games.jsonl'),
        Path('/private/tmp/rl-chess-d03-prepared/split-audit.json')]
    inputs += [ROOT / 'runs/d05/tiny-final' / name for name in
        ('selected.jsonl', 'labels.jsonl', 'attempts.jsonl', 'exclusions.jsonl', 'settings.json', 'summary.json')]
    inputs += [Path('/private/tmp/rl-chess-d04-prepared') / name for name in
        ('samples.jsonl', 'exclusions.jsonl', 'overlap-audit.json', 'protected-heldout.jsonl', 'heldout-games.jsonl')]
    plan = dict(task='R03 only; stop pending independent review before R04',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), base_commit=git('rev-parse', 'HEAD').strip(),
        branch=git('branch', '--show-current').strip(), supervised_fraction=0.5,
        supervised_source=dict(dataset_id='lichess-elite-2025-01-pilot-v1',
            bundle='runs/d05/tiny-final', loader='audited D07', split='train', positions=12,
            identity='D07 verified artifact/bundle/teacher identities plus split; full target membership hash'),
        replay=dict(path='runs/r02-integration/fresh-replay.sqlite', export_sha256=digest(export),
            sample_ids=[s['sample_id'] for s in export['samples']], positions=len(export['samples']),
            schedule='fixed audited membership for this update cycle; no new games, admissions or eviction'),
        optimizer=dict(kind='Adam', lr=1e-5, betas=[0.9,0.999], eps=1e-8,
            weight_decay=0.0, amsgrad=False, foreach=False, fused=False),
        conservative_reference=dict(T01_lr=1e-4, T03_lr=3e-4),
        loss_weights=dict(policy=1.0, wdl=1.0), batch_size=4, supervised_per_batch=2, selfplay_per_batch=2,
        sampling='independent shuffled cyclic permutations, no replacement within each source epoch; carry tails across boundaries; dedicated Python Random seed 20261010',
        update_schedule='one optimizer update per full 4-sample mixed batch; constant LR; no scheduler, accumulation, clipping or prefetch',
        fixture_schedule='three updates from scratch; separate one-update checkpoint followed by two updates in a fresh CPU process; no games generated',
        model=dict(d_policy=128, d_value=128, dropout=0.1, dtype='float32', device='cpu', relative_enabled=True),
        initialization='seeded M06 plumbing model with new Adam; never immutable pretrained P',
        seed=20261010, runtime=dict(threads=1, interop_threads=1, deterministic=True),
        inputs={str(p):identity(p) for p in inputs},
        primary=dict(head=git('rev-parse','HEAD',cwd=PRIMARY).strip(),
            status=git('status','--porcelain=v1',cwd=PRIMARY),
            diff=git('diff','--binary',cwd=PRIMARY), staged=git('diff','--cached','--binary',cwd=PRIMARY),
            coding_plan=identity(PRIMARY/'coding-plan.md')))
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:plan[k] for k in ('base_commit','branch','supervised_fraction','batch_size','optimizer','replay')}))

if __name__ == '__main__':
    main()
