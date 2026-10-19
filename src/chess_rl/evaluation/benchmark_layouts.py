"""Explicit process/game/batch layouts for bounded development studies."""
from .execution import Execution

DEFAULT_LAYOUTS = ('5x1x1', '2x8x4', '1x16x8')


def parse_layout(value, *, batch_size=32, max_wait_ms=20.0):
    """Accept PxKxB; retain PxK with the caller's explicit shared B."""
    try:
        parts = [int(part) for part in value.split('x')]
    except (AttributeError, ValueError) as exc:
        raise ValueError('layout must be PxK or PxKxB with positive integers') from exc
    if len(parts) not in (2, 3):
        raise ValueError('layout must be PxK or PxKxB with positive integers')
    processes, games = parts[:2]
    if not 1 <= processes <= 5:
        raise ValueError('use 1 to 5 processes within the six-CPU allocation')
    batch = parts[2] if len(parts) == 3 else 1 if games == 1 else batch_size
    return Execution(mode='serial' if games == 1 else 'batched', processes=processes,
                     games_per_process=games, batch_size=batch, max_wait_ms=max_wait_ms)
