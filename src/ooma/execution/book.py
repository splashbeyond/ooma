from __future__ import annotations


def walk_book(levels: list[tuple[float, float]], qty: float, beyond_bps: float = 25.0, side: int = 1):
    """Fill `qty` base units against opposite-side `levels` (best first). Returns (avg_px, filled_visible, exceeded).
    If the visible book (20 levels) is too thin, the remainder fills `beyond_bps` worse than the last level
    and `exceeded=True` so callers can flag an unrealistic size."""
    left, cost, got = qty, 0.0, 0.0
    for px, sz in levels:
        take = min(left, sz)
        cost += take * px
        got += take
        left -= take
        if left <= 1e-12:
            break
    exceeded = left > 1e-12
    if exceeded and levels:
        worst = levels[-1][0] * (1 + side * beyond_bps / 1e4)
        cost += left * worst
        got += left
    return (cost / got if got else float("nan")), min(qty, got), exceeded
