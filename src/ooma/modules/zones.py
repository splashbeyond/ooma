"""Support/resistance zones from swing pivots. Strictly causal.

A swing pivot (highest high / lowest low within +-L bars) is only KNOWN L bars after it forms, so it
enters the zone map L bars late. At each bar, recent confirmed pivots are clustered: prices within
k x ATR of each other merge into one zone; >= 2 pivots make a zone ("touches" = pivots in it).
Output per bar: ATR-distance to the nearest zone below (support) and above (resistance) the close.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pivot_arrays(high: np.ndarray, low: np.ndarray, L: int):
    """Returns (idx, conf_idx, price) of all pivots sorted by idx. Needs n >= 2L+1."""
    n = len(high)
    if n < 2 * L + 1:
        return np.array([], int), np.array([], int), np.array([])
    wh, wl = sliding_window_view(high, 2 * L + 1), sliding_window_view(low, 2 * L + 1)
    ph = np.nonzero(wh[:, L] == wh.max(axis=1))[0] + L
    pl_ = np.nonzero(wl[:, L] == wl.min(axis=1))[0] + L
    idx = np.concatenate([ph, pl_])
    price = np.concatenate([high[ph], low[pl_]])
    o = np.argsort(idx, kind="stable")
    return idx[o], idx[o] + L, price[o]


def zone_columns(high, low, close, atr, L: int = 5, k: float = 0.5, window: int = 300) -> dict:
    high, low, close, atr = (np.asarray(x, dtype=float) for x in (high, low, close, atr))
    n = len(close)
    idx, conf, price = pivot_arrays(high, low, L)
    out = {name: np.full(n, np.nan) for name in ("sup_dist", "res_dist", "sup_touches", "res_touches")}
    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        hi = np.searchsorted(conf, i, side="right")            # pivots confirmed by bar i
        lo = np.searchsorted(idx, i - window, side="left")
        if hi - lo < 2:
            continue
        s = np.sort(price[lo:hi])
        groups = np.split(s, np.nonzero(np.diff(s) > k * atr[i])[0] + 1)
        best_s = best_r = None
        for g in groups:
            if len(g) < 2:
                continue
            c = g.mean()
            if c < close[i] and (best_s is None or c > best_s[0]):
                best_s = (c, len(g))
            elif c >= close[i] and (best_r is None or c < best_r[0]):
                best_r = (c, len(g))
        if best_s:
            out["sup_dist"][i], out["sup_touches"][i] = (close[i] - best_s[0]) / atr[i], best_s[1]
        if best_r:
            out["res_dist"][i], out["res_touches"][i] = (best_r[0] - close[i]) / atr[i], best_r[1]
    return out
