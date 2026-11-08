"""ooma_lib: causal numpy helpers available to LLM-written strategy scripts (`from ooma_lib import ema, rsi`).
Every function uses only values up to and including the last element of its input. NaN until warmed up.
Kept dependency-free (numpy only) because it also runs inside the sandbox."""
import numpy as np


def sma(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    a, v = 2.0 / (n + 1), x[0]
    for i in range(len(x)):
        v = a * x[i] + (1 - a) * v
        out[i] = v
    return out


def rsi(close, n=14):
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    g, l = np.maximum(d, 0), np.maximum(-d, 0)
    ag, al = g[:n].mean(), l[:n].mean()
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag, al = (ag * (n - 1) + g[i]) / n, (al * (n - 1) + l[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr(high, low, close, n=14):
    h, l, c = (np.asarray(v, dtype=float) for v in (high, low, close))
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return sma(tr, n)


def rolling_max(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].max()
    return out


def rolling_min(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = x[i - n + 1:i + 1].min()
    return out


def zscore(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        w = x[i - n + 1:i + 1]
        s = w.std()
        out[i] = (x[i] - w.mean()) / s if s > 0 else 0.0
    return out


def pct_change(x, n=1):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = x[n:] / x[:-n] - 1
    return out
