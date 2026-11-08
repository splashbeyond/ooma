"""Wallet behaviour profile from fills. Answers two questions:
  1. WHAT KIND of trader is this?  directional | market_maker | hedger | carry | vault | agent | sparse
  2. HOW GOOD is the directional PnL, net of fees, and is it luck?

Fills come from HL `userFillsByTime`: coin, px, sz, side (B/A), dir, closedPnl, fee, crossed (False = maker),
startPosition, time (ms), twapId. Positions are rebuilt as startPosition +- sz, so the book can be replayed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import median

import numpy as np
import polars as pl

DEFAULTS = dict(
    min_fills=60, min_days=7,
    mm_maker_share=0.60, mm_fills_per_day=200, mm_median_hold_s=300,
    hedge_net_to_gross=0.30,           # median |net|/gross exposure below this = hedged book
    spot_hedge_ratio=0.50,             # perp positions offset by spot balances in the same token
    carry_funding_share=0.40,          # share of total profit that came from funding
    min_closes=100, min_active_days=20,
    max_top5_share=0.60, max_top1_share=0.35,
    alpha=0.05,
)


def prepare(fills: list[dict]) -> pl.DataFrame:
    rows = [dict(time=int(f["time"]), coin=f["coin"], px=float(f["px"]), sz=float(f["sz"]), side=f["side"],
                 dir=f.get("dir", ""), closed_pnl=float(f.get("closedPnl", 0) or 0), fee=float(f.get("fee", 0) or 0),
                 maker=not bool(f.get("crossed", True)), start_pos=float(f.get("startPosition", 0) or 0),
                 twap=f.get("twapId") is not None) for f in fills]
    schema = dict(time=pl.Int64, coin=pl.Utf8, px=pl.Float64, sz=pl.Float64, side=pl.Utf8, dir=pl.Utf8,
                  closed_pnl=pl.Float64, fee=pl.Float64, maker=pl.Boolean, start_pos=pl.Float64, twap=pl.Boolean)
    return pl.DataFrame(rows, schema=schema).sort("time").unique(maintain_order=True)


@dataclass
class Profile:
    address: str
    kind: str = "sparse"
    n_fills: int = 0
    span_days: float = 0.0
    active_days: int = 0
    n_closes: int = 0
    fills_per_day: float = 0.0
    maker_share: float = 0.0
    twap_share: float = 0.0
    median_hold_s: float | None = None
    net_to_gross: float | None = None       # median |net|/gross exposure while in the market
    spot_hedge: float = 0.0
    funding_share: float | None = None
    notional: float = 0.0
    net_pnl: float = 0.0                     # closed PnL minus ALL fees
    pnl_bps: float = 0.0                     # net pnl per $ traded
    win_rate: float | None = None
    profit_factor: float | None = None
    daily_t: float = 0.0
    max_dd: float = 0.0                      # peak-to-trough of cumulative net pnl, in USD
    top1_share: float = 0.0
    top5_share: float = 0.0
    coins: list = field(default_factory=list)
    flags: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _positions_and_holds(df: pl.DataFrame):
    pos, last_px, open_t, holds, ratios = {}, {}, {}, [], []
    for r in df.select("time", "coin", "px", "sz", "side", "start_pos").iter_rows():
        t, coin, px, sz, side, start = r
        after = start + (sz if side == "B" else -sz)
        if abs(start) < 1e-12 < abs(after):
            open_t[coin] = t
        if abs(after) < 1e-12 and coin in open_t:
            holds.append((t - open_t.pop(coin)) / 1000)
        pos[coin], last_px[coin] = after, px
        vals = [pos[c] * last_px[c] for c in pos if abs(pos[c]) > 1e-12]
        gross = sum(abs(v) for v in vals)
        if gross > 0:
            ratios.append(abs(sum(vals)) / gross)
    return holds, ratios


def profile_wallet(address: str, df: pl.DataFrame, *, role: str = "user", spot_hedge: float = 0.0,
                   funding_pnl: float | None = None, cfg: dict | None = None) -> Profile:
    c = {**DEFAULTS, **(cfg or {})}
    p = Profile(address=address, n_fills=df.height, spot_hedge=spot_hedge)
    if role != "user":
        p.kind = "vault" if role == "vault" else "agent"
        return p
    if df.height < c["min_fills"]:
        return p
    t0, t1 = df["time"].min(), df["time"].max()
    p.span_days = (t1 - t0) / 86_400_000
    if p.span_days < c["min_days"]:
        return p
    d = df.with_columns((pl.col("px") * pl.col("sz")).alias("notional"),
                        (pl.col("closed_pnl") - pl.col("fee")).alias("net"),
                        pl.from_epoch("time", time_unit="ms").dt.date().alias("day"))
    p.notional = float(d["notional"].sum())
    p.fills_per_day = df.height / max(p.span_days, 1e-9)
    p.maker_share = float(d.filter(pl.col("maker"))["notional"].sum() / p.notional) if p.notional else 0.0
    p.twap_share = float(d["twap"].mean())
    p.coins = d["coin"].unique().to_list()
    p.net_pnl = float(d["net"].sum())
    p.pnl_bps = p.net_pnl / p.notional * 1e4 if p.notional else 0.0
    closes = d.filter(pl.col("dir").str.starts_with("Close") | (pl.col("closed_pnl") != 0))
    p.n_closes = closes.height
    cn = (closes["closed_pnl"] - closes["fee"]).to_numpy()
    if p.n_closes:
        p.win_rate = float((cn > 0).mean())
        gp, gl = cn[cn > 0].sum(), -cn[cn < 0].sum()
        p.profit_factor = float(gp / gl) if gl > 0 else None
        pos_sorted = np.sort(cn[cn > 0])[::-1]
        if gp > 0:
            p.top1_share = float(pos_sorted[:1].sum() / gp)
            p.top5_share = float(pos_sorted[:5].sum() / gp)
    daily = d.group_by("day").agg(pl.col("net").sum()).sort("day")["net"].to_numpy()
    p.active_days = len(daily)
    if len(daily) >= 3 and daily.std(ddof=1) > 0:
        p.daily_t = float(daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))))
    cum = np.cumsum(daily)
    p.max_dd = float(np.max(np.maximum.accumulate(cum) - cum)) if len(cum) else 0.0
    holds, ratios = _positions_and_holds(d)
    p.median_hold_s = float(median(holds)) if holds else None
    p.net_to_gross = float(median(ratios)) if ratios else None
    if funding_pnl is not None and (abs(p.net_pnl) + abs(funding_pnl)) > 0 and funding_pnl > 0:
        p.funding_share = float(funding_pnl / max(p.net_pnl + funding_pnl, 1e-9))

    # ---- classification: order matters, the most "not directional" labels win ----
    if (p.maker_share >= c["mm_maker_share"] and p.fills_per_day >= c["mm_fills_per_day"]) or \
       (p.median_hold_s is not None and p.median_hold_s < c["mm_median_hold_s"] and p.fills_per_day >= c["mm_fills_per_day"] / 2):
        p.kind = "market_maker"
    elif spot_hedge >= c["spot_hedge_ratio"]:
        p.kind = "hedger"; p.flags.append("perp offset by spot")
    elif p.net_to_gross is not None and p.net_to_gross < c["hedge_net_to_gross"] and len(p.coins) >= 2:
        p.kind = "hedger"; p.flags.append("book is net-flat across coins")
    elif p.funding_share is not None and p.funding_share >= c["carry_funding_share"]:
        p.kind = "carry"; p.flags.append("profit mostly from funding")
    else:
        p.kind = "directional"
    if p.top1_share > c["max_top1_share"] or p.top5_share > c["max_top5_share"]:
        p.flags.append("profit concentrated in a few trades")
    return p
