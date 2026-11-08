"""Hyperliquid wallet endpoints (verified live): leaderboard (public stats file), userFillsByTime, clearinghouseState,
spotClearinghouseState, userRole, userFunding. Only the 10,000 most recent fills per wallet are retrievable."""
from __future__ import annotations

import polars as pl

LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
SPOT_ALIASES = {"BTC": ("UBTC", "BTC"), "ETH": ("UETH", "ETH"), "SOL": ("USOL", "SOL")}


class WalletAPI:
    def __init__(self, adapter):
        self.a = adapter

    def leaderboard(self) -> pl.DataFrame:
        r = self.a._c.get(LEADERBOARD, timeout=120)
        r.raise_for_status()
        rows = []
        for x in r.json()["leaderboardRows"]:
            w = {k: v for k, v in x["windowPerformances"]}
            rows.append(dict(address=x["ethAddress"], account_value=float(x["accountValue"]),
                             **{f"{m}_{f}": float(w[m][f]) for m in ("day", "week", "month", "allTime") for f in ("pnl", "roi", "vlm")}))
        return pl.DataFrame(rows)

    def fills(self, address: str, start_ms: int = 0, end_ms: int | None = None, max_fills: int = 10_000) -> list[dict]:
        import time
        end_ms = end_ms or int(time.time() * 1000)
        out, cur, seen = [], start_ms, set()
        while len(out) < max_fills:
            chunk = self.a._post({"type": "userFillsByTime", "user": address, "startTime": cur, "endTime": end_ms})
            new = [f for f in chunk if f.get("tid") not in seen]
            seen.update(f.get("tid") for f in new)
            out += new
            if len(chunk) < 2000 or not new:
                break
            cur = max(f["time"] for f in chunk) + 1
        return sorted(out, key=lambda f: f["time"])

    def state(self, address: str) -> dict:
        return self.a._post({"type": "clearinghouseState", "user": address})

    def spot(self, address: str) -> dict:
        return self.a._post({"type": "spotClearinghouseState", "user": address})

    def role(self, address: str) -> str:
        return self.a._post({"type": "userRole", "user": address}).get("role", "user")

    def funding_pnl(self, address: str, start_ms: int, pages: int = 6) -> float:
        total, cur = 0.0, start_ms
        for _ in range(pages):
            rows = self.a._post({"type": "userFunding", "user": address, "startTime": cur})
            if not rows:
                break
            total += sum(float(r.get("delta", {}).get("usdc", 0) or 0) for r in rows)
            nxt = max(r["time"] for r in rows) + 1
            if len(rows) < 500 or nxt <= cur:
                break
            cur = nxt
        return total


def spot_hedge_ratio(state: dict, spot: dict) -> float:
    """Share of perp exposure offset by spot balances in the same token (delta-neutral carry / basis)."""
    perp = {}
    for ap in state.get("assetPositions", []):
        p = ap["position"]
        perp[p["coin"]] = (float(p["szi"]), float(p.get("positionValue", 0) or 0))
    gross = sum(v for _, v in perp.values())
    if gross <= 0:
        return 0.0
    bal = {b["coin"]: float(b["total"]) for b in spot.get("balances", [])}
    hedged = 0.0
    for coin, (szi, val) in perp.items():
        if szi >= 0 or val <= 0:
            continue                                     # a spot-long hedge only offsets a perp SHORT
        spot_sz = max((bal.get(a, 0.0) for a in SPOT_ALIASES.get(coin, (coin,))), default=0.0)
        hedged += val * min(1.0, spot_sz / abs(szi))
    return hedged / gross
