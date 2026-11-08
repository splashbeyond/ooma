from __future__ import annotations

from datetime import datetime, timezone

import time

import httpx
import polars as pl

from .base import BookSnapshot, Capability, MarketContext

URL = "https://api.hyperliquid.xyz/info"
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "1d": 86_400_000}
MAX_CANDLES = 5000  # API returns at most the 5000 most recent in a range


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


class HyperliquidAdapter:
    name = "hyperliquid"
    capabilities = frozenset({Capability.CANDLES, Capability.CONTEXT, Capability.BOOK})
    # TRADES (websocket recorder) and PARTICIPANTS arrive in milestone 4.

    def __init__(self, client: httpx.Client | None = None):
        self._c = client or httpx.Client(timeout=15)

    MIN_GAP = 0.25          # seconds between requests: stay well under Hyperliquid's weight limits
    _last = 0.0

    def _post(self, body: dict):
        for attempt in range(6):
            wait = self.MIN_GAP - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            HyperliquidAdapter._last = time.monotonic()
            r = self._c.post(URL, json=body)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(30, 2 ** attempt))               # back off, then retry
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def markets(self) -> list[str]:
        return [u["name"] for u in self._post({"type": "meta"})["universe"]]

    def candles(self, market: str, interval: str, start: datetime, end: datetime) -> pl.DataFrame:
        step = INTERVAL_MS[interval]
        end_ms, cur, rows = _ms(end), _ms(start), []
        while cur < end_ms:
            chunk = self._post({"type": "candleSnapshot", "req": {
                "coin": market, "interval": interval, "startTime": cur, "endTime": end_ms}})
            if not chunk:
                break
            rows += chunk
            nxt = chunk[-1]["t"] + step
            if len(chunk) < MAX_CANDLES or nxt <= cur:
                break
            cur = nxt
        schema = {"t": pl.Datetime("ms", "UTC"), "open": pl.Float64, "high": pl.Float64,
                  "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64, "trades": pl.Int64}
        # keep only fully closed bars: bar close time T must be <= end
        data = {"t": [], "open": [], "high": [], "low": [], "close": [], "volume": [], "trades": []}
        for c in rows:
            if c["T"] + 1 > end_ms:
                continue
            data["t"].append(_dt(c["t"]))
            for k, src in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v")):
                data[k].append(float(c[src]))
            data["trades"].append(int(c["n"]))
        return pl.DataFrame(data, schema=schema).unique("t").sort("t")

    def all_contexts(self) -> list[MarketContext]:
        """Funding, open interest, premium, prices, volume for EVERY listed perp in one request."""
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        now = datetime.now(timezone.utc)
        out = []
        for u, c in zip(meta["universe"], ctxs):
            if u.get("isDelisted"):
                continue
            try:
                out.append(MarketContext(market=u["name"], as_of=now, funding_rate=float(c["funding"]),
                                         open_interest=float(c["openInterest"]), mark_price=float(c["markPx"]),
                                         oracle_price=float(c["oraclePx"]), premium=float(c["premium"] or 0),
                                         day_notional_volume=float(c["dayNtlVlm"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def top_markets(self, n: int = 15, min_day_volume: float = 30e6) -> list[str]:
        """The most liquid perps by 24h notional volume. Liquidity matters: thin coins cost far more than our cost model assumes."""
        ctxs = [c for c in self.all_contexts() if c.day_notional_volume >= min_day_volume]
        return [c.market for c in sorted(ctxs, key=lambda c: -c.day_notional_volume)[:n]]

    def context(self, market: str) -> MarketContext:
        meta, ctxs = self._post({"type": "metaAndAssetCtxs"})
        for u, c in zip(meta["universe"], ctxs):
            if u["name"] == market:
                return MarketContext(
                    market=market, as_of=datetime.now(timezone.utc),
                    funding_rate=float(c["funding"]), open_interest=float(c["openInterest"]),
                    mark_price=float(c["markPx"]), oracle_price=float(c["oraclePx"]),
                    premium=float(c["premium"] or 0), day_notional_volume=float(c["dayNtlVlm"]))
        raise KeyError(f"unknown market {market}")

    def funding_history(self, market: str, start: datetime, end: datetime) -> pl.DataFrame:
        rows, cur = [], _ms(start)
        while cur < _ms(end):
            chunk = self._post({"type": "fundingHistory", "coin": market,
                                "startTime": cur, "endTime": _ms(end)})
            if not chunk:
                break
            rows += chunk
            if chunk[-1]["time"] + 1 <= cur or len(chunk) < 400:
                break
            cur = chunk[-1]["time"] + 1
        rows = list({r["time"]: r for r in rows}.values())
        rows.sort(key=lambda r: r["time"])
        return pl.DataFrame({
            "t": [_dt(r["time"]) for r in rows],
            "funding_rate": [float(r["fundingRate"]) for r in rows],
            "premium": [float(r["premium"]) for r in rows],
        }, schema={"t": pl.Datetime("ms", "UTC"), "funding_rate": pl.Float64, "premium": pl.Float64})

    def book(self, market: str) -> BookSnapshot:
        d = self._post({"type": "l2Book", "coin": market})
        bids, asks = d["levels"]
        f = lambda side: [(float(x["px"]), float(x["sz"])) for x in side]
        return BookSnapshot(market=market, time=_dt(d["time"]), bids=f(bids), asks=f(asks))
