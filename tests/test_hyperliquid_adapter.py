from datetime import datetime, timezone

import httpx
import pytest

from ooma.adapters.base import Capability
from ooma.adapters.hyperliquid import HyperliquidAdapter

H4 = 14_400_000


def bar(t, c=100.0):
    return {"t": t, "T": t + H4 - 1, "s": "ETH", "i": "4h", "o": str(c), "c": str(c),
            "h": str(c + 1), "l": str(c - 1), "v": "10", "n": 5}


def adapter(handler):
    return HyperliquidAdapter(httpx.Client(transport=httpx.MockTransport(handler)))


def test_capabilities_declared():
    a = adapter(lambda r: httpx.Response(200, json={}))
    assert Capability.CANDLES in a.capabilities
    assert Capability.PARTICIPANTS not in a.capabilities


def test_candles_drop_unclosed_bar_no_lookahead():
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % H4)
    bars = [bar(t0), bar(t0 + H4), bar(t0 + 2 * H4)]
    a = adapter(lambda r: httpx.Response(200, json=bars))
    # as_of is mid-way through the 3rd bar: it must be excluded
    end = datetime.fromtimestamp((t0 + 2 * H4 + 1000) / 1000, tz=timezone.utc)
    df = a.candles("ETH", "4h", datetime.fromtimestamp(t0 / 1000, tz=timezone.utc), end)
    assert df.height == 2
    assert df["close"].dtype.is_float()


def test_candles_paginate_and_dedupe():
    t0 = 1_700_000_000_000 - (1_700_000_000_000 % H4)
    pages = [[bar(t0 + i * H4) for i in range(5000)],
             [bar(t0 + i * H4) for i in range(5000, 5010)]]
    calls = iter(pages)
    a = adapter(lambda r: httpx.Response(200, json=next(calls)))
    end = datetime.fromtimestamp((t0 + 6000 * H4) / 1000, tz=timezone.utc)
    df = a.candles("ETH", "4h", datetime.fromtimestamp(t0 / 1000, tz=timezone.utc), end)
    assert df.height == 5010


def test_book_parse():
    body = {"coin": "BTC", "time": 1700000000000, "levels": [
        [{"px": "99", "sz": "2", "n": 1}], [{"px": "101", "sz": "3", "n": 1}]]}
    b = adapter(lambda r: httpx.Response(200, json=body)).book("BTC")
    assert b.mid == 100 and b.bids[0] == (99.0, 2.0)


@pytest.mark.live
def test_live_four_markets():
    a = HyperliquidAdapter()
    now = datetime.now(timezone.utc)
    for m in ["BTC", "ETH", "SOL", "HYPE"]:
        assert a.candles(m, "4h", datetime(2026, 10, 1, tzinfo=timezone.utc), now).height > 50
        assert a.context(m).open_interest > 0
        assert a.book(m).mid > 0
        assert a.funding_history(m, datetime(2026, 11, 1, tzinfo=timezone.utc), now).height > 0
