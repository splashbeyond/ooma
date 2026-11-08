import random
from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from ooma.modules.participants import run as participants_run
from ooma.wallets.api import spot_hedge_ratio
from ooma.wallets.profile import prepare, profile_wallet
from ooma.wallets.scoring import persistence_report, status

DAY = 86_400_000
T0 = 1_700_000_000_000


def fill(t, coin, px, sz, side, start, closed=0.0, fee=0.0, maker=False, d=""):
    return dict(time=t, coin=coin, px=str(px), sz=str(sz), side=side, dir=d, closedPnl=str(closed),
                fee=str(fee), crossed=not maker, startPosition=str(start))


def directional(days=60, trades_per_day=3, win=0.58, win_size=1.4, loss_size=1.0, seed=0, risk=100.0):
    """Open then close hours later. Wins/losses in USD; realistic taker fees."""
    rng, out, t = random.Random(seed), [], T0
    for d in range(days):
        for k in range(trades_per_day):
            t0 = T0 + d * DAY + k * 3 * 3_600_000
            coin = rng.choice(["BTC", "ETH", "SOL"])
            long_ = rng.random() < 0.5
            px, sz = 2000.0, 1.0
            pnl = risk * (win_size if rng.random() < win else -loss_size)
            o_side, c_side = ("B", "A") if long_ else ("A", "B")
            sgn = 1 if long_ else -1
            out.append(fill(t0, coin, px, sz, o_side, 0.0, fee=0.9, d="Open Long" if long_ else "Open Short"))
            out.append(fill(t0 + 2 * 3_600_000, coin, px, sz, c_side, sgn * sz, closed=pnl, fee=0.9,
                            d="Close Long" if long_ else "Close Short"))
    return out


def market_maker(days=30, per_day=1500, seed=1):
    rng, out = random.Random(seed), []
    for d in range(days):
        pos, t = 0.0, T0 + d * DAY
        for i in range(per_day):
            side = "B" if rng.random() < 0.5 else "A"
            sz = 0.1
            out.append(fill(t + i * 50_000, "ETH", 2000.0, sz, side, pos, closed=rng.uniform(-0.05, 0.08), fee=-0.01, maker=rng.random() < 0.85,
                            d="Close Long" if (pos > 0) == (side == "A") else "Open Long"))
            pos += sz if side == "B" else -sz
    return out


def hedger_pair(days=40, seed=2):
    """Long BTC and short ETH of equal notional, held for weeks, adjusted a few times."""
    out = []
    out.append(fill(T0, "BTC", 100000.0, 1.0, "B", 0.0, d="Open Long"))
    out.append(fill(T0 + 1000, "ETH", 2000.0, 50.0, "A", 0.0, d="Open Short"))
    for d in range(1, days):
        out.append(fill(T0 + d * DAY, "BTC", 100000.0, 0.01, "B", 1.0 + (d - 1) * 0.01, closed=1.0, fee=0.5, d="Open Long"))
        out.append(fill(T0 + d * DAY + 1000, "ETH", 2000.0, 0.5, "A", -50.0 - (d - 1) * 0.5, closed=0.5, fee=0.5, d="Open Short"))
    for i in range(30):                       # a few more fills so it clears min_fills
        out.append(fill(T0 + 41 * DAY + i, "BTC", 100000.0, 0.001, "B", 1.4, d="Open Long"))
    return out


def prof(fills, **kw):
    return profile_wallet("0xabc", prepare(fills), **kw)


def test_skilled_directional_wallet_is_recognised():
    p = prof(directional(days=80, win=0.62, win_size=1.5))
    assert p.kind == "directional" and p.maker_share < 0.1
    assert p.net_pnl > 0 and p.daily_t > 3 and p.median_hold_s == pytest.approx(7200)
    assert status(p, n_screened=200) == "skilled"


def test_market_maker_is_excluded_even_if_profitable():
    p = prof(market_maker())
    assert p.kind == "market_maker" and p.maker_share > 0.6
    assert status(p, 10) == "excluded:market_maker"


def test_hedged_pair_book_is_excluded():
    p = prof(hedger_pair())
    assert p.net_to_gross is not None and p.net_to_gross < 0.3
    assert p.kind == "hedger" and status(p, 10).startswith("excluded")


def test_spot_hedged_perp_short_is_a_hedger():
    f = directional(days=40, win=0.7)
    p = prof(f, spot_hedge=0.9)
    assert p.kind == "hedger"
    state = {"assetPositions": [{"position": {"coin": "HYPE", "szi": "-1000", "positionValue": "30000"}}]}
    spot = {"balances": [{"coin": "HYPE", "total": "950"}]}
    assert spot_hedge_ratio(state, spot) == pytest.approx(0.95)
    assert spot_hedge_ratio({"assetPositions": [{"position": {"coin": "HYPE", "szi": "1000", "positionValue": "30000"}}]}, spot) == 0


def test_funding_carry_and_vaults_are_excluded():
    f = directional(days=40, win=0.5, win_size=1.0)
    p = prof(f, funding_pnl=10_000.0)
    assert p.kind == "carry"
    assert prof(f, role="vault").kind == "vault" and prof(f, role="agent").kind == "agent"


def test_one_lucky_trade_is_not_skill():
    f = directional(days=60, win=0.5, win_size=1.0, loss_size=1.0, seed=3)
    f.append(fill(T0 + 70 * DAY, "BTC", 100000.0, 1.0, "B", 0.0, d="Open Long"))
    f.append(fill(T0 + 70 * DAY + 3_600_000, "BTC", 100000.0, 1.0, "A", 1.0, closed=50_000.0, fee=1, d="Close Long"))
    p = prof(f)
    assert p.top1_share > 0.35 and any("concentrated" in x for x in p.flags)
    assert status(p, 200) == "neutral"                    # profitable on paper, but one trade


def test_losing_trader_is_weak_and_noise_is_neutral():
    weak = prof(directional(days=80, win=0.33, win_size=1.0, seed=5))
    assert status(weak, 200) == "weak"
    noise = prof(directional(days=80, win=0.5, win_size=1.0, loss_size=1.0, seed=6))
    assert status(noise, 200) == "neutral"


def test_multiple_testing_blocks_lucky_wallets():
    p = prof(directional(days=40, win=0.54, win_size=1.0, seed=8))
    assert status(p, 1) in ("skilled", "neutral", "weak")
    # screening thousands of wallets makes a modest t-stat meaningless
    assert status(p, 50_000) != "skilled"


def test_persistence_separates_skill_from_luck():
    luck, skill = {}, {}
    for i in range(40):                                   # pure luck: 50% wins, so first-half t says nothing about second half
        luck[f"l{i}"] = prepare(directional(days=60, win=0.5, win_size=1.0, seed=100 + i))
    for i in range(40):                                   # persistent skill: win-rate differs by wallet and persists
        skill[f"s{i}"] = prepare(directional(days=60, win=0.40 + 0.25 * (i / 40), win_size=1.0, seed=200 + i))
    r_luck, r_skill = persistence_report(luck), persistence_report(skill)
    assert abs(r_luck["spearman"]) < 0.4
    assert r_skill["spearman"] > 0.4 and r_skill["top_quintile_bps"] > r_skill["rest_bps"]


def test_participants_module_respects_as_of_and_staleness():
    t = lambda h: datetime(2026, 1, 1, h, tzinfo=timezone.utc)
    co = pl.DataFrame({"t": [t(1), t(2), t(5)], "skill_net": [0.6, 0.7, -0.9], "weak_net": [-0.2, -0.3, 0.8]})
    f = participants_run("ETH", "1h", t(2), cohort=co)
    assert f.bias == "bullish" and f.data_quality == "ok"            # t(5) snapshot is the future: invisible
    assert participants_run("ETH", "1h", t(9), cohort=co).data_quality == "stale"
    assert participants_run("ETH", "1h", t(1), cohort=None).data_quality == "partial"
    assert participants_run("ETH", "1h", t(6), cohort=co).bias == "bearish"


# ---------- tracker: the database + orchestration layer (a fake API, no network) ----------
import duckdb

from ooma.wallets.tracker import WalletTracker, cohort_frame


class FakeAPI:
    def __init__(self, book):
        self.book = book                                   # address -> dict(fills, role, positions)

    def leaderboard(self):
        rows = [dict(address=a, account_value=200_000.0, allTime_pnl=50_000.0, allTime_roi=0.5 + i / 10,
                     month_pnl=5_000.0, month_vlm=1_000_000.0) for i, a in enumerate(self.book)]
        rows.append(dict(address="0xMM", account_value=1e6, allTime_pnl=9e6, allTime_roi=9.0, month_pnl=1e6, month_vlm=5e9))
        return pl.DataFrame(rows)

    def fills(self, address, start_ms=0, end_ms=None, max_fills=10_000):
        return [f for f in self.book[address]["fills"] if f["time"] >= start_ms]

    def role(self, address): return self.book[address].get("role", "user")
    def state(self, address):
        pos = self.book[address].get("positions", [])
        return {"assetPositions": [{"position": {"coin": c, "szi": str(s), "positionValue": str(abs(s) * 100),
                                                 "entryPx": "100", "leverage": {"value": 3}, "liquidationPx": None}} for c, s in pos]}
    def spot(self, address): return {"balances": []}
    def funding_pnl(self, address, start_ms, pages=6): return 0.0


def test_tracker_end_to_end_with_fake_api():
    book = {"0xGOOD": dict(fills=directional(days=90, win=0.62, win_size=1.5, seed=11), positions=[("ETH", 10.0)]),
            "0xBAD": dict(fills=directional(days=90, win=0.30, win_size=1.0, seed=12), positions=[("ETH", -10.0)]),
            "0xVAULT": dict(fills=directional(days=30, seed=13), role="vault")}
    con = duckdb.connect()
    t = WalletTracker(con, FakeAPI(book))
    found = t.discover(n=10, max_turnover=100)
    assert "0xMM" not in found and set(found) == set(book)            # the turnover pre-filter drops the market maker
    profs = t.profile_all(found)                                       # must survive repeated DB calls (regression)
    assert len(profs) == 3
    st = dict(con.execute("SELECT address, status FROM wallets").fetchall())
    assert st["0xGOOD"] == "skilled" and st["0xBAD"] == "weak" and st["0xVAULT"] == "excluded:vault"
    t.profile_all(found)                                               # incremental refresh adds no duplicate fills
    assert con.execute("SELECT COUNT(*) FROM wallet_fills WHERE address='0xGOOD'").fetchall()[0][0] == len(book["0xGOOD"]["fills"])
    assert t.snapshot() == 2                                           # skilled + weak tracked, vault ignored
    co = cohort_frame(con, "ETH")
    r = co.row(0, named=True)
    assert r["skill_net"] == 1.0 and r["weak_net"] == -1.0             # skilled long, weak short


def test_short_hold_bots_are_not_followable():
    f = []
    for d in range(60):
        for k in range(4):                      # open/close within seconds, always wins: a latency bot, not a trader to follow
            t0 = T0 + d * DAY + k * 3_600_000
            f.append(fill(t0, "ETH", 2000.0, 1.0, "B", 0.0, fee=0.5, d="Open Long"))
            f.append(fill(t0 + 5_000, "ETH", 2000.0, 1.0, "A", 1.0, closed=40.0, fee=0.5, d="Close Long"))
    p = prof(f)
    assert p.kind == "directional" and p.median_hold_s == pytest.approx(5.0)
    assert status(p, 10) == "excluded:short_term"
