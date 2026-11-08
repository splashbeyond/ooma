from __future__ import annotations

import typer

from .journal.notes import NoteStore
from .journal.store import Journal
from .learning.cycle import LearningCycle
from .strategies.base import all_strategies

app = typer.Typer(help="ooma: market analyst that learns what works (paper only)", no_args_is_help=True)
DB = "data/ooma.duckdb"


def _journal() -> Journal:
    import os
    os.makedirs("data", exist_ok=True)
    return Journal(DB)


@app.command()
def strategies():
    """List the strategy catalog and which are implemented."""
    for name, cls in sorted(all_strategies(include_stubs=True).items()):
        typer.echo(f"{cls.status:8} {name:24} {cls.description}")


@app.command()
def analyze(market: str, timeframe: str = typer.Argument("4h")):
    """On-demand thesis for one market. (milestones 2-6)"""
    typer.echo(f"analyze {market} {timeframe}: not implemented yet")


@app.command()
def scan():
    """Run all active strategies across markets and list candidate setups. (after milestone 2)"""
    typer.echo("scan: not implemented yet")


@app.command()
def backtest(strategy: str, market: str = "ETH", timeframe: str = "4h", days: int = 180):
    """Replay a strategy over history with grading and stats. (needs an active strategy)"""
    typer.echo("backtest: no active strategies yet")


wallets_app = typer.Typer(help="Find and track profitable directional wallets (not market makers or hedgers)")
app.add_typer(wallets_app, name="wallets")


def _tracker():
    from .adapters.hyperliquid import HyperliquidAdapter
    from .wallets.api import WalletAPI
    from .wallets.tracker import WalletTracker
    return WalletTracker(_journal().con, WalletAPI(HyperliquidAdapter()))


@wallets_app.command("discover")
def wallets_discover(n: int = 300, min_account: float = 50_000, max_turnover: float = 100.0):
    """Pick candidates from the leaderboard (profitable, not high-turnover), then fetch + classify their fills."""
    t = _tracker()
    addrs = t.discover(n, min_account, max_turnover)
    typer.echo(f"{len(addrs)} candidates after leaderboard pre-filter; profiling (~4 API calls each)...")
    profs = t.profile_all(addrs, log=typer.echo)
    from collections import Counter
    typer.echo("kinds: " + str(dict(Counter(p.kind for p in profs.values()))))
    typer.echo("status: " + str(dict(Counter(r[0] for r in t.con.execute("SELECT status FROM wallets").fetchall()))))


@wallets_app.command("restatus")
def wallets_restatus():
    """Re-apply skill/exclusion rules to stored profiles (no API calls)."""
    from collections import Counter
    typer.echo(str(dict(Counter(_tracker().restatus().values()))))


@wallets_app.command("top")
def wallets_top(status: str = "skilled", limit: int = 20):
    """Show wallets by status: skilled | weak | neutral | excluded:market_maker | excluded:hedger ..."""
    t = _tracker()
    rows = sorted(t.by_status(status), key=lambda p: -p["daily_t"])[:limit]
    typer.echo(f"{'address':14} {'kind':12} {'closes':>6} {'days':>5} {'netPnL':>12} {'bps/$':>7} {'win%':>5} {'t':>5} {'top5%':>5} {'maker%':>6} {'hold(min)':>9}")
    for p in rows:
        hold = (p["median_hold_s"] or 0) / 60
        typer.echo(f"{p['address'][:12]}.. {p['kind']:12} {p['n_closes']:6d} {p['active_days']:5d} {p['net_pnl']:12,.0f} "
                   f"{p['pnl_bps']:7.2f} {100 * (p['win_rate'] or 0):5.1f} {p['daily_t']:5.2f} {100 * p['top5_share']:5.0f} "
                   f"{100 * p['maker_share']:6.0f} {hold:9.1f}")


@wallets_app.command("snapshot")
def wallets_snapshot():
    """Record where skilled and weak wallets are positioned right now."""
    n = _tracker().snapshot()
    typer.echo(f"snapshot of {n} wallets saved")


@wallets_app.command("persistence")
def wallets_persistence():
    """Does a wallet's first-half skill predict its second-half profit? (the luck check)"""
    from .wallets.profile import prepare
    t = _tracker()
    sets = {a: t._fills_df(a) for (a,) in t.con.execute("SELECT address FROM wallets WHERE kind='directional'").fetchall()}
    from .wallets.scoring import persistence_report
    typer.echo(str(persistence_report({a: d for a, d in sets.items() if d.height})))


@app.command()
def positions():
    """Open paper positions. (milestone 6)"""
    typer.echo("positions: not implemented yet")


@app.command()
def notes(status: str = typer.Option(None), strategy: str = typer.Option(None)):
    """Read ooma's notebook."""
    for n in NoteStore(_journal()).find(status=status, strategy=strategy):
        typer.echo(f"[{n.status:9}] {n.kind:13} {n.strategy or '-':20} +{n.n_support}/-{n.n_against}  {n.text}")


@app.command()
def learn():
    """Run one learning cycle: grade, measure, reflect, test, propose."""
    j = _journal()
    rep = LearningCycle(j, NoteStore(j)).run()
    typer.echo(f"graded {rep.graded}, notes written {rep.notes_written}")
    for k, s in rep.stats.items():
        flag = "" if s.sufficient else "  (n<30: insufficient)"
        typer.echo(f"  {k}: n={s.n} avgR={s.avg_r:.2f} CI[{s.ci_low:.2f},{s.ci_high:.2f}]{flag}")
    for step, why in rep.skipped.items():
        typer.echo(f"  skipped {step}: {why}")


@app.command()
def lab(days: int = 540, ideas: int = 40, timeframes: str = "1h,4h", seed: int = 0, llm: bool = False):
    """Invent strategies, test them honestly (train/holdout), and start forward paper tests on survivors."""
    import random
    from datetime import datetime, timedelta, timezone

    import yaml

    from .adapters.hyperliquid import HyperliquidAdapter
    from .lab.cycle import Lab
    from .lab.data import CachedLoader
    from .lab.ideation import TemplateIdeator

    cfg = yaml.safe_load(open("config.yaml"))
    tfs, markets = timeframes.split(","), cfg["coins"]
    end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    loader = CachedLoader(HyperliquidAdapter())
    for m in markets:
        for tf in tfs:
            loader.prefetch(m, tf, start - timedelta(days=60), end)
    j = _journal()
    notes = NoteStore(j)
    if llm:
        from .lab.ideation import LLMIdeator
        from .llm import LLMClient
        c = cfg["llm"]
        ideator = LLMIdeator(LLMClient(c["model"], c["fallbacks"], con=j.con, daily_budget_usd=c["daily_budget_usd"]), notes, j)
    else:
        ideator = TemplateIdeator()
    from .execution.model import from_config
    L = Lab(j, notes, loader, ideator, seed=seed or random.randint(1, 10**6), model=from_config(cfg.get("execution")))
    rep = L.discover(start, end, markets, tfs, n_ideas=ideas)
    rep = L.review_live(rep)
    typer.echo(f"tried {rep.tried} (dupes {rep.duplicates}) | rejected in train {rep.rejected_train} | "
               f"holdout-tested {rep.holdout_tested} | rejected in holdout {rep.rejected_holdout}")
    typer.echo(f"incubating: {rep.incubating or 'none'} | promoted: {rep.promoted or 'none'} | retired: {rep.retired or 'none'}")


@app.command("exec-calibrate")
def exec_calibrate():
    """Compare the execution model with the live order book: real round-trip cost to trade $1k/$10k/$100k."""
    import yaml

    from .adapters.hyperliquid import HyperliquidAdapter
    from .execution.book import walk_book
    from .execution.model import from_config
    cfg = yaml.safe_load(open("config.yaml"))
    m, a = from_config(cfg.get("execution")), HyperliquidAdapter()
    typer.echo("market  mid        spread_bps | taker round-trip bps (book walk, excl. fees) at $1k / $10k / $100k | model half-spread")
    for mk in cfg["coins"]:
        b = a.book(mk)
        mid = b.mid
        row = []
        for notional in (1_000, 10_000, 100_000):
            q = notional / mid
            buy, _, ex1 = walk_book(b.asks, q, side=1)
            sell, _, ex2 = walk_book(b.bids, q, side=-1)
            row.append(f"{(buy - sell) / mid * 1e4:6.2f}{'*' if ex1 or ex2 else ' '}")
        spread = (b.asks[0][0] - b.bids[0][0]) / mid * 1e4
        typer.echo(f"{mk:6} {mid:10.2f} {spread:8.2f}   | {'  '.join(row)} | {m.half_spread_bps.get(mk, m.default_half_spread_bps):.2f} (x2 = {2 * m.half_spread_bps.get(mk, m.default_half_spread_bps):.2f} round trip)")
    typer.echo("* = exceeded the 20 visible levels. Fees on top: taker %.1f bps/side, maker %.1f bps/side." % (m.fees.bps("taker"), m.fees.bps("maker")))


def _director():
    import os

    import yaml

    from .adapters.hyperliquid import HyperliquidAdapter
    from .director.data import DataHub
    from .director.session import Director
    from .llm import LLMClient
    from .scripts.binary import Gates
    from .scripts.sandbox import SandboxRunner
    cfg = yaml.safe_load(open("config.yaml"))
    os.makedirs("data", exist_ok=True)
    j = Journal("data/director.duckdb")
    notes = NoteStore(j)
    c, d = cfg["llm"], cfg.get("director", {})
    llm = LLMClient(c["model"], c["fallbacks"], con=j.con, daily_budget_usd=c["daily_budget_usd"])
    sb = SandboxRunner(isolated_user=os.environ.get("OOMA_SANDBOX_USER") or None)
    focus = d.get("focus") or None
    hub = DataHub(HyperliquidAdapter(), markets=focus or cfg["coins"])
    return Director(j, notes, llm, hub, sb, gates=Gates(), hypotheses=d.get("hypotheses", 4), rounds=d.get("rounds", 2), focus=focus), j, d


@app.command()
def direct():
    """Run ONE research session now: the LLM proposes, writes and tests strategy scripts (sandboxed, judged by code)."""
    d, j, _ = _director()
    rep = d.run_session()
    typer.echo(str(rep))


@app.command("director-run")
def director_run():
    """Run the director forever: research sessions back to back, plus an hourly forward-test tick for live scripts."""
    import json
    import logging
    import time
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    log = logging.getLogger("ooma.director")
    d, j, cfg = _director()
    pause = cfg.get("pause_s", 120)
    Path("data/reports").mkdir(parents=True, exist_ok=True)

    def next_tick_time():
        return (datetime.now(timezone.utc) + timedelta(hours=1)).replace(minute=5, second=0, microsecond=0)
    next_tick = datetime.now(timezone.utc)
    while True:
        sleep_s = pause
        try:
            now = datetime.now(timezone.utc)
            if now >= next_tick:
                log.info("forward tick: %s", d.tick())
                next_tick = next_tick_time()
            rep = d.run_session()
            log.info("session: %s", rep)
            with open(f"data/reports/director-{now.date()}.md", "a") as f:
                f.write(f"\n## session {rep.session}\n{rep}\n")
            if "budget" in rep.stopped.lower():
                sleep_s = 900                                  # out of daily LLM budget: wait, keep forward-testing
            Path("data/director_heartbeat.json").write_text(json.dumps({"ok": True, "at": datetime.now(timezone.utc).isoformat(), "looks_7d": d.recent_looks()}))
        except Exception:                                    # noqa: BLE001  never die
            log.exception("director cycle failed")
            sleep_s = 300
        time.sleep(sleep_s)


@app.command("scripts")
def scripts_list(stage: str = typer.Option(None)):
    """Every script the director has written and where it stands."""
    j = Journal("data/director.duckdb")
    recs = sorted(j.all("scripts"), key=lambda r: r.created_at)
    for r in recs:
        if stage and r.stage != stage:
            continue
        e = {x["stage"]: x for x in r.evals}
        t, f = e.get("train", {}), e.get("forward", {})
        typer.echo(f"{r.stage:17} {r.name:32} {r.timeframe}/H{r.horizon_bars:<3} train n={t.get('n')} hit={t.get('hit_rate') and round(t['hit_rate'], 3)} "
                   f"vs {t.get('chance') and round(t['chance'], 3)} | fwd n={f.get('n')}  {r.reason[:60]}")


@app.command("lab-status")
def lab_status():
    """Every invented strategy and where it stands."""
    for r in sorted(_journal().all("strategy_records"), key=lambda r: r.stage):
        ev = r.evaluations[-1] if r.evaluations else {}
        typer.echo(f"{r.stage:17} {r.spec.name:24} {r.spec.timeframe} n={ev.get('n')} avgR={ev.get('avg_r')} {r.reason}")


def _loop(seed: int = 0):
    import yaml

    from .predict.loop import RecursiveLoop
    from .predict.store import PredictStore
    cfg = yaml.safe_load(open("config.yaml"))
    j = _journal()
    return RecursiveLoop(PredictStore(j.con), NoteStore(j), cfg.get("predict"), seed=seed), cfg["coins"]


@app.command("predict-replay")
def predict_replay(days: int = 120):
    """Warm start: walk-forward over recent history so the loop begins with learned state."""
    from .adapters.hyperliquid import HyperliquidAdapter
    from .predict.report import digest
    from .predict.runner import replay
    loop, markets = _loop()
    typer.echo(f"replayed bars: {replay(loop, HyperliquidAdapter(), markets, days)}")
    typer.echo(digest(loop))


@app.command()
def run():
    """Run the loop forever: tick every hour, learn, write daily digests. Safe to restart."""
    import logging

    from .adapters.hyperliquid import HyperliquidAdapter
    from .predict.runner import run_forever
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    from .wallets.api import WalletAPI
    from .wallets.tracker import WalletTracker
    loop, markets = _loop()
    a = HyperliquidAdapter()
    run_forever(loop, a, markets, tracker=WalletTracker(loop.s.con, WalletAPI(a)))


@app.command("predict-status")
def predict_status():
    """Current digest: ensemble vs baseline, trusted predictors, population."""
    from .predict.report import digest
    typer.echo(digest(_loop()[0]))


@app.command()
def scorecard():
    """Per-strategy stats with n and CI, vs baselines. (milestone 7)"""
    typer.echo("scorecard: not implemented yet")


@app.command()
def review():
    """Weekly review with proposed changes awaiting your approval. (milestone 7)"""
    typer.echo("review: not implemented yet")


if __name__ == "__main__":
    app()
