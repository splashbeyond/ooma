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


@app.command()
def wallets():
    """Tracked wallet leaderboard and cohort positioning. (milestone 4)"""
    typer.echo("wallets: not implemented yet")


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
def lab(days: int = 540, ideas: int = 40, timeframes: str = "1h,4h", seed: int = 0):
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
    L = Lab(j, NoteStore(j), loader, TemplateIdeator(), seed=seed or random.randint(1, 10**6))
    rep = L.discover(start, end, markets, tfs, n_ideas=ideas)
    rep = L.review_live(rep)
    typer.echo(f"tried {rep.tried} (dupes {rep.duplicates}) | rejected in train {rep.rejected_train} | "
               f"holdout-tested {rep.holdout_tested} | rejected in holdout {rep.rejected_holdout}")
    typer.echo(f"incubating: {rep.incubating or 'none'} | promoted: {rep.promoted or 'none'} | retired: {rep.retired or 'none'}")


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
    loop, markets = _loop()
    run_forever(loop, HyperliquidAdapter(), markets)


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
