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
def scorecard():
    """Per-strategy stats with n and CI, vs baselines. (milestone 7)"""
    typer.echo("scorecard: not implemented yet")


@app.command()
def review():
    """Weekly review with proposed changes awaiting your approval. (milestone 7)"""
    typer.echo("review: not implemented yet")


if __name__ == "__main__":
    app()
