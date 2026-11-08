import typer

app = typer.Typer(help="ooma personal trading ooma (paper only)")


@app.callback()
def main():
    """ooma ooma."""


@app.command()
def analyze(coin: str, timeframe: str = typer.Argument("4h")):
    """Analyze a coin. Not implemented until A1."""
    typer.echo(f"analyze {coin} {timeframe}: not implemented yet")


if __name__ == "__main__":
    app()
