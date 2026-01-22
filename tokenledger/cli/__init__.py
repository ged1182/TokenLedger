"""
TokenLedger CLI

Command-line interface for TokenLedger database management and server operations.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from tokenledger import __version__
from tokenledger.cli.db import app as db_app

# Create main CLI app
app = typer.Typer(
    name="tokenledger",
    help="TokenLedger - LLM Cost Analytics for Postgres",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Rich console for styled output
console = Console()

# Register subcommand groups
app.add_typer(db_app, name="db", help="Database management commands")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind the server to")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port to bind the server to")] = 8765,
    reload: Annotated[bool, typer.Option(help="Enable auto-reload on code changes")] = False,
) -> None:
    """Start the TokenLedger dashboard API server."""
    from tokenledger.server import app as fastapi_app

    console.print("[bold green]Starting TokenLedger server...[/bold green]")
    console.print(f"  Host: [cyan]{host}[/cyan]")
    console.print(f"  Port: [cyan]{port}[/cyan]")
    console.print(f"  Dashboard: [link=http://{host}:{port}]http://{host}:{port}[/link]")
    console.print()

    import uvicorn

    uvicorn.run(
        "tokenledger.server:app" if reload else fastapi_app,
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def version() -> None:
    """Show TokenLedger version."""
    console.print(f"TokenLedger [bold cyan]v{__version__}[/bold cyan]")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
