"""
TokenLedger Database CLI Commands

Commands for managing TokenLedger database schema and migrations.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

# Create db subcommand group
app = typer.Typer(
    name="db",
    help="Database management commands",
    no_args_is_help=True,
)

console = Console()


def get_database_url(database_url: str | None = None) -> str:
    """Get database URL from argument or environment."""
    url = (
        database_url or os.environ.get("DATABASE_URL") or os.environ.get("TOKENLEDGER_DATABASE_URL")
    )
    if not url:
        console.print(
            "[bold red]Error:[/bold red] No database URL provided.\n"
            "Set DATABASE_URL environment variable or use --database-url option."
        )
        raise typer.Exit(1) from None
    return url


@app.command()
def init(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
) -> None:
    """
    Initialize TokenLedger database schema.

    Creates the schema and migration tracking table if they don't exist.
    """
    url = get_database_url(database_url)

    console.print("[bold]Initializing TokenLedger database...[/bold]")
    console.print(f"  Schema: [cyan]{schema}[/cyan]")

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        runner.init()

        console.print("[bold green]Database initialized successfully.[/bold green]")
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None


@app.command()
def upgrade(
    revision: Annotated[
        str,
        typer.Argument(help="Target revision (default: head)"),
    ] = "head",
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be done without making changes"),
    ] = False,
) -> None:
    """
    Run database migrations.

    Upgrades the database schema to the specified revision.
    Use 'head' to run all pending migrations.
    """
    url = get_database_url(database_url)

    console.print("[bold]Upgrading TokenLedger database...[/bold]")
    console.print(f"  Schema: [cyan]{schema}[/cyan]")
    console.print(f"  Target: [cyan]{revision}[/cyan]")

    if dry_run:
        console.print("  Mode: [yellow]DRY RUN[/yellow]")

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        applied = runner.upgrade(revision, dry_run=dry_run)

        if applied:
            console.print(f"\n[bold green]Applied {len(applied)} migration(s):[/bold green]")
            for m in applied:
                console.print(f"  [green]\u2713[/green] {m}")
        else:
            console.print("\n[dim]Database is already up to date.[/dim]")

    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None


@app.command()
def downgrade(
    revision: Annotated[
        str,
        typer.Argument(help="Target revision to downgrade to"),
    ],
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """
    Revert database migrations.

    Downgrades the database schema to the specified revision.
    This is a destructive operation - data may be lost.
    """
    url = get_database_url(database_url)

    if not yes:
        confirm = typer.confirm(
            f"This will downgrade the database to revision '{revision}'. "
            "This may result in data loss. Continue?"
        )
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    console.print("[bold]Downgrading TokenLedger database...[/bold]")
    console.print(f"  Schema: [cyan]{schema}[/cyan]")
    console.print(f"  Target: [cyan]{revision}[/cyan]")

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        reverted = runner.downgrade(revision)

        if reverted:
            console.print(f"\n[bold yellow]Reverted {len(reverted)} migration(s):[/bold yellow]")
            for m in reverted:
                console.print(f"  [yellow]\u21b6[/yellow] {m}")
        else:
            console.print("\n[dim]No migrations to revert.[/dim]")

    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None


@app.command()
def current(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
) -> None:
    """Show the current migration version."""
    url = get_database_url(database_url)

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        version = runner.current()

        if version:
            console.print(f"Current version: [bold cyan]{version}[/bold cyan]")
        else:
            console.print("[dim]No migrations applied yet.[/dim]")

    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None


@app.command()
def status(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
) -> None:
    """Show migration status (applied and pending)."""
    url = get_database_url(database_url)

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        status_info = runner.status()

        table = Table(title="Migration Status")
        table.add_column("Version", style="cyan")
        table.add_column("Description")
        table.add_column("Status", justify="center")
        table.add_column("Applied At")

        for migration in status_info:
            status_str = (
                "[green]\u2713 Applied[/green]"
                if migration["applied"]
                else "[yellow]\u25cb Pending[/yellow]"
            )
            applied_at = migration.get("applied_at", "-")
            table.add_row(
                migration["version"],
                migration["description"],
                status_str,
                str(applied_at) if applied_at else "-",
            )

        console.print(table)

    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None


@app.command()
def history(
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url", "-d", help="PostgreSQL connection string", envvar="DATABASE_URL"
        ),
    ] = None,
    schema: Annotated[
        str,
        typer.Option("--schema", "-s", help="Schema name for TokenLedger tables"),
    ] = "token_ledger",
) -> None:
    """Show migration history (applied migrations only)."""
    url = get_database_url(database_url)

    try:
        from tokenledger.migrations import MigrationRunner

        runner = MigrationRunner(database_url=url, schema=schema)
        history_list = runner.history()

        if not history_list:
            console.print("[dim]No migrations have been applied yet.[/dim]")
            return

        table = Table(title="Migration History")
        table.add_column("Version", style="cyan")
        table.add_column("Description")
        table.add_column("Applied At")

        for entry in history_list:
            table.add_row(
                entry["version"],
                entry["description"],
                str(entry["applied_at"]),
            )

        console.print(table)

    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] Migration system not fully implemented yet.\n"
            "This feature is coming soon."
        )
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1) from None
