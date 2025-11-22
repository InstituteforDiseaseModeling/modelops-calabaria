"""Calabaria CLI entry point.

Provides commands for model discovery, export, verification, and manifest building.
"""

import logging
import sys

import typer

from .sampling import sobol_command, grid_command
from .calibration import optuna_command
from .diagnostics import report_command

# Create the main app
app = typer.Typer(
    name="cb",
    help="Calabaria CLI for model export and bundle management",
    invoke_without_command=True,
)

# Create subcommands
sampling_app = typer.Typer(help="Generate simulation jobs from parameter sampling")
calibration_app = typer.Typer(help="Generate calibration specs for parameter optimization")
diagnostics_app = typer.Typer(help="Diagnostic reports and analysis tools")

app.add_typer(sampling_app, name="sampling")
app.add_typer(calibration_app, name="calibration")
app.add_typer(diagnostics_app, name="diagnostics")
# Register sampling commands directly from implementation modules
sampling_app.command("sobol")(sobol_command)
sampling_app.command("grid")(grid_command)

# Register calibration commands directly from implementation modules
calibration_app.command("optuna")(optuna_command)

# Add the diagnostics report command
diagnostics_app.command("report")(report_command)


@app.command("version")
def version():
    """Show version information."""
    from . import __version__
    typer.echo(f"Calabaria CLI version {__version__}")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Calabaria CLI for model export and bundle management."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        typer.echo("\nError: Missing command.", err=True)
        raise typer.Exit(1)


def cli_main():
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nAborted", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
