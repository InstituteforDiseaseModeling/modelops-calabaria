"""Calabaria CLI entry point.

Provides commands for model discovery, export, verification, and manifest building.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .discover import discover_models, suggest_model_config
from .config import write_model_config, read_pyproject, validate_config
from .verify import verify_all_models, print_verification_summary
from .manifest import build_manifest, check_manifest_drift, print_manifest_summary

# Create the main app
app = typer.Typer(
    name="cb",
    help="Calabaria CLI for model export and bundle management",
    no_args_is_help=True
)

# Create subcommands
models_app = typer.Typer(help="Model discovery and export commands")
manifest_app = typer.Typer(help="Manifest generation commands", invoke_without_command=True)

app.add_typer(models_app, name="models")
app.add_typer(manifest_app, name="manifest")


@models_app.command("discover")
def models_discover(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Discover BaseModel subclasses in the project (AST scan, no imports)."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        models = discover_models()

        if not models:
            typer.echo("No BaseModel subclasses found in the project.")
            return

        typer.echo(f"Found {len(models)} models:")

        for model in models:
            typer.echo(f"  {model['full_path']}")
            if verbose:
                typer.echo(f"    File: {model['file_path']} (line {model['line_number']})")
                if model['methods']['model_outputs']:
                    typer.echo(f"    Outputs: {', '.join(model['methods']['model_outputs'])}")
                if model['methods']['model_scenarios']:
                    typer.echo(f"    Scenarios: {', '.join(model['methods']['model_scenarios'])}")

        typer.echo("\nTo export a model, run:")
        typer.echo("  cb models export <class> --files <pattern>...")

        if not verbose:
            suggestions = suggest_model_config(models)
            typer.echo("\nSuggested configurations:")
            for suggestion in suggestions:
                files_str = " ".join(f'"{f}"' for f in suggestion['files'])
                typer.echo(f"  cb models export {suggestion['class']} --files {files_str}")

    except Exception as e:
        typer.echo(f"Error discovering models: {e}", err=True)
        raise typer.Exit(1)


@models_app.command("export")
def models_export(
    class_path: str = typer.Argument(..., help="Model class path (module:Class)"),
    files: List[str] = typer.Option(..., "--files", help="File patterns (glob)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without writing")
):
    """Add or update model configuration in pyproject.toml."""

    # Validate class path format
    if ":" not in class_path:
        typer.echo("Error: class path must be in format 'module:Class'", err=True)
        raise typer.Exit(1)

    if dry_run:
        typer.echo(f"Would add to pyproject.toml:")
        typer.echo(f"  [[tool.calabaria.model]]")
        typer.echo(f"  class = \"{class_path}\"")
        typer.echo(f"  files = {files}")
        return

    try:
        write_model_config(class_path, files)
        typer.echo(f"Added model '{class_path}' to pyproject.toml")

        typer.echo(f"\nNext steps:")
        typer.echo(f"  1. cb models verify    # Check import boundaries")
        typer.echo(f"  2. cb manifest build   # Generate manifest.json")

    except Exception as e:
        typer.echo(f"Error writing configuration: {e}", err=True)
        raise typer.Exit(1)


@models_app.command("verify")
def models_verify(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    model_class: Optional[str] = typer.Option(None, "--model", help="Verify specific model class only")
):
    """Verify model import boundaries and parameter spaces."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        config = read_pyproject()
        if not config:
            typer.echo("Error: No [tool.calabaria] configuration found in pyproject.toml", err=True)
            raise typer.Exit(1)

        # Validate configuration
        errors = validate_config(config)
        if errors:
            typer.echo("Configuration errors:", err=True)
            for error in errors:
                typer.echo(f"  - {error}", err=True)
            raise typer.Exit(1)

        # Filter to specific model if requested
        if model_class:
            models = [m for m in config.get("model", []) if m["class"] == model_class]
            if not models:
                typer.echo(f"Error: Model '{model_class}' not found", err=True)
                raise typer.Exit(1)
            config["model"] = models

        # Run verification
        results = verify_all_models(config)

        # Print results
        if verbose:
            print_verification_summary(results)

        # Check if all passed
        all_passed = all(result["ok"] for result in results.values())

        if not all_passed:
            typer.echo("\nSome models failed verification. Fix import boundaries and try again.", err=True)

            # Suggest fixes for failed models
            for model_class, result in results.items():
                if not result["ok"] and result.get("unexpected"):
                    from .verify import suggest_file_patterns
                    suggestions = suggest_file_patterns(result["unexpected"])
                    typer.echo(f"\nTo fix {model_class}, add these file patterns:")
                    for suggestion in suggestions:
                        typer.echo(f"  \"{suggestion}\"")

            raise typer.Exit(1)

        typer.echo("✓ All models passed verification")

    except Exception as e:
        typer.echo(f"Error during verification: {e}", err=True)
        raise typer.Exit(1)


@manifest_app.callback(invoke_without_command=True)
def manifest_main(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check", help="Check if manifest is up to date"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Build manifest.json deterministically."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # If no subcommand, default to build
    if ctx.invoked_subcommand is None:
        if check:
            manifest_check()
        else:
            manifest_build()


@manifest_app.command("build")
def manifest_build(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output")
):
    """Build manifest.json from pyproject.toml configuration."""
    if verbose:
        logging.basicConfig(level=logging.INFO)

    try:
        manifest, bundle_id = build_manifest()

        typer.echo(f"Generated manifest.json")
        typer.echo(f"Bundle ID: {bundle_id}")

        if verbose:
            print_manifest_summary(manifest)

    except Exception as e:
        typer.echo(f"Error building manifest: {e}", err=True)
        raise typer.Exit(1)


@manifest_app.command("check")
def manifest_check():
    """Check if manifest.json is up to date (exit 1 if outdated)."""
    try:
        up_to_date = check_manifest_drift()

        if up_to_date:
            typer.echo("✓ manifest.json is up to date")
        else:
            typer.echo("✗ manifest.json is out of date", err=True)
            typer.echo("Run: cb manifest build", err=True)
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"Error checking manifest: {e}", err=True)
        raise typer.Exit(1)


@app.command("version")
def version():
    """Show version information."""
    from . import __version__
    typer.echo(f"Calabaria CLI version {__version__}")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """Calabaria CLI for model export and bundle management."""
    if verbose:
        logging.basicConfig(level=logging.INFO)


def cli_main():
    """Entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nAborted", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli_main()