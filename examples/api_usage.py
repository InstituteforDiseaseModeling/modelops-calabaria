#!/usr/bin/env python3
"""Example usage of the modelops-calabaria programmatic API.

This demonstrates how other packages can use modelops-calabaria
functionality programmatically without going through the CLI.
"""

from pathlib import Path
from modelops_calabaria import CalabariaCLI, quick_discover, quick_bundle_id


def main():
    """Demonstrate API usage."""
    print("🔍 ModelOps-Calabaria API Usage Examples\n")

    # Example 1: Quick functions for simple operations
    print("1. Using convenience functions:")
    models = quick_discover()
    print(f"   Found {len(models)} models:")
    for model in models:
        print(f"     - {model['full_path']}")

    bundle_id = quick_bundle_id()
    print(f"   Bundle ID: {bundle_id}")
    print()

    # Example 2: Using the CalabariaCLI class for more control
    print("2. Using CalabariaCLI class:")
    cli = CalabariaCLI()

    # Model discovery
    models = cli.discover()
    print(f"   Discovered {len(models)} models")

    # Get suggested configurations
    suggestions = cli.suggest_configs(models)
    print(f"   Generated {len(suggestions)} configuration suggestions")

    # Verify models
    try:
        results = cli.verify()
        passed = sum(1 for r in results.values() if r["ok"])
        total = len(results)
        print(f"   Verification: {passed}/{total} models passed")
    except ValueError as e:
        print(f"   Verification: {e}")

    # Get manifest
    try:
        manifest = cli.get_manifest()
        print(f"   Manifest: {len(manifest['models'])} models, schema {manifest['schema']}")
    except Exception as e:
        print(f"   Manifest: Error - {e}")

    print()

    # Example 3: Using context manager for different directory
    print("3. Working with different directories:")
    epi_models_path = Path("epi_models")
    if epi_models_path.exists():
        with CalabariaCLI(epi_models_path) as epi_cli:
            epi_models = epi_cli.discover()
            print(f"   Found {len(epi_models)} models in epi_models/")
            epi_bundle = epi_cli.get_bundle_id()
            print(f"   Epi models bundle ID: {epi_bundle}")
    else:
        print("   No epi_models directory found")

    print("\n✅ API examples completed!")


if __name__ == "__main__":
    main()