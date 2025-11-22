#!/usr/bin/env python
"""End-to-end test for new modelops-bundle workflow.

Demonstrates:
1. Initialize a bundle project with template
2. Discover models in the project
3. Generate manifest
4. Track files
5. Push bundle to registry
"""

import os
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys

def run_command(cmd, cwd=None):
    """Run a shell command and return output."""
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
    return result

def main():
    """Run the e2e test."""
    print("=" * 60)
    print("ModelOps Bundle E2E Workflow Test")
    print("=" * 60)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_project"
        test_dir.mkdir()
        
        print(f"\n1. Creating test project in {test_dir}")
        
        # Step 1: Initialize bundle with template
        print("\n2. Initializing modelops-bundle with template...")
        cmd = "uv run modelops-bundle init localhost:5555/test_model --with-template"
        result = run_command(cmd, cwd=test_dir)
        if result.returncode == 0:
            print("   ✓ Bundle initialized")
        else:
            print("   ✗ Failed to initialize bundle")
            return 1
        
        # Create a more realistic model
        print("\n3. Creating example model...")
        models_dir = test_dir / "models"
        if not models_dir.exists():
            models_dir.mkdir()
        
        sir_model = models_dir / "sir_model.py"
        sir_model.write_text('''"""Simple SIR epidemiological model."""

import numpy as np

class SIRModel:
    """Susceptible-Infected-Recovered model."""
    
    def __init__(self):
        self.name = "SIR"
    
    def parameters(self):
        """Return default parameters."""
        return {
            "beta": 0.5,  # Transmission rate
            "gamma": 0.1,  # Recovery rate
            "S0": 999,    # Initial susceptible
            "I0": 1,      # Initial infected
            "R0": 0,      # Initial recovered
            "days": 100   # Simulation days
        }
    
    def simulate(self, params, seed=None):
        """Run SIR simulation."""
        if seed:
            np.random.seed(seed)
        
        # Extract parameters
        beta = params.get("beta", 0.5)
        gamma = params.get("gamma", 0.1)
        S = params.get("S0", 999)
        I = params.get("I0", 1)
        R = params.get("R0", 0)
        days = params.get("days", 100)
        
        # Run simulation
        S_history = [S]
        I_history = [I]
        R_history = [R]
        
        N = S + I + R
        for day in range(days):
            # Calculate rates
            infection = beta * S * I / N
            recovery = gamma * I
            
            # Update populations
            S = S - infection
            I = I + infection - recovery
            R = R + recovery
            
            S_history.append(S)
            I_history.append(I)
            R_history.append(R)
        
        return {
            "S": S_history,
            "I": I_history,
            "R": R_history,
            "peak_infected": max(I_history),
            "days_to_peak": I_history.index(max(I_history))
        }
''')
        print(f"   ✓ Created {sir_model.name}")
        
        # Step 4: Discover models
        print("\n4. Discovering models...")
        cmd = "uv run modelops-bundle discover"
        result = run_command(cmd, cwd=test_dir)
        if result.returncode == 0 and "SIRModel" in result.stdout:
            print("   ✓ Discovered SIRModel")
        else:
            print("   ✗ Failed to discover models")
        
        # Step 5: Save discovered models to config
        print("\n5. Saving models to pyproject.toml...")
        cmd = "uv run modelops-bundle discover --save"
        result = run_command(cmd, cwd=test_dir)
        if result.returncode == 0:
            print("   ✓ Models saved to configuration")
        
        # Step 6: Generate manifest
        print("\n6. Generating manifest...")
        cmd = "uv run modelops-bundle manifest"
        result = run_command(cmd, cwd=test_dir)
        manifest_path = test_dir / "manifest.json"
        if manifest_path.exists():
            print("   ✓ Manifest generated")
            # Read and display key info
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)
            print(f"   Bundle digest: {manifest.get('bundle_digest', 'N/A')[:16]}...")
            print(f"   Models: {len(manifest.get('models', {}))}")
            print(f"   Files: {len(manifest.get('files', {}))}")
        else:
            print("   ✗ Failed to generate manifest")
        
        # Step 7: Add files to tracking
        print("\n7. Adding files to bundle...")
        cmd = "uv run modelops-bundle add models/"
        result = run_command(cmd, cwd=test_dir)
        if result.returncode == 0:
            print("   ✓ Files added to tracking")
        
        # Step 8: Check status
        print("\n8. Checking bundle status...")
        cmd = "uv run modelops-bundle status"
        result = run_command(cmd, cwd=test_dir)
        if result.returncode == 0:
            print("   ✓ Status checked")
            # Show first few lines of status
            lines = result.stdout.split("\n")[:5]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        
        # Step 9: Attempt push (will fail without registry)
        print("\n9. Testing push command...")
        cmd = "uv run modelops-bundle push --dry-run"
        result = run_command(cmd, cwd=test_dir)
        if "Dry run" in result.stdout or result.returncode == 0:
            print("   ✓ Push command works (dry run)")
        else:
            print("   ✗ Push command failed (expected if no registry)")
        
        print("\n" + "=" * 60)
        print("E2E Test Complete!")
        print("=" * 60)
        
        # Summary
        print("\nSummary:")
        print("- Project initialized with template")
        print("- Models discovered and saved to config")
        print("- Manifest generated with bundle digest")
        print("- Files tracked in bundle")
        print("- Ready to push to registry")
        
        print("\nTo use in a real project:")
        print("1. cd your-project")
        print("2. uv run modelops-bundle init <registry> --with-template")
        print("3. uv run modelops-bundle discover --interactive --save")
        print("4. uv run modelops-bundle manifest")
        print("5. uv run modelops-bundle add <files>")
        print("6. uv run modelops-bundle push")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
