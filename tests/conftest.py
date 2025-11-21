"""Ensure local modelops-contracts sources are importable for tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
contracts_src = ROOT.parent / "modelops-contracts" / "src"
if contracts_src.exists():
    sys.path.insert(0, str(contracts_src))
