# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ModelOps-Calabaria is the bridge package that connects the Calabaria science framework with the ModelOps infrastructure. It implements the protocols defined in modelops-contracts to enable distributed simulation and optimization workflows.

This package serves as the integration layer between:
- **Calabaria**: The science framework for model calibration and simulation
- **ModelOps**: The infrastructure orchestration for distributed ML experimentation
- **ModelOps-Contracts**: The stable interface (seam) defining protocols both systems use

## Development Commands

### Package Management
This project uses `uv` for dependency management:
```bash
# Install dependencies
uv sync

# Run commands within the virtual environment
uv run <command>
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_<module>.py

# Run with verbose output
uv run pytest -v
```

### Linting and Type Checking
```bash
# Run Ruff linter
uv run ruff check .

# Run Ruff with auto-fixes
uv run ruff check . --fix

# Run type checking (when configured)
uv run pyright
uv run mypy
```

### Build
```bash
# Build the package
uv build
```

## Architecture

### Core Components to Implement

1. **AdaptiveAlgorithm Bridge**
   - Wrap Calabaria's optimization algorithms (Optuna, GridSearch) to implement the `AdaptiveAlgorithm` protocol
   - Translate between Calabaria's parameter representation and `UniqueParameterSet`
   - Handle the ask-tell loop with proper trial tracking

2. **SimulationService Bridge**
   - Implement `SimulationService` protocol wrapping ModelOps' distributed execution
   - Convert Calabaria models to simulation functions compatible with the protocol
   - Handle data serialization between Polars DataFrames and Arrow IPC format

3. **Model Adapter**
   - Transform Calabaria `BaseModel.simulate()` to match `SimulationFunction` signature
   - Handle parameter transformations (log, logit) at the boundary
   - Manage seed derivation for replicates

4. **Data Translation Layer**
   - Convert between Calabaria's Polars DataFrames and Arrow IPC bytes
   - Handle `SimReturn` (dict of named tables) format
   - Implement aggregation functions compatible with both systems

### Key Integration Points

**From Calabaria:**
- Models with `parameters()` and `simulate()` methods
- CalibrationTask with targets and alignment strategies
- Evaluation pipelines with loss functions
- Dispatchers (Serial, Joblib) for local execution

**From ModelOps:**
- Distributed simulation service (Dask-based)
- Bundle management for code/data artifacts
- Cache and storage services
- Execution context management

**Protocol Implementation:**
- `adaptive.py`: AdaptiveAlgorithm for optimization
- `sim.py`: SimulationService for distributed execution
- `types.py`: Common data structures (TrialResult, UniqueParameterSet)

### Design Principles

1. **Zero coupling**: This package depends on both systems but neither depends on it
2. **Protocol-first**: All integration through modelops-contracts interfaces
3. **Type safety**: Use Pydantic models and type hints throughout
4. **Immutable data**: Follow frozen/immutable patterns from contracts
5. **Deterministic execution**: Proper seed management for reproducibility

### Data Flow

1. **Optimization Loop**:
   ```
   Calabaria Optimizer → AdaptiveAlgorithm adapter → ask()
   → UniqueParameterSet → SimulationService → ModelOps execution
   → SimReturn → TrialResult → tell() → Calabaria Optimizer
   ```

2. **Simulation Execution**:
   ```
   Calabaria Model → Model Adapter → SimulationFunction
   → Serialized to bundle → Distributed execution
   → Arrow IPC results → Polars DataFrame → Calabaria evaluation
   ```

## Dependencies

- **modelops-contracts**: Protocol definitions (required)
- **modelops-bundle**: Bundle management for artifacts (required)
- **calabaria**: Science framework (development dependency)
- **modelops**: Infrastructure (development dependency)

## Implementation Priority

1. Basic AdaptiveAlgorithm wrapper for Optuna
2. Simple SimulationService client for ModelOps
3. Model adapter for SIR-type models
4. Data serialization utilities
5. Integration tests with both systems