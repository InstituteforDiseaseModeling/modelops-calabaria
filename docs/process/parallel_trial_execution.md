# Parallel Trial Execution for Calibration

**Date:** 2026-01-24
**Author:** Engineering
**Status:** Deployed
**Commit:** fd73a05

## Summary

Added parallel trial execution mode to the calibration wire function, enabling multiple independent ask/tell loops to run concurrently. This eliminates synchronization barriers present in the batched execution model, improving calibration throughput by approximately 1.7x.

## Problem Statement

The original batched execution model in `calibration/wire.py` operated as follows:

1. Ask Optuna for `batch_size` parameter sets
2. Submit all simulations to Dask workers
3. **Wait for ALL simulations to complete** (synchronization barrier)
4. Tell Optuna all results
5. Repeat

This synchronization barrier meant that even if 15 out of 16 simulations completed quickly, the system waited for the slowest one before proceeding. With variable simulation times (due to parameter-dependent convergence), this created significant idle time.

```
Batched Execution Timeline:
Worker 1: |===sim===|........wait........|===sim===|
Worker 2: |===sim===|........wait........|===sim===|
Worker 3: |=====long sim=====|          |===sim===|
                              ^ barrier
```

## Solution

Implemented parallel trial execution using Python's `ThreadPoolExecutor`. Each worker thread independently:

1. Asks Optuna for 1 trial (thread-safe via PostgreSQL storage)
2. Submits replicates to Dask
3. Waits only for its own results
4. Tells Optuna the result
5. Immediately asks for the next trial

```
Parallel Execution Timeline:
Thread 1: |===sim===|===sim===|===sim===|===sim===|
Thread 2: |===sim===|===sim===|===sim===|===sim===|
Thread 3: |=====long sim=====|===sim===|===sim===|
                              ^ no barrier
```

## Implementation

### New Flag

```python
# wire.py line 46
PARALLEL_TRIALS = True  # Set to False for original batched behavior
```

### Execution Modes

| Mode | Function | Description |
|------|----------|-------------|
| Parallel (default) | `_run_parallel_trials()` | Multi-threaded independent ask/tell loops |
| Batched | `_run_batched_trials()` | Original single-threaded batch processing |

### Key Code Changes

```python
def _run_parallel_trials(
    job: CalibrationJob,
    adapter,
    sim_service: SimulationService,
    n_replicates: int,
    n_workers: int,  # Uses batch_size as thread count
    target_entrypoints: list,
    sim_entrypoint: str,
) -> int:
    """Run trials in parallel with independent ask/tell loops."""

    def worker_loop(worker_id: int):
        while not stop_flag.is_set():
            if completed_count >= max_trials:
                return

            # Ask for 1 trial
            param_sets = adapter.ask(n=1)

            # Submit and gather for each target
            for target in targets_to_run:
                future = sim_service.submit_replicate_set(replicate_set, target)
                results = sim_service.gather([future])

            # Tell result immediately
            adapter.tell([final_result])

    # Launch worker threads
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(worker_loop, i) for i in range(n_workers)]
```

## Benchmark Results

### Test Configuration

- **Model:** WHO Typhoidsim (typhoid fever simulation)
- **Parameters:** 5 calibratable (tai, teer, susc_slope, tppi, p_acute)
- **Targets:** 2 (incidence_distribution, incidence_by_age)
- **Replicates:** 5 per trial
- **Max trials:** 100
- **Dask workers:** 20
- **Parallel threads:** 16 (batch_size)

### Throughput Comparison

| Metric | Batched Mode | Parallel Mode | Improvement |
|--------|--------------|---------------|-------------|
| Trials/minute | 2.7 | 4.5 | **1.67x** |
| Time for 100 trials | ~37 min | ~22 min | 15 min saved |

### Observations

1. **Burst completions:** Multiple trials often complete within seconds of each other (e.g., trials 36-37, 38-40), demonstrating effective parallelism.

2. **Worker utilization:** Parallel mode keeps Dask workers consistently busy rather than having idle periods between batches.

3. **Limiting factors:**
   - Simulation time (~30s/trial) dominates overall execution
   - 20 Dask workers serving 16 threads creates some contention
   - 2 targets evaluated sequentially within each thread

### Sample Progress Log

```
05:04:49 - Progress: 35 completed, best loss: 26.99
05:04:55 - Progress: 36 completed, best loss: 26.99
05:04:55 - Progress: 37 completed, best loss: 26.99  # Burst completion
05:05:30 - Progress: 38 completed, best loss: 26.99
05:05:30 - Progress: 39 completed, best loss: 26.99  # Burst completion
05:05:31 - Progress: 40 completed, best loss: 26.99
```

## Thread Safety

The implementation relies on thread-safe components:

1. **Optuna RDBStorage (PostgreSQL):** Handles concurrent `ask()` and `tell()` calls via database transactions
2. **Dask Client:** Designed for concurrent access from multiple threads
3. **Python threading primitives:** `Lock` for counter, `Event` for stop flag

## Configuration

To switch between modes:

```python
# In wire.py
PARALLEL_TRIALS = True   # Parallel mode (default)
PARALLEL_TRIALS = False  # Batched mode (original)
```

## Future Improvements

1. **Parallel target evaluation:** Currently targets are evaluated sequentially within each thread. Could parallelize for additional speedup.

2. **Dynamic thread count:** Adjust thread count based on available Dask workers rather than using fixed batch_size.

3. **Adaptive batch sizing:** For very fast simulations, batching might still be more efficient due to reduced Optuna DB round-trips.

## Files Changed

| File | Change |
|------|--------|
| `src/modelops_calabaria/calibration/wire.py` | Added parallel execution mode (+197 lines) |

## Dependencies

No new dependencies. Uses Python standard library `threading` and `concurrent.futures`.
