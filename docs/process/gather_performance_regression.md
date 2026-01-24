# Gather Performance Regression Analysis

**Date:** 2026-01-24
**Status:** Needs Fix
**Severity:** Critical - 50% throughput loss

## Summary

A fix for exception handling in `DaskSimulationService.gather()` introduced a severe performance regression by changing parallel waiting to sequential waiting. This cut throughput in half.

## The Original Problem

The optimized calibration path (`submit_replicates()` + `submit_aggregation()`) in `wire.py` expects `gather()` to return exceptions as values:

```python
# wire.py lines 256-266
all_agg_results = sim_service.gather([f for _, f in agg_futures])
for (target, _), result in zip(agg_futures, all_agg_results):
    if result and not isinstance(result, Exception):  # <-- expects Exception as value
        trial_result = convert_to_trial_result(params, result)
    else:
        error_msg = str(result) if result else "No result"
        # handle error...
```

But Dask's `Client.gather()` **raises** exceptions by default, causing threads to crash when any aggregation fails (e.g., NaN loss).

## The Bad Fix (Current Code)

I changed `gather()` to catch exceptions individually:

```python
# dask_simulation.py lines 263-275 (CURRENT - BAD)
def gather(self, futures: list[Future[SimReturn]]) -> list[SimReturn]:
    dask_futures = [f.wrapped for f in futures]

    # PROBLEM: This is SEQUENTIAL - waits for each future one at a time!
    results = []
    for future in dask_futures:
        try:
            results.append(future.result())  # <-- BLOCKS here for each future
        except Exception as e:
            results.append(e)
    return results
```

**Why this is terrible:** With N futures, this waits for future[0] to complete, then future[1], then future[2]... Instead of waiting for all N in parallel.

### Performance Impact

```
Scenario: 8 aggregation futures, each takes 30 seconds

PARALLEL (original):
  All 8 start immediately, all complete around t=30s
  Total time: ~30 seconds

SEQUENTIAL (my bad fix):
  future[0]: wait 30s
  future[1]: wait 30s (starts at t=0, but we don't check until t=30)
  ...
  Total time: up to 8 × 30s = 240 seconds (worst case)

  Actually slightly better because futures run concurrently in Dask,
  but we still serialize the CHECKING, adding latency.
```

### Measured Impact

| Metric | Expected | Actual | Loss |
|--------|----------|--------|------|
| Throughput (8 workers) | 2.0 trials/min | 1.0 trials/min | **50%** |
| Per-worker efficiency | 0.25 trials/worker/min | 0.125 | **50%** |

## The Correct Fix

Use `as_completed()` to check futures as they finish, or use Dask's built-in error handling:

### Option A: Use `as_completed()` with timeout

```python
from dask.distributed import as_completed

def gather(self, futures: list[Future[SimReturn]]) -> list[SimReturn]:
    dask_futures = [f.wrapped for f in futures]

    # Create mapping to preserve order
    future_to_idx = {f: i for i, f in enumerate(dask_futures)}
    results = [None] * len(dask_futures)

    # Wait for all futures in parallel, handle errors individually
    for future in as_completed(dask_futures):
        idx = future_to_idx[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            results[idx] = e

    return results
```

### Option B: Gather with errors='skip' then check individually

```python
def gather(self, futures: list[Future[SimReturn]]) -> list[SimReturn]:
    dask_futures = [f.wrapped for f in futures]

    # Wait for all futures in parallel first
    # Use wait() to let all complete without raising
    from dask.distributed import wait
    wait(dask_futures)

    # Now check results - all futures are done, so .result() returns immediately
    results = []
    for future in dask_futures:
        try:
            results.append(future.result())
        except Exception as e:
            results.append(e)
    return results
```

### Option C: Parallel gather with concurrent.futures (if Dask lacks good option)

```python
from concurrent.futures import ThreadPoolExecutor, as_completed as cf_as_completed

def gather(self, futures: list[Future[SimReturn]]) -> list[SimReturn]:
    dask_futures = [f.wrapped for f in futures]

    def get_result(future):
        try:
            return future.result()
        except Exception as e:
            return e

    # Check all futures in parallel using thread pool
    with ThreadPoolExecutor(max_workers=len(dask_futures)) as executor:
        result_futures = {executor.submit(get_result, f): i for i, f in enumerate(dask_futures)}
        results = [None] * len(dask_futures)
        for cf_future in cf_as_completed(result_futures):
            idx = result_futures[cf_future]
            results[idx] = cf_future.result()

    return results
```

## Recommended Fix

**Option B** is simplest and most correct:

```python
def gather(self, futures: list[Future[SimReturn]]) -> list[SimReturn]:
    """Gather results from submitted tasks.

    Waits for all futures in parallel, then collects results.
    Failed tasks return their Exception object instead of raising.
    """
    from dask.distributed import wait

    dask_futures = [f.wrapped for f in futures]

    # Wait for ALL futures to complete (parallel wait)
    wait(dask_futures)

    # All futures are now done - collect results (no blocking)
    results = []
    for future in dask_futures:
        try:
            results.append(future.result())
        except Exception as e:
            results.append(e)
    return results
```

This:
1. Waits for all futures in parallel (no throughput loss)
2. Returns exceptions as values (fixes the original crash bug)
3. Preserves order (matches original behavior)

## Verification Plan

After fix:
1. Run calibration job with 8 workers
2. Expected throughput: ~2.0 trials/min (vs current 1.0)
3. Verify NaN errors don't crash threads (check for "Error in trial loop" logs)

## Files to Change

| File | Change |
|------|--------|
| `modelops/src/modelops/services/dask_simulation.py` | Replace sequential gather with parallel wait + collect |
