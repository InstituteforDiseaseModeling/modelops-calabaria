# Calibration Throughput Analysis

**Date:** 2026-01-24
**Status:** Investigation Complete
**Related:** parallel_trial_execution.md

## Executive Summary

Despite implementing parallel trial execution (1.67x improvement), calibration throughput remains limited at ~4.5 trials/min with low apparent CPU utilization (~5-10% per worker in Dask dashboard). This report analyzes the root causes and identifies optimization opportunities.

## Current Architecture

### Cluster Configuration

| Component | Configuration |
|-----------|--------------|
| Worker pods | 5 |
| Workers per pod | 4 (`--nworkers 4`) |
| Threads per worker | 1 (`--nthreads 1`) |
| **Total Dask workers** | **20** |
| CPU per pod | 6 cores |
| Memory per pod | 6 GiB |
| Aggregation resource | 1 per worker |

### Task Structure Per Trial

Current implementation (with 2 targets, 5 replicates):

```
Thread N in job pod:
  │
  ├─► Target 1:
  │     ├─ Submit 5 simulation tasks ──► Dask workers
  │     ├─ Submit 1 aggregation task ──► Dask workers
  │     └─ WAIT (gather) ◄─────────────── blocks here
  │
  └─► Target 2:
        ├─ Submit 5 NEW simulation tasks ──► Dask workers  ← REDUNDANT!
        ├─ Submit 1 aggregation task ──────► Dask workers
        └─ WAIT (gather) ◄───────────────── blocks here
```

**Tasks per trial: 12** (5 sims × 2 targets + 2 aggs)

## Identified Bottlenecks

### 1. Duplicate Simulation Execution (CRITICAL)

**Problem:** Each target triggers a fresh set of simulation tasks, even though simulations are parameter-dependent, not target-dependent.

```python
# Current code (wire.py lines 216-218)
for target in targets_to_run:  # Loops 2 times
    future = sim_service.submit_replicate_set(replicate_set, target)  # Submits 5 NEW sims each time
    results = sim_service.gather([future])
```

**Impact:**
- Running 10 simulations per trial instead of 5
- **2x computational waste**
- Dask's `pure=False` prevents deduplication

**Solution:** Use `submit_replicates()` once, then `submit_aggregation()` for each target:

```python
# Optimized approach
sim_futures = sim_service.submit_replicates(replicate_set)  # 5 sims, once
for target in targets_to_run:
    agg_future = sim_service.submit_aggregation(sim_futures, target, ...)
    # Can even submit all aggs without waiting...
```

### 2. Sequential Target Evaluation

**Problem:** Targets are evaluated sequentially within each thread. Target 2 cannot start until Target 1's aggregation completes.

```
Timeline for 1 trial:
[──sim──][──sim──][──sim──][──sim──][──sim──][agg1][──sim──][──sim──][──sim──][──sim──][──sim──][agg2]
|←────────── Target 1 ──────────────────────────→||←────────── Target 2 ──────────────────────────→|
```

**Impact:**
- Thread spends 50% of time waiting for Target 1 before starting Target 2
- Workers sit idle between target evaluations

**Solution:** Submit all work upfront, gather once:

```python
# Submit all sims once
sim_futures = sim_service.submit_replicates(replicate_set)

# Submit all aggregations (no waiting between)
agg_futures = [
    sim_service.submit_aggregation(sim_futures, target, ...)
    for target in targets_to_run
]

# Single gather for all aggregations
results = sim_service.gather(agg_futures)
```

### 3. Worker Thread Limitation

**Problem:** Each Dask worker has `--nthreads 1`, meaning it can only execute 1 task at a time.

**Impact:**
- 20 workers = maximum 20 concurrent tasks
- With 16 parallel threads × 5 sims = 80 potential concurrent sims
- Only 20 can actually run simultaneously

**Analysis:**
- Single-threaded workers are intentional for GIL-heavy workloads
- typhoidsim likely uses NumPy/SciPy which release GIL
- Multi-threading within worker could help IF simulation releases GIL

### 4. Aggregation Resource Constraint

**Problem:** Aggregation tasks require `resources={'aggregation': 1}` and each worker only has 1 aggregation slot.

**Impact:**
- Only 20 aggregations can run simultaneously (1 per worker)
- If all 16 threads submit aggregations at once, 16 compete for 20 slots
- Aggregation serializes some work

**Rationale:** This constraint exists to prevent deadlock (see dask_simulation.py header comments).

### 5. Actual CPU Utilization Discrepancy

**Observation:** User sees 5-10% CPU in Dask dashboard, but `kubectl top` shows 70%:

| Metric | Value |
|--------|-------|
| Pod CPU usage | ~4200m (4.2 cores) |
| Pod CPU limit | 6 cores |
| **Pod utilization** | **70%** |
| Dashboard per-worker | 5-10% |

**Explanation:**
- Dashboard shows per-worker-thread CPU, not pod CPU
- 4 workers × ~10% = 40% worker utilization
- Remaining 30% is overhead: GC, serialization, networking, Python interpreter
- Actual compute is happening, but overhead is significant

### 6. I/O and Serialization Overhead

**Sources of I/O wait:**
1. Task serialization (pickle) to workers
2. Result serialization back from workers
3. Network transfer between pods
4. PostgreSQL round-trips for Optuna ask/tell

**Evidence:** Burst completions (multiple trials completing within milliseconds) followed by gaps suggest batched I/O patterns.

## Throughput Calculation

### Current State

```
Trials completed: 66 in ~22 minutes
Throughput: 3.0 trials/min

Tasks per trial: 12 (with duplicate sims)
Total tasks: 66 × 12 = 792 tasks in 22 min = 36 tasks/min
With 20 workers: 36/20 = 1.8 tasks/worker/min
Average task duration: 33 seconds
```

### After Fixing Duplicate Simulations

```
Tasks per trial: 7 (5 sims + 2 aggs)
Same task duration: 33 seconds
Potential throughput: 36 tasks/min / 7 tasks = 5.1 trials/min
Improvement: 1.7x
```

### After Parallel Target Evaluation

```
All targets can overlap:
- 5 sims run once
- 2 aggs can run in parallel (different workers)
Effective tasks: 5 sims + max(agg1, agg2) ≈ 6 task-equivalents
Potential throughput: 36/6 = 6.0 trials/min
Additional improvement: 1.2x
```

### Theoretical Maximum

```
With 20 workers, 1 task each, 33s/task:
20 tasks / 33s = 0.6 tasks/sec = 36 tasks/min

If each trial needs 5 sims (parallel) + 1 agg (sequential):
Time per trial = max(sim_time/workers_available) + agg_time
With full parallelism and instant agg: ~5-10 trials/min

Current: 3.0 trials/min
Optimized: ~6.0 trials/min (2x improvement possible)
```

## Recommendations

### High Impact (Implement First)

1. **Eliminate duplicate simulations**
   - Modify `_run_parallel_trials()` to use `submit_replicates()` once per trial
   - Use `submit_aggregation()` for each target against shared sim results
   - Expected improvement: **1.7x**

2. **Parallel target aggregation**
   - Submit all aggregation tasks before gathering any
   - Single gather call for all targets
   - Expected improvement: **1.2x**

### Medium Impact

3. **Increase worker count**
   - Scale `--replicas` or `--nworkers` per pod
   - More workers = more concurrent tasks
   - Trade-off: memory pressure

4. **Profile simulation code**
   - Determine if typhoidsim releases GIL
   - If yes, consider `--nthreads 2` per worker
   - Trade-off: potential contention

### Low Impact / Future

5. **Reduce Optuna round-trips**
   - Batch ask() for multiple trials
   - Trade-off: may affect TPE sampling quality

6. **Async aggregation**
   - Don't wait for aggregation before asking for next trial
   - Complex: requires tracking in-flight trials

## Conclusion

The primary throughput bottleneck is **duplicate simulation execution** (running sims twice for 2 targets). Fixing this alone would nearly double throughput. Combined with parallel target evaluation, a **2x improvement** is achievable without infrastructure changes.

The low CPU percentage in Dask dashboard is misleading - actual pod utilization is 70%. The remaining 30% overhead is inherent to distributed execution (serialization, networking, scheduling).

## Implementation Status

**BLOCKED**: The `submit_replicates()` + `submit_aggregation()` pattern in `DaskSimulationService` has bugs.

### Failed Attempt (2026-01-24)

Commit `afe0e10` attempted to use the optimized pattern but caused:
- Throughput dropped from ~3 trials/min to ~0.5 trials/min
- Most worker threads stuck waiting indefinitely
- Reverted in commit `b99fc76`

### Root Cause (To Investigate)

The `submit_replicates()` and `submit_aggregation()` methods in `dask_simulation.py` have a TODO noting they lack integration tests. Possible issues:
1. Key collision between threads submitting the same param_id's tasks
2. Aggregation resource deadlock with multiple threads
3. Dask dependency graph issues when separating sim submission from agg submission
4. `gather()` type mismatch (expects `Future[SimReturn]`, given `Future[AggregationReturn]`)

### Next Steps

1. Add integration tests for `submit_replicates()` + `submit_aggregation()` pattern
2. Test with single-threaded execution first
3. Debug Dask task graph to identify blocking point
4. May need to fix `DaskSimulationService` before wire.py can use the optimized pattern

## Files to Modify (When Bugs Fixed)

| File | Change |
|------|--------|
| `dask_simulation.py` | Fix `submit_replicates()` + `submit_aggregation()` pattern |
| `wire.py` | Use optimized pattern once service is fixed |

## Appendix: Task Execution Timeline

```
Current (wasteful):
Thread 1: [ask][sim1-5 T1][agg T1][sim1-5 T2][agg T2][tell]
Thread 2: [ask][sim1-5 T1][agg T1][sim1-5 T2][agg T2][tell]
...
Total sim tasks: 16 threads × 10 sims = 160 sims per round

Optimized:
Thread 1: [ask][sim1-5][agg T1 + agg T2 parallel][tell]
Thread 2: [ask][sim1-5][agg T1 + agg T2 parallel][tell]
...
Total sim tasks: 16 threads × 5 sims = 80 sims per round (2x reduction)
```
