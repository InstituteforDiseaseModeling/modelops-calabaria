# Calibration Throughput Analysis

**Date:** 2026-01-24
**Status:** Optimization Complete - Results Mixed
**Related:** parallel_trial_execution.md, gather_performance_regression.md

## Executive Summary

We implemented the `submit_replicates()` + `submit_aggregation()` optimization to eliminate duplicate simulation execution (running sims twice for 2 targets). The optimization is now **working correctly**, but **per-worker efficiency did not improve** as expected.

| Metric | Before | After | Expected |
|--------|--------|-------|----------|
| Tasks per trial | 12 | 7 | 7 ✓ |
| Per-worker efficiency | 0.150 | 0.131 | 0.26+ |
| Improvement | — | **-13%** | **+70%** |

The optimization correctly reduces computational waste (42% fewer tasks), but overhead (serialization, scheduling, coordination) appears to dominate at our current scale. Further profiling is needed to understand why task reduction didn't translate to throughput gains.

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

The primary throughput bottleneck was **duplicate simulation execution** (running sims twice for 2 targets). We fixed this with the `submit_replicates()` + `submit_aggregation()` pattern, reducing tasks per trial from 12 to 7 (42% reduction).

**However, the expected 1.7x throughput improvement did not materialize.** Per-worker efficiency actually dropped 13% (0.150 → 0.131 trials/worker/min). This suggests:

1. **Overhead dominates at small scale**: With only 5 replicates, serialization/scheduling overhead is proportionally large
2. **Coordination costs matter**: The new pattern adds coordination between sim submission and aggregation submission
3. **Dask locality effects**: Removing aggregation resource constraints may have hurt data locality

The low CPU percentage in Dask dashboard is misleading - actual pod utilization is 70%. The remaining 30% overhead is inherent to distributed execution (serialization, networking, scheduling).

**Next steps to improve throughput:**
- Profile to identify the new bottleneck (serialization? scheduling? network?)
- Test with larger replicate counts to amortize overhead
- Consider whether the optimization helps more at higher worker counts

## Measured Results (2026-01-24)

**Status:** FIXED - Optimized pattern now working

### Summary Table (Normalized Per Worker)

| Configuration | Workers | Raw Throughput | Per-Worker Efficiency | Change vs Baseline |
|---------------|---------|----------------|----------------------|-------------------|
| **Baseline** (duplicate sims) | 20 | 3.0 trials/min | 0.150 trials/worker/min | — |
| **Optimized** (sequential gather bug) | 8 | 1.0 trials/min | 0.125 trials/worker/min | **-17%** |
| **Optimized** (wait() fix) | 8 | 1.05 trials/min | 0.131 trials/worker/min | **-13%** |

### Key Finding

**Per-worker efficiency is ~13% lower than baseline**, despite eliminating duplicate simulations. The optimization reduced tasks per trial from 12 to 7 (42% reduction), but this did not translate to improved per-worker throughput.

### Detailed Measurements

#### Baseline Configuration (20 workers, duplicate sims)
```
Job: calib-* (pre-optimization)
Workers: 20 (5 pods × 4 workers)
Throughput: 3.0 trials/min
Tasks per trial: 12 (5 sims × 2 targets + 2 aggs)
Per-worker efficiency: 0.150 trials/worker/min
```

#### Optimized with Sequential Gather Bug (8 workers)
```
Job: calib-9e98d08a
Workers: 8 (2 pods × 4 workers)
Throughput: 1.0 trials/min
Tasks per trial: 7 (5 sims + 2 aggs)
Per-worker efficiency: 0.125 trials/worker/min

BUG: gather() iterated sequentially through futures with .result(),
causing head-of-line blocking. See gather_performance_regression.md.
```

#### Optimized with wait() Fix (8 workers)
```
Job: calib-574e0975
Workers: 8 (2 pods × 4 workers)
Throughput: ~1.05 trials/min (estimated from burst patterns)
Tasks per trial: 7 (5 sims + 2 aggs)
Per-worker efficiency: 0.131 trials/worker/min

Evidence of parallel wait working:
- Trials complete in bursts (8 trials within 2 seconds)
- No sequential completion pattern
- Progress: 8→16→24→32... in rapid succession
```

### Why Per-Worker Efficiency Didn't Improve

Despite cutting tasks per trial by 42%, per-worker efficiency dropped 13%. Possible causes:

1. **Overhead dominates small jobs**: With only 5 replicates per trial, serialization/scheduling overhead is proportionally larger
2. **Thread coordination costs**: The optimized pattern submits sims once, then multiple aggs - this may add coordination overhead
3. **Fewer workers = less parallelism**: 8 workers vs 20 means less concurrent task execution
4. **Data locality**: Removing aggregation resource constraint may have hurt worker locality

### Conclusion

The `submit_replicates()` + `submit_aggregation()` optimization is **working correctly** (no duplicate sims, parallel gather), but it does **not improve per-worker efficiency** under current conditions.

**To realize the theoretical 2x improvement:**
- Scale to more workers (the task reduction helps more at scale)
- Increase replicates per trial (amortize overhead)
- Profile to identify new bottlenecks (serialization? scheduling?)

## Implementation History

### Phase 1: Identify Duplicate Simulations
- Discovered each target triggered redundant sim submissions
- Created `submit_replicates()` + `submit_aggregation()` pattern

### Phase 2: Initial Deployment (Failed)
- Commit `afe0e10` deployed optimized pattern
- Throughput dropped to ~0.5 trials/min
- Reverted in commit `b99fc76`

### Phase 3: Fix Task Key Collisions
- Added `run_id` parameter to prevent key collisions between threads
- Commit `*` in `dask_simulation.py`

### Phase 4: Fix Aggregation Resource Constraint
- Removed `resources={'aggregation': 1}` from `submit_aggregation()`
- This was blocking worker locality (data had to move to workers with resource)

### Phase 5: Fix Graph Pressure
- Limited parallel threads to match actual worker count
- Prevents overwhelming scheduler with too many concurrent submissions

### Phase 6: Fix Sequential Gather (Critical)
- **Bug:** `gather()` was calling `.result()` sequentially on each future
- **Impact:** 50% throughput loss from head-of-line blocking
- **Fix:** Use `wait(dask_futures)` then collect results (non-blocking)
- See `gather_performance_regression.md` for full analysis

## Files Modified

| File | Change |
|------|--------|
| `dask_simulation.py` | Fixed `submit_replicates()`, `submit_aggregation()`, and `gather()` |
| `wire.py` | Use optimized pattern, limit threads to worker count |

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
