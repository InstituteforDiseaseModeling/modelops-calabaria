# Throughput Diagnostic Plan

**Date:** 2026-01-24
**Status:** IMPLEMENTED - Ready to Run
**In Response To:** Colleague's diagnostic suggestions

## Implementation Status

The following instrumentation has been added:

| Measurement | Location | Log Prefix |
|-------------|----------|------------|
| Sim task timing | `dask_simulation.py:_worker_run_task` | `SIM_TIMING:` |
| Agg task timing | `dask_simulation.py:_worker_run_aggregation_direct` | `AGG_TIMING:` |
| Payload sizes | `dask_simulation.py:_inline_bytes()` | (in AGG_TIMING) |
| Worker locality | `dask_simulation.py:_worker_run_aggregation_direct` | (in AGG_TIMING) |
| Performance report | `wire.py` | Set `ENABLE_PERF_REPORT=1` |

### How to Run Diagnostics

```bash
# 1. Build and deploy with instrumentation
cd /Users/vsb/projects/work/modelops-core/modelops
gh workflow run docker-build.yml --ref main -f push_images=true
# Wait for build...
make rollout-images

# 2. Enable performance report for the job
# Edit your job submission to include:
#   env:
#     - name: ENABLE_PERF_REPORT
#       value: "1"

# 3. Run calibration (~30-50 trials)
cd /Users/vsb/projects/work/typhoidsim-calib-modelops
make calib-submit

# 4. Analyze logs
kubectl logs job/calib-XXXXX | python scripts/analyze_perf_logs.py

# 5. Get performance report HTML
kubectl cp calib-XXXXX:/tmp/dask-perf-report-*.html ./perf-report.html
```

### Log Format

**SIM_TIMING:**
```
SIM_TIMING: param_id=abc12345 seed=42 duration_ms=1234.5 worker=tcp://10.0.0.1:8786
```

**AGG_TIMING:**
```
AGG_TIMING: run_id=abc123 param_id=def45678 target=my_target n_sim_returns=10 n_outputs=15 total_inline_mb=2.34 max_single_kb=156.2 duration_ms=567.8 worker=tcp://10.0.0.2:8786
```

### Analysis Script

```bash
kubectl logs job/calib-XXXXX | python scripts/analyze_perf_logs.py
```

Outputs:
- Sim/Agg timing: median, p95
- Payload sizes: median total_inline_mb
- Split-agg rate: % of trials where targets ran on different workers

## 1) Workload Facts (Known)

| Metric | Value | Source |
|--------|-------|--------|
| n_replicates per trial | 10 | `Makefile:N_REPLICATES := 10` |
| Targets per trial | 2 | Configured in calibration job |
| Tasks per trial (optimized) | 12 | 10 sims + 2 aggs |
| Tasks per trial (old) | 22 | 10 sims × 2 targets + 2 aggs |
| Concurrency model | ThreadPoolExecutor | `wire.py:318` |
| Parallel threads | min(batch_size, actual_workers) | `wire.py:116` |
| With 8 workers | 8 concurrent trials | Each thread asks for 1 param, runs full trial |

### Trial Loop Code (wire.py lines 200-320)

```python
# Each thread runs this loop independently:
def worker_loop(worker_id):
    while not stop_flag.is_set() and not adapter.finished():
        params = adapter.ask(n=1)[0]  # Get one trial

        # Optimized path: sims once, aggs for each target
        sim_futures = sim_service.submit_replicates(replicate_set, run_id=run_id)

        agg_futures = []
        for target in targets_to_run:  # 2 targets
            agg_future = sim_service.submit_aggregation(sim_futures, target, ...)
            agg_futures.append((target, agg_future))

        # Single gather for all aggregations
        all_agg_results = sim_service.gather([f for _, f in agg_futures])

        adapter.tell([final_result])  # Report result
```

**Unknown (need to measure):**
- Median/p95 runtime of `_worker_run_task` (simulation)
- Median/p95 runtime of `_worker_run_aggregation_direct` (aggregation)

## 2) Cluster Configuration (Known)

| Setting | Value | Source |
|---------|-------|--------|
| Worker pods | 2 (test) / 4 (default) | `workspace.yaml:replicas` |
| Workers per pod | 4 | `--nworkers 4` |
| Threads per worker | 1 | `--nthreads 1` |
| Total Dask workers | 8 (test) | 2 pods × 4 workers |
| Memory per pod | 12 GiB | `workspace.yaml:resources.limits.memory` |
| CPU per pod | 4 cores | `workspace.yaml:resources.limits.cpu` |
| Memory spill threshold | 90% | `DASK_WORKER__MEMORY__TARGET` |

**Unknown (need to check):**
- Actual CPU throttling (kubectl describe pod)
- Worker logs for spilling/pausing
- "unmanaged memory" warnings

## 3) SimReturn Size Analysis

### Structure

```python
@dataclass(frozen=True)
class SimReturn:
    task_id: str
    outputs: Mapping[str, TableArtifact]  # ← This is the big one
    error: Optional[ErrorInfo] = None
    error_details: Optional[TableArtifact] = None
    logs_ref: Optional[str] = None
    metrics: Optional[Mapping[str, float]] = None
    cached: bool = False

@dataclass(frozen=True)
class TableArtifact:
    content_type: str = "application/vnd.apache.arrow.stream"
    size: int = 0
    inline: Optional[bytes] = None  # ← Up to 512KB per artifact
    ref: Optional[str] = None
    checksum: str = ""

INLINE_CAP = 524288  # 512KB
```

### Potential Size

With typhoidsim extractors producing monitor outputs:
- Multiple age bins (appears to be ~7 based on extraction code)
- Annual incidence data
- Each `TableArtifact.inline` can be up to **512KB**
- If N outputs: up to N × 512KB per SimReturn
- With 10 replicates: up to N × 512KB × 10 = **5MB+ per aggregation**

**Unknown (need to measure):**
- Actual number of outputs per SimReturn
- Actual size of each output (likely much smaller than 512KB cap)
- `len(cloudpickle.dumps(sim_returns[0]))` sample

## 4) The Aggregation Data Movement Problem

### Current Pattern

```python
def _worker_run_aggregation_direct(*sim_returns, target_ep, bundle_ref):
    # sim_returns are MATERIALIZED here - all 10 replicates on ONE worker
    agg_task = AggregationTask(
        bundle_ref=bundle_ref,
        target_entrypoint=target_ep,
        sim_returns=list(sim_returns),  # All 10 SimReturns
    )
    return worker.modelops_exec_env.run_aggregation(agg_task)
```

### Data Flow

```
sim-0 runs on worker A ─┐
sim-1 runs on worker B ─┤
sim-2 runs on worker C ─┤
sim-3 runs on worker D ─┼──► agg-target1 on worker X (needs ALL data)
sim-4 runs on worker A ─┤
...                     │
sim-9 runs on worker B ─┘

                        └──► agg-target2 on worker Y (needs ALL data AGAIN)
```

With 2 targets: **same data potentially transferred twice** to different workers.

## 5) Proposed Diagnostics

### A. Enable Performance Report

Add to wire.py around calibration run:

```python
from dask.distributed import performance_report

with performance_report(filename="dask-report.html"):
    _run_parallel_trials(...)
```

This will show:
- Scheduler "processing" vs "transfer" vs "compute"
- Task stream (idle? queued?)
- Bandwidth plots

### B. Add Logging to Aggregation

In `_worker_run_aggregation_direct`:

```python
import cloudpickle
import logging

logger = logging.getLogger(__name__)

def _worker_run_aggregation_direct(*sim_returns, target_ep, bundle_ref):
    # Diagnostic: measure data size
    sample_size = len(cloudpickle.dumps(sim_returns[0])) if sim_returns else 0
    total_estimate = sample_size * len(sim_returns)

    logger.info(
        f"Aggregation: {len(sim_returns)} sim_returns, "
        f"sample_size={sample_size/1024:.1f}KB, "
        f"total_estimate={total_estimate/1024:.1f}KB"
    )
    # ... rest of function
```

### C. Check Worker Locality

```python
# Add to wire.py after submitting sim_futures:
client = sim_service.client
dask_futures = [f.wrapped for f in sim_futures]
who_has = client.who_has(dask_futures)
logger.info(f"SimReturn locations: {who_has}")
```

### D. Controlled Experiments

**Test A: Sims only (no aggregation)**
- Skip aggregation, just run 10 sims
- If throughput jumps → aggregation/transfer is bottleneck

**Test B: Single target vs two targets**
- Same sims, 1 target vs 2 targets
- If 2-target is ~2× slower → data duplication to different agg workers

## 6) Likely Diagnosis (Hypothesis)

Based on colleague's analysis: **aggregation data movement dominates**.

Evidence:
- Per-worker efficiency dropped 13% despite 42% fewer tasks
- Multi-target pattern forces same data to potentially different workers
- No `resources={'aggregation': 1}` on optimized path → Dask picks arbitrary workers

### Potential Fixes

1. **Pin aggregations to locality**: Use `who_has()` + `workers=[...]` to run aggregation on worker that has most sim data

2. **Multi-target single aggregation**: Compute all target losses in one pass:
   ```python
   def _worker_run_multi_target_aggregation(*sim_returns, targets, bundle_ref):
       results = {}
       for target in targets:
           results[target] = compute_loss(sim_returns, target)
       return results
   ```
   This transfers data once instead of N times.

## 7) Immediate Next Steps

1. **Enable performance_report** for a 50-trial run
2. **Add cloudpickle size logging** to aggregation
3. **Run Test A** (sims only) to baseline
4. **Run Test B** (1 target vs 2 targets) to quantify duplication cost

If transfer dominates in performance report, implement fix #2 (multi-target aggregation).

## Response to Colleague

Here's what we know:

**Workload:** 10 replicates, 2 targets, 8 concurrent trial threads, each thread submits 10 sims + 2 aggs.

**SimReturn size:** Unknown but potentially large - each extractor output can be up to 512KB inline Arrow. Need to measure with cloudpickle.

**Cluster:** 8 workers (2 pods × 4), 1 thread each, 12GB memory/pod, 4 CPU/pod.

**Suspicion matches yours:** The `*sim_returns` pattern likely forces all 10 replicates to one worker per aggregation. With 2 targets, we may be transferring the same ~XMB twice.

Will run diagnostics and report back with:
- Performance report (transfer vs compute breakdown)
- Actual SimReturn sizes
- Test A/B results
