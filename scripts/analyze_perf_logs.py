#!/usr/bin/env python3
"""Analyze calibration performance logs to identify throughput bottlenecks.

Usage:
    kubectl logs job/calib-XXXXX | python analyze_perf_logs.py

Or save logs first:
    kubectl logs job/calib-XXXXX > logs.txt
    python analyze_perf_logs.py < logs.txt

Outputs:
    1. Sim task timing: median, p95
    2. Agg task timing: median, p95
    3. Payload sizes: median total_inline_mb per aggregation
    4. Split-agg rate: % of trials where targets ran on different workers
"""

import re
import sys
from collections import defaultdict
from statistics import median, quantiles


def parse_logs(lines):
    """Parse SIM_TIMING and AGG_TIMING log lines."""
    sim_times = []
    agg_times = []
    agg_payloads = []  # total_inline_mb per aggregation

    # Track which worker ran each target for each run_id
    # {run_id: {target: worker_address}}
    agg_locations = defaultdict(dict)

    sim_pattern = re.compile(
        r'SIM_TIMING:.*duration_ms=(\d+\.?\d*)'
    )
    agg_pattern = re.compile(
        r'AGG_TIMING: run_id=(\S+) param_id=(\S+) target=(\S+) '
        r'n_sim_returns=(\d+) n_outputs=(\d+) total_inline_mb=(\d+\.?\d*) '
        r'.*duration_ms=(\d+\.?\d*) worker=(\S+)'
    )

    for line in lines:
        # Parse sim timing
        sim_match = sim_pattern.search(line)
        if sim_match:
            sim_times.append(float(sim_match.group(1)))
            continue

        # Parse agg timing
        agg_match = agg_pattern.search(line)
        if agg_match:
            run_id = agg_match.group(1)
            target = agg_match.group(3)
            total_inline_mb = float(agg_match.group(6))
            duration_ms = float(agg_match.group(7))
            worker = agg_match.group(8)

            agg_times.append(duration_ms)
            agg_payloads.append(total_inline_mb)
            agg_locations[run_id][target] = worker

    return sim_times, agg_times, agg_payloads, agg_locations


def compute_stats(values, name):
    """Compute and print statistics for a list of values."""
    if not values:
        print(f"  {name}: NO DATA")
        return

    p50 = median(values)
    try:
        p95 = quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    except Exception:
        p95 = max(values)

    print(f"  {name}: median={p50:.1f}, p95={p95:.1f}, n={len(values)}")


def compute_split_rate(agg_locations):
    """Compute % of trials where target A and B ran on different workers."""
    if not agg_locations:
        return 0.0, 0

    total_trials = 0
    split_trials = 0

    for run_id, targets in agg_locations.items():
        if len(targets) < 2:
            continue

        total_trials += 1
        workers = list(targets.values())
        if len(set(workers)) > 1:
            split_trials += 1

    if total_trials == 0:
        return 0.0, 0

    return (split_trials / total_trials) * 100, total_trials


def main():
    print("Parsing logs from stdin...")
    print("-" * 60)

    lines = sys.stdin.readlines()
    sim_times, agg_times, agg_payloads, agg_locations = parse_logs(lines)

    print("\n## Task Runtimes (ms)")
    compute_stats(sim_times, "sim_duration_ms")
    compute_stats(agg_times, "agg_duration_ms")

    print("\n## Payload Sizes (MB)")
    compute_stats(agg_payloads, "total_inline_mb")

    print("\n## Locality Analysis")
    split_rate, n_trials = compute_split_rate(agg_locations)
    print(f"  split_agg_rate: {split_rate:.1f}% (of {n_trials} trials with 2+ targets)")

    if split_rate > 30:
        print("  WARNING: High split rate suggests data duplication between workers")
    elif n_trials > 0:
        print("  OK: Most aggregations co-located")

    print("\n## Summary")
    if agg_payloads:
        median_payload = median(agg_payloads)
        if median_payload > 5:
            print(f"  Large payloads ({median_payload:.1f}MB) - consider multi-target agg")
        elif median_payload > 1:
            print(f"  Moderate payloads ({median_payload:.1f}MB) - check perf report for transfer")
        else:
            print(f"  Small payloads ({median_payload:.1f}MB) - transfer unlikely bottleneck")

    if agg_times and sim_times:
        agg_median = median(agg_times)
        sim_median = median(sim_times)
        if agg_median > sim_median * 0.5:
            print(f"  Agg time ({agg_median:.0f}ms) significant vs sim ({sim_median:.0f}ms)")
        else:
            print(f"  Agg time ({agg_median:.0f}ms) small vs sim ({sim_median:.0f}ms)")


if __name__ == "__main__":
    main()
