"""
scheduler.py  —  OS roles 3, 4, 6

Discrete-event simulation of request scheduling under two policies:
  - PRIORITY : highest urgency score first, non-preemptive, FIFO tiebreak
  - FCFS     : first-come-first-served (the baseline to beat)

WHY SIMULATION, NOT REAL THREADS
Running real concurrent requests would let OS thread jitter, GC pauses and
cache effects dominate the signal. Here the real measured model latencies
are used as fixed service times, so the ONLY variable between the two runs
is the scheduling policy. That is a controlled comparison, which is what
the "prioritized queue vs FCFS" claim needs.

RESOURCE MODEL
Each compute resource (fast_local, smart_cloud) is an independent pool of
identical workers with its OWN ready queue — exactly like separate CPUs
with separate run queues. This matters: with one shared queue, a request
blocked on a busy resource also blocks every request behind it that
targets an idle resource, which silently inflates both policies'
response times and destroys the comparison.

SCHEDULING OVERHEAD (role 6)
Simulated time says nothing about what the queue discipline itself costs.
We therefore measure real wall-clock nanoseconds spent inside heap
push/pop operations and report it separately. This is the evidence that
prioritisation is cheap relative to the service times it reorders.
"""

import copy
import heapq
import itertools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SchedulingPolicy(Enum):
    PRIORITY = "priority"
    FCFS = "fcfs"


@dataclass
class Request:
    """One strategy-prediction request for one driver at one lap."""
    id: str
    arrival_time_ms: float
    urgency_score: float           # from urgency.py — the priority key
    resource: str                  # "fast_local" | "smart_cloud" — from router.py
    service_time_ms: float         # measured model latency + data-source cost
    is_urgent: bool = False
    # provenance, carried through for the predictions_log and for plots
    race_id: str = ""
    driver: str = ""
    lap_number: int = 0
    data_source: str = ""
    burst_reason: str = ""

    # filled in by the simulation
    start_time_ms: Optional[float] = None
    finish_time_ms: Optional[float] = None

    @property
    def wait_time_ms(self) -> float:
        return self.start_time_ms - self.arrival_time_ms

    @property
    def response_time_ms(self) -> float:
        return self.finish_time_ms - self.arrival_time_ms


@dataclass
class ResourcePool:
    """A compute resource with a fixed number of identical workers.

    free_at[i] is the simulation time at which worker i becomes idle.
    """
    name: str
    num_workers: int
    free_at: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.free_at:
            self.free_at = [0.0] * self.num_workers

    def earliest_free_worker(self):
        idx = min(range(self.num_workers), key=lambda i: self.free_at[i])
        return idx, self.free_at[idx]

    def occupy(self, idx: int, start: float, duration: float):
        self.free_at[idx] = start + duration


def run_simulation(
    requests: List[Request],
    policy: SchedulingPolicy,
    fast_workers: int = 1,
    smart_workers: int = 1,
) -> Dict:
    """Run the request stream under one policy. Input list is not mutated,
    so the identical stream can be replayed under both policies."""
    reqs = [copy.deepcopy(r) for r in requests]
    if not reqs:
        raise ValueError("empty request stream")

    pools = {
        "fast_local": ResourcePool("fast_local", fast_workers),
        "smart_cloud": ResourcePool("smart_cloud", smart_workers),
    }
    ready: Dict[str, list] = {name: [] for name in pools}
    counter = itertools.count()

    overhead_ns = 0
    queue_ops = 0
    # Peak queue depth per resource. Average utilisation over a whole race
    # is near zero (sub-millisecond service times against 90-second laps),
    # which is true but says nothing. Contention only exists inside the
    # burst windows, and queue depth is what actually measures it.
    peak_depth = {name: 0 for name in ("fast_local", "smart_cloud")}

    def push(r: Request):
        nonlocal overhead_ns, queue_ops
        t0 = time.perf_counter_ns()
        if policy == SchedulingPolicy.PRIORITY:
            # heapq is a min-heap: negate so highest urgency pops first.
            key = (-r.urgency_score, next(counter))
        else:
            key = (r.arrival_time_ms, next(counter))
        heapq.heappush(ready[r.resource], (key, r))
        overhead_ns += time.perf_counter_ns() - t0
        queue_ops += 1
        if len(ready[r.resource]) > peak_depth[r.resource]:
            peak_depth[r.resource] = len(ready[r.resource])

    def pop(resource: str) -> Request:
        nonlocal overhead_ns, queue_ops
        t0 = time.perf_counter_ns()
        _, r = heapq.heappop(ready[resource])
        overhead_ns += time.perf_counter_ns() - t0
        queue_ops += 1
        return r

    reqs.sort(key=lambda r: r.arrival_time_ms)
    ptr = 0
    n = len(reqs)
    completed = 0
    t = reqs[0].arrival_time_ms

    while completed < n:
        # 1. Admit everything that has arrived by now.
        while ptr < n and reqs[ptr].arrival_time_ms <= t:
            push(reqs[ptr])
            ptr += 1

        # 2. Dispatch on every resource independently. Each pool drains its
        #    own queue for as long as it has a worker free at time t.
        for name, pool in pools.items():
            while ready[name]:
                idx, free_at = pool.earliest_free_worker()
                if free_at > t:
                    break
                r = pop(name)
                r.start_time_ms = t
                r.finish_time_ms = t + r.service_time_ms
                pool.occupy(idx, t, r.service_time_ms)
                completed += 1

        if completed >= n:
            break

        # 3. Nothing more can run at time t. Advance to the sooner of the
        #    next arrival or the next worker freeing up on a pool that
        #    actually has work waiting.
        candidates = []
        if ptr < n:
            candidates.append(reqs[ptr].arrival_time_ms)
        for name, pool in pools.items():
            if ready[name]:
                candidates.append(pool.earliest_free_worker()[1])
        future = [c for c in candidates if c > t]
        if not future:
            raise RuntimeError(
                f"simulation stalled at t={t} with {n - completed} requests unserved"
            )
        t = min(future)

    return _collect_metrics(reqs, pools, policy, overhead_ns, queue_ops,
                            peak_depth)


PEAK_WINDOW_MS = 1000.0


def _peak_window_utilization(reqs: List[Request], num_workers: int,
                             window_ms: float = PEAK_WINDOW_MS) -> float:
    """Busiest `window_ms` slice: total service time executed inside the
    window, over the window's worker-capacity. Service time is split
    across whichever buckets the execution actually spans, so a long
    request is not attributed entirely to the bucket it started in."""
    if not reqs:
        return 0.0
    buckets: Dict[int, float] = {}
    for r in reqs:
        start, end = r.start_time_ms, r.finish_time_ms
        b0, b1 = int(start // window_ms), int(end // window_ms)
        for b in range(b0, b1 + 1):
            lo = max(start, b * window_ms)
            hi = min(end, (b + 1) * window_ms)
            if hi > lo:
                buckets[b] = buckets.get(b, 0.0) + (hi - lo)
    if not buckets:
        return 0.0
    return max(buckets.values()) / (num_workers * window_ms)


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(pct * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _summarize(subset: List[Request]) -> Dict:
    if not subset:
        return {"count": 0, "avg_wait_ms": 0.0, "max_wait_ms": 0.0,
                "avg_response_ms": 0.0, "p95_response_ms": 0.0, "p99_response_ms": 0.0}
    waits = sorted(r.wait_time_ms for r in subset)
    resp = sorted(r.response_time_ms for r in subset)
    return {
        "count": len(subset),
        "avg_wait_ms": sum(waits) / len(waits),
        "max_wait_ms": waits[-1],                 # starvation indicator
        "avg_response_ms": sum(resp) / len(resp),
        "p95_response_ms": _percentile(resp, 0.95),
        "p99_response_ms": _percentile(resp, 0.99),
    }


def _collect_metrics(reqs, pools, policy, overhead_ns, queue_ops,
                     peak_depth) -> Dict:
    sim_start = min(r.arrival_time_ms for r in reqs)
    sim_end = max(r.finish_time_ms for r in reqs)
    span = max(sim_end - sim_start, 1e-9)

    utilization = {}
    busy_utilization = {}
    for name, pool in pools.items():
        mine = [r for r in reqs if r.resource == name]
        busy = sum(r.service_time_ms for r in mine)
        utilization[name] = busy / (pool.num_workers * span)
        # PEAK utilisation: the busiest one-second window anywhere in the
        # run. Whole-race average utilisation is near zero by construction
        # and tells you nothing about whether the resource was ever
        # saturated; the peak is what sizing decisions are made from.
        busy_utilization[name] = _peak_window_utilization(mine, pool.num_workers)

    urgent = [r for r in reqs if r.is_urgent]
    non_urgent = [r for r in reqs if not r.is_urgent]

    return {
        "policy": policy.value,
        "overall": _summarize(reqs),
        "urgent": _summarize(urgent),
        "non_urgent": _summarize(non_urgent),
        "utilization": utilization,
        "busy_utilization": busy_utilization,
        "peak_queue_depth": peak_depth,
        "makespan_ms": span,
        "scheduling_overhead": {
            "total_ns": overhead_ns,
            "queue_ops": queue_ops,
            "ns_per_op": overhead_ns / queue_ops if queue_ops else 0.0,
            "us_per_request": (overhead_ns / len(reqs)) / 1000.0,
            # overhead as a fraction of the useful work it schedules
            "pct_of_service_time": (
                (overhead_ns / 1e6) / sum(r.service_time_ms for r in reqs) * 100.0
            ),
        },
        "requests": reqs,
    }


def compare_policies(requests: List[Request], fast_workers: int = 1,
                     smart_workers: int = 1) -> Dict:
    """Run PRIORITY and FCFS over the identical stream. This is the core
    experiment (OS role 5)."""
    return {
        "priority": run_simulation(requests, SchedulingPolicy.PRIORITY,
                                   fast_workers, smart_workers),
        "fcfs": run_simulation(requests, SchedulingPolicy.FCFS,
                               fast_workers, smart_workers),
    }
