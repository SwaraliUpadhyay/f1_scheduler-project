"""
scheduler.py

Discrete-event simulation of request scheduling under two policies:
  - PRIORITY: highest urgency first (ties broken by arrival order)
  - FCFS: first-come-first-served (the baseline to beat)

Why discrete-event simulation instead of real threads/sleep(): running
real concurrent requests against real models would make response-time
comparisons noisy and non-reproducible (OS thread scheduling jitter,
GC pauses, etc. would dominate the signal you actually care about —
the scheduling POLICY's effect). A discrete-event simulation uses the
real measured latencies (from the ML lead's benchmark_speed results) as
fixed service times, so the ONLY variable between PRIORITY and FCFS runs
is the scheduling policy — a controlled comparison, which is what your
"prioritized queue vs FCFS" experiment claim needs.

Each compute resource (fast/local, smart/cloud) is modeled as a pool of
identical workers. A request occupies one worker for its service_time_ms;
when no worker is free, it waits in queue. This is the standard model
used in OS/queueing-theory scheduling simulators (directly analogous to
CPU cores handling processes).
"""

import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class SchedulingPolicy(Enum):
    PRIORITY = "priority"
    FCFS = "fcfs"


@dataclass
class Request:
    id: str
    arrival_time_ms: float
    urgency_score: float           # from urgency.py — used as priority key
    resource: str                  # "fast_local" or "smart_cloud" — from router.py
    service_time_ms: float         # real measured model latency for this resource
    is_urgent: bool = False        # for reporting urgent-vs-non-urgent breakdowns

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
    fast_local: typically 1 (a single local process/thread).
    smart_cloud: can be >1 to model cloud elasticity, if you want that
    comparison — start with 1 for a clean like-for-like baseline."""
    name: str
    num_workers: int
    # free_at[i] = simulation time worker i becomes free
    free_at: List[float] = field(default_factory=list)

    def __post_init__(self):
        if not self.free_at:
            self.free_at = [0.0] * self.num_workers

    def earliest_free_worker(self):
        idx = min(range(self.num_workers), key=lambda i: self.free_at[i])
        return idx, self.free_at[idx]

    def occupy(self, idx: int, start: float, duration: float):
        self.free_at[idx] = start + duration

    def busy_time_total(self, sim_end_ms: float) -> float:
        # crude but adequate: sum of (sim_end - idle_since) isn't tracked
        # per-interval here; utilization is computed from request service
        # times directly in run_simulation for accuracy instead.
        raise NotImplementedError


def run_simulation(
    requests: List[Request],
    policy: SchedulingPolicy,
    fast_workers: int = 1,
    smart_workers: int = 1,
) -> Dict:
    """
    Runs the full request stream through the given policy and returns
    per-request results plus aggregate metrics. Does not mutate the input
    Request objects (works on copies) so the same request list can be run
    under both policies for a fair comparison.
    """
    import copy
    reqs = [copy.deepcopy(r) for r in requests]

    pools = {
        "fast_local": ResourcePool("fast_local", fast_workers),
        "smart_cloud": ResourcePool("smart_cloud", smart_workers),
    }

    # Priority queue entries: (sort_key, tiebreak_seq, request)
    # PRIORITY: sort_key = -urgency_score (heapq is a min-heap, so negate
    #           for "highest urgency first"), tiebreak = arrival order.
    # FCFS:     sort_key = arrival_time_ms outright.
    counter = itertools.count()
    ready_heap = []

    reqs.sort(key=lambda r: r.arrival_time_ms)
    arrival_ptr = 0
    n = len(reqs)

    # We process in time order: advance to the next arrival or next
    # worker-free event, whichever is sooner, admitting arrived requests
    # into the ready queue and dispatching to free workers.
    completed = 0
    current_time = 0.0

    def push_ready(r: Request):
        if policy == SchedulingPolicy.PRIORITY:
            key = (-r.urgency_score, next(counter))
        else:
            key = (r.arrival_time_ms, next(counter))
        heapq.heappush(ready_heap, (key, r))

    while completed < n:
        # Admit all requests that have arrived by current_time.
        while arrival_ptr < n and reqs[arrival_ptr].arrival_time_ms <= current_time:
            push_ready(reqs[arrival_ptr])
            arrival_ptr += 1

        if not ready_heap:
            # Nothing ready yet — jump to the next arrival.
            if arrival_ptr < n:
                current_time = reqs[arrival_ptr].arrival_time_ms
                continue
            else:
                break  # shouldn't happen if completed < n, but guards infinite loop

        # Try to dispatch the highest-priority ready request to a free worker.
        _, r = ready_heap[0]
        pool = pools[r.resource]
        worker_idx, worker_free_at = pool.earliest_free_worker()

        if worker_free_at <= current_time:
            # A worker is free right now — dispatch immediately.
            heapq.heappop(ready_heap)
            r.start_time_ms = current_time
            r.finish_time_ms = current_time + r.service_time_ms
            pool.occupy(worker_idx, current_time, r.service_time_ms)
            completed += 1
        else:
            # No worker free for the head-of-queue request yet. Advance
            # time to the sooner of: this worker freeing up, or the next
            # arrival (which might admit a request for a DIFFERENT,
            # currently-free resource).
            next_arrival = reqs[arrival_ptr].arrival_time_ms if arrival_ptr < n else float("inf")
            current_time = min(worker_free_at, next_arrival)

    # ---- Aggregate metrics ----
    def summarize(subset: List[Request]) -> Dict:
        if not subset:
            return {"count": 0, "avg_wait_ms": 0.0, "avg_response_ms": 0.0, "p95_response_ms": 0.0}
        waits = sorted(r.wait_time_ms for r in subset)
        resp = sorted(r.response_time_ms for r in subset)
        p95_idx = max(0, int(len(resp) * 0.95) - 1)
        return {
            "count": len(subset),
            "avg_wait_ms": sum(waits) / len(waits),
            "avg_response_ms": sum(resp) / len(resp),
            "p95_response_ms": resp[p95_idx],
        }

    urgent_reqs = [r for r in reqs if r.is_urgent]
    non_urgent_reqs = [r for r in reqs if not r.is_urgent]

    sim_end = max((r.finish_time_ms for r in reqs), default=0.0)
    utilization = {}
    for name, pool in pools.items():
        busy = sum(r.service_time_ms for r in reqs if r.resource == name)
        capacity = pool.num_workers * sim_end if sim_end > 0 else 1
        utilization[name] = busy / capacity if capacity > 0 else 0.0

    return {
        "policy": policy.value,
        "overall": summarize(reqs),
        "urgent": summarize(urgent_reqs),
        "non_urgent": summarize(non_urgent_reqs),
        "utilization": utilization,
        "requests": reqs,  # per-request detail, for plotting/debugging
    }


def compare_policies(
    requests: List[Request], fast_workers: int = 1, smart_workers: int = 1
) -> Dict:
    """Runs both PRIORITY and FCFS over the identical request stream and
    returns both results side by side — this is the core experiment."""
    priority_result = run_simulation(requests, SchedulingPolicy.PRIORITY, fast_workers, smart_workers)
    fcfs_result = run_simulation(requests, SchedulingPolicy.FCFS, fast_workers, smart_workers)
    return {"priority": priority_result, "fcfs": fcfs_result}
