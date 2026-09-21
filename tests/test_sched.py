"""
Unit tests for the OS workstream. Run with:  python -m pytest tests/ -v
(or `python tests/test_sched.py` if pytest isn't installed)

These exist because the scheduling results are the project's evidence.
An undetected scheduler bug does not crash — it quietly produces numbers
that look plausible and are wrong. That is the failure mode to guard.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import URGENCY_THRESHOLD
from sched.router import DataSource, ModelChoice, route
from sched.scheduler import (
    Request, SchedulingPolicy, compare_policies, run_simulation,
)
from sched.urgency import RaceSituation, caution_from_track_status, score_urgency


def _sit(**kw):
    base = dict(gap_ahead=5.0, gap_behind=5.0, tyre_life=5.0,
                is_caution=0.0, rainfall=False, field_pits_last_5=0)
    base.update(kw)
    return RaceSituation(**base)


# --- urgency ---------------------------------------------------------
def test_calm_lap_is_not_urgent():
    assert score_urgency(_sit()) == 0.0


def test_caution_plus_attack_is_urgent():
    s = _sit(gap_behind=0.4, is_caution=1.0)
    assert score_urgency(s) >= URGENCY_THRESHOLD


def test_score_is_bounded():
    s = _sit(gap_ahead=0.1, gap_behind=0.1, tyre_life=40,
             is_caution=1.0, rainfall=True, field_pits_last_5=9)
    assert score_urgency(s) == 1.0


def test_leading_car_has_no_gap_ahead_penalty():
    # None must mean "no car there", not "gap of zero".
    assert score_urgency(_sit(gap_ahead=None)) == score_urgency(_sit(gap_ahead=99.0))


def test_caution_derivation_matches_training():
    assert caution_from_track_status("1") == 0.0
    assert caution_from_track_status("11") == 0.0
    assert caution_from_track_status("4") == 1.0
    assert caution_from_track_status("24") == 1.0   # multi-code lap
    assert caution_from_track_status(None) == 0.0


# --- router ----------------------------------------------------------
def test_urgent_routes_local_and_cached():
    d = route(_sit(gap_behind=0.4, is_caution=1.0))
    assert d.model is ModelChoice.FAST_LOCAL
    assert d.data_source is DataSource.REDIS_CACHE


def test_non_urgent_routes_cloud_and_durable():
    d = route(_sit())
    assert d.model is ModelChoice.SMART_CLOUD
    assert d.data_source is DataSource.CASSANDRA


def test_service_time_includes_data_source_cost():
    urgent = route(_sit(gap_behind=0.4, is_caution=1.0))
    calm = route(_sit())
    assert urgent.service_time_ms < calm.service_time_ms


# --- scheduler -------------------------------------------------------
def test_no_cross_resource_head_of_line_blocking():
    """The bug that silently corrupts every response-time number: a
    request queued behind a busy resource must not block a request
    targeting an IDLE resource."""
    reqs = [
        Request("a", 0.0, 0.9, "smart_cloud", 100.0, is_urgent=True),
        Request("b", 1.0, 0.8, "smart_cloud", 100.0, is_urgent=True),
        Request("c", 2.0, 0.1, "fast_local", 5.0),
    ]
    out = run_simulation(reqs, SchedulingPolicy.PRIORITY)
    c = next(r for r in out["requests"] if r.id == "c")
    assert c.start_time_ms == 2.0, "fast_local was idle; c should not have waited"


def test_priority_orders_by_urgency_within_a_resource():
    reqs = [
        Request("blocker", 0.0, 0.5, "fast_local", 10.0),
        Request("low", 1.0, 0.10, "fast_local", 1.0),
        Request("high", 2.0, 0.95, "fast_local", 1.0, is_urgent=True),
    ]
    out = run_simulation(reqs, SchedulingPolicy.PRIORITY)
    order = {r.id: r.start_time_ms for r in out["requests"]}
    assert order["high"] < order["low"]


def test_fcfs_preserves_arrival_order():
    reqs = [
        Request("blocker", 0.0, 0.5, "fast_local", 10.0),
        Request("low", 1.0, 0.10, "fast_local", 1.0),
        Request("high", 2.0, 0.95, "fast_local", 1.0, is_urgent=True),
    ]
    out = run_simulation(reqs, SchedulingPolicy.FCFS)
    order = {r.id: r.start_time_ms for r in out["requests"]}
    assert order["low"] < order["high"]


def test_every_request_is_served_exactly_once():
    reqs = [Request(f"r{i}", float(i % 3), (i % 10) / 10,
                    "fast_local" if i % 2 else "smart_cloud", 2.0)
            for i in range(200)]
    for policy in (SchedulingPolicy.PRIORITY, SchedulingPolicy.FCFS):
        out = run_simulation(reqs, policy)
        assert len(out["requests"]) == 200
        assert all(r.start_time_ms is not None for r in out["requests"])
        assert all(r.wait_time_ms >= -1e-9 for r in out["requests"])


def test_both_policies_do_identical_total_work():
    """Policy changes ORDER, never the amount of work. If total service
    time differs between policies, the simulation is losing or
    duplicating requests."""
    reqs = [Request(f"r{i}", float(i), (i % 7) / 10,
                    "fast_local" if i % 3 else "smart_cloud", 1.5)
            for i in range(100)]
    out = compare_policies(reqs)
    work = {p: sum(r.service_time_ms for r in out[p]["requests"])
            for p in ("priority", "fcfs")}
    assert abs(work["priority"] - work["fcfs"]) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
