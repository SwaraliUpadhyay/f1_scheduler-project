"""
replay.py  —  OS role 5

Replays historical races lap-by-lap as a simulated live feed and emits a
timed stream of prediction requests.

HOW THE ARRIVAL PROCESS IS BUILT
Each lap of a race is one clock tick (LAP_DURATION_MS). Every driver still
running on that lap issues one strategy request. What varies is *when*
within the tick:

  - Calm lap      : requests are spread over NORMAL_JITTER_MS. The pit
                    wall is polling; nothing is contended.
  - High-pressure : every driver's request fires within BURST_SPREAD_MS
                    of the same instant. This is the contention event the
                    whole experiment exists to measure — several cars
                    needing a decision at once.

A lap counts as high-pressure if any of these hold for any driver:
  * caution period active (safety car / VSC / red flag)
  * rainfall
  * a gap ahead or behind inside BURST_GAP_SEC (overtake in progress)
  * tyre life past BURST_TYRE_LIFE (pit window open)

These are exactly the moments the brief names — overtakes, pit windows,
weather changes — and they are read from the real data, not injected, so
the burst pattern is a property of the race rather than of the harness.

Both policies are then run over this ONE stream, so any difference in the
results is attributable to the scheduling policy alone.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    DATASET_PATH, LAP_DURATION_MS, NORMAL_JITTER_MS, BURST_SPREAD_MS,
    BURST_TYRE_LIFE, BURST_GAP_SEC, URGENCY_THRESHOLD, ROUTING_MODE,
    REFINEMENT_MAX_URGENCY, FAST_SERVICE_MS, REDIS_LOOKUP_MS,
    SMART_COMPUTE_MS, SMART_NETWORK_MS, CASSANDRA_LOOKUP_MS,
)
from sched.router import route
from sched.scheduler import Request
from sched.urgency import (
    RaceSituation, caution_from_track_status, is_urgent, score_urgency,
)


def load_races(path: str = DATASET_PATH, race_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """Load the dataset and derive the two columns the OS layer needs but
    the raw table does not carry under those names."""
    df = pd.read_parquet(path)

    # Derived once, here, so urgency scoring and model inference agree.
    df["is_caution"] = df["track_status"].apply(caution_from_track_status)

    # The pipeline column named `laps_since_field_pit` is actually a
    # rolling COUNT of pit stops across the field in the last 5 laps.
    # Rename on load rather than downstream, so the semantics are honest
    # everywhere the OS layer touches it.
    df = df.rename(columns={"laps_since_field_pit": "field_pits_last_5"})
    df["field_pits_last_5"] = df["field_pits_last_5"].fillna(0).astype(int)

    if race_ids:
        df = df[df["race_id"].isin(race_ids)]
    return df.sort_values(["race_id", "lap_number", "driver"]).reset_index(drop=True)


def lap_to_dict(row) -> Dict:
    """One dataframe row -> the lap dict every other module consumes."""
    def _num(v, default=0.0):
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    return {
        "race_id": row["race_id"],
        "driver": row["driver"],
        "lap_number": int(row["lap_number"]),
        "tyre_life": _num(row["tyre_life"]),
        "compound": row["compound"] if isinstance(row["compound"], str) else "MEDIUM",
        # gaps stay None when absent — leading/last car genuinely has no gap,
        # and the urgency rules treat None as "no threat" rather than "zero".
        "gap_ahead": None if pd.isna(row["gap_ahead"]) else float(row["gap_ahead"]),
        "gap_behind": None if pd.isna(row["gap_behind"]) else float(row["gap_behind"]),
        "gap_ahead_delta": _num(row["gap_ahead_delta"]),
        "air_temp": _num(row["air_temp"], 25.0),
        "track_temp": _num(row["track_temp"], 35.0),
        "rainfall": bool(row["rainfall"]),
        "is_caution": _num(row["is_caution"]),
        "field_pits_last_5": int(row["field_pits_last_5"]),
    }


def _burst_reason(lap_rows: pd.DataFrame) -> str:
    """Why this lap is high-pressure, or "" if it is calm. Returned as a
    label so results can be broken down by trigger type in the report."""
    reasons = []
    if (lap_rows["is_caution"] > 0).any():
        reasons.append("caution")
    if lap_rows["rainfall"].any():
        reasons.append("rain")
    close = (
        (lap_rows["gap_behind"] < BURST_GAP_SEC) | (lap_rows["gap_ahead"] < BURST_GAP_SEC)
    ).fillna(False)
    if close.any():
        reasons.append("overtake")
    if (lap_rows["tyre_life"] >= BURST_TYRE_LIFE).any():
        reasons.append("pit_window")
    return "+".join(reasons)


def build_request_stream(
    df: pd.DataFrame,
    seed: int = 42,
    threshold: float = URGENCY_THRESHOLD,
    mode: str = None,
) -> List[Request]:
    """Turn a replayed race (or several) into the timed request stream that
    both scheduling policies will be run over.

    mode="partitioned": one request per driver-lap, resource chosen by
        urgency. Queues end up urgency-homogeneous — priority scheduling
        has nothing to reorder. This is the original brief and it is the
        negative result worth reporting.

    mode="two_stage": every driver-lap emits a fast-path request on the
        local tier (so that tier carries the full urgency range), and
        laps with latency slack emit a second, asynchronous cloud
        refinement request. This is the configuration in which priority
        scheduling is actually measurable.
    """
    mode = mode or ROUTING_MODE
    if mode not in ("partitioned", "two_stage"):
        raise ValueError(f"unknown routing mode: {mode}")
    rng = np.random.default_rng(seed)
    requests: List[Request] = []
    race_offset_ms = 0.0

    for race_id, race_df in df.groupby("race_id", sort=True):
        max_lap = int(race_df["lap_number"].max())

        for lap_number, lap_rows in race_df.groupby("lap_number", sort=True):
            tick_ms = race_offset_ms + (lap_number - 1) * LAP_DURATION_MS
            reason = _burst_reason(lap_rows)
            spread = BURST_SPREAD_MS if reason else NORMAL_JITTER_MS

            # Offsets are drawn, then sorted, so arrival order inside the
            # tick is well defined and FCFS has a meaningful baseline.
            offsets = np.sort(rng.uniform(0.0, spread, size=len(lap_rows)))

            for offset, (_, row) in zip(offsets, lap_rows.iterrows()):
                lap = lap_to_dict(row)
                situation = RaceSituation(
                    gap_ahead=lap["gap_ahead"],
                    gap_behind=lap["gap_behind"],
                    tyre_life=lap["tyre_life"],
                    is_caution=lap["is_caution"],
                    rainfall=lap["rainfall"],
                    field_pits_last_5=lap["field_pits_last_5"],
                )
                decision = route(situation, threshold)
                arrival = tick_ms + float(offset)
                base_id = f"{race_id}|{lap['driver']}|{int(lap_number)}"
                common = dict(
                    urgency_score=decision.urgency_score,
                    is_urgent=decision.urgency_score >= URGENCY_THRESHOLD,
                    race_id=race_id, driver=lap["driver"],
                    lap_number=int(lap_number), burst_reason=reason,
                )

                if mode == "partitioned":
                    requests.append(Request(
                        id=base_id, arrival_time_ms=arrival,
                        resource=decision.model.value,
                        service_time_ms=decision.service_time_ms,
                        data_source=decision.data_source.value, **common))
                else:
                    # Stage 1: the guaranteed-latency answer. Every lap
                    # gets one, whatever its urgency — this is what puts
                    # mixed-priority work on a single contended resource.
                    requests.append(Request(
                        id=base_id + "|fast", arrival_time_ms=arrival,
                        resource="fast_local",
                        service_time_ms=FAST_SERVICE_MS + REDIS_LOOKUP_MS,
                        data_source="redis_cache", **common))
                    # Stage 2: refinement, only where the decision window
                    # is wide enough for a cloud round trip to still land
                    # inside it. Fired at the same instant; it competes
                    # for the cloud tier, not the local one.
                    if decision.urgency_score <= REFINEMENT_MAX_URGENCY:
                        requests.append(Request(
                            id=base_id + "|smart", arrival_time_ms=arrival,
                            resource="smart_cloud",
                            service_time_ms=(SMART_COMPUTE_MS + SMART_NETWORK_MS
                                             + CASSANDRA_LOOKUP_MS),
                            data_source="cassandra", **common))

        race_offset_ms += max_lap * LAP_DURATION_MS

    requests.sort(key=lambda r: r.arrival_time_ms)
    return requests


def stream_summary(requests: List[Request]) -> Dict:
    """Descriptive stats about the generated workload — goes in the report's
    methodology section so the arrival process is auditable."""
    n = len(requests)
    urgent = sum(1 for r in requests if r.is_urgent)
    bursty = sum(1 for r in requests if r.burst_reason)
    by_resource: Dict[str, int] = {}
    by_reason: Dict[str, int] = {}
    for r in requests:
        by_resource[r.resource] = by_resource.get(r.resource, 0) + 1
        if r.burst_reason:
            by_reason[r.burst_reason] = by_reason.get(r.burst_reason, 0) + 1
    return {
        "total_requests": n,
        "urgent_requests": urgent,
        "urgent_fraction": urgent / n if n else 0.0,
        "burst_requests": bursty,
        "burst_fraction": bursty / n if n else 0.0,
        "by_resource": by_resource,
        "by_burst_reason": dict(sorted(by_reason.items(), key=lambda kv: -kv[1])),
        "offered_load_ms": sum(r.service_time_ms for r in requests),
    }
