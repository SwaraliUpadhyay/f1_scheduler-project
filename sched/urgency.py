"""
urgency.py  —  OS role 1

Rule-based urgency classifier. Scores a race situation on a continuous
[0, 1] scale (the priority key for the scheduler) and exposes a binary
is_urgent() for the router's fast/smart decision.

Pure rules by design: explainable, needs no training data, and every
weight below is a defensible domain claim you can cite in the report.
A learned classifier can later replace score_urgency() without touching
router.py or scheduler.py — they depend only on this module's functions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import (
    GAP_CRITICAL_SEC, GAP_WARNING_SEC, TYRE_LIFE_HIGH_LAPS,
    FIELD_PIT_PRESSURE_HIGH, URGENCY_THRESHOLD,
)


@dataclass
class RaceSituation:
    """Inputs for one driver at one lap. Field names match the lap dict
    that predict_service.py consumes, so one record drives both urgency
    scoring and model inference without reshaping twice."""
    gap_ahead: Optional[float]        # seconds; None if leading
    gap_behind: Optional[float]       # seconds; None if last
    tyre_life: float                  # laps on the current tyre
    is_caution: float                 # 0.0/1.0 — safety car / VSC / red flag
    rainfall: bool
    field_pits_last_5: int            # rivals that pitted in the last 5 laps


def score_urgency(situation: RaceSituation) -> float:
    """Continuous urgency in [0, 1]. Higher = more urgent. Used directly
    as the scheduler's priority key — finer-grained than a binary label,
    so ordering is meaningful and ties are rare."""
    score = 0.0

    # Defensive threat: a rival inside overtake range is the clearest
    # "decide now" signal in racing.
    if situation.gap_behind is not None:
        if situation.gap_behind < GAP_CRITICAL_SEC:
            score += 0.35
        elif situation.gap_behind < GAP_WARNING_SEC:
            score += 0.15

    # Offensive/undercut opportunity — real, but less forcing than being
    # attacked, so weighted lower.
    if situation.gap_ahead is not None:
        if situation.gap_ahead < GAP_CRITICAL_SEC:
            score += 0.15
        elif situation.gap_ahead < GAP_WARNING_SEC:
            score += 0.05

    # Caution periods compress the pit window hard — decisions become
    # extremely time-sensitive and this is a top real-world pit trigger.
    if situation.is_caution:
        score += 0.30

    # Rain flips tyre choice instantly.
    if situation.rainfall:
        score += 0.20

    # Degradation past threshold makes the stop pressing.
    if situation.tyre_life >= TYRE_LIFE_HIGH_LAPS:
        score += 0.15

    # Rivals stopping builds undercut/overcut pressure.
    if situation.field_pits_last_5 >= FIELD_PIT_PRESSURE_HIGH:
        score += 0.10

    return min(score, 1.0)


def is_urgent(situation: RaceSituation, threshold: float = URGENCY_THRESHOLD) -> bool:
    return score_urgency(situation) >= threshold


def explain(situation: RaceSituation) -> List[Tuple[str, float]]:
    """Per-rule contribution breakdown. Used for the report's worked
    examples and for debugging why a lap scored the way it did."""
    parts = []
    if situation.gap_behind is not None:
        if situation.gap_behind < GAP_CRITICAL_SEC:
            parts.append(("under_attack", 0.35))
        elif situation.gap_behind < GAP_WARNING_SEC:
            parts.append(("pressure_behind", 0.15))
    if situation.gap_ahead is not None:
        if situation.gap_ahead < GAP_CRITICAL_SEC:
            parts.append(("undercut_window", 0.15))
        elif situation.gap_ahead < GAP_WARNING_SEC:
            parts.append(("closing_ahead", 0.05))
    if situation.is_caution:
        parts.append(("caution", 0.30))
    if situation.rainfall:
        parts.append(("rain", 0.20))
    if situation.tyre_life >= TYRE_LIFE_HIGH_LAPS:
        parts.append(("tyre_degradation", 0.15))
    if situation.field_pits_last_5 >= FIELD_PIT_PRESSURE_HIGH:
        parts.append(("field_pit_pressure", 0.10))
    return parts


def situation_from_lap_dict(lap: dict) -> RaceSituation:
    """Adapter from the lap dict shape used everywhere else in the system."""
    return RaceSituation(
        gap_ahead=lap.get("gap_ahead"),
        gap_behind=lap.get("gap_behind"),
        tyre_life=float(lap.get("tyre_life", 0.0) or 0.0),
        is_caution=float(lap.get("is_caution", 0.0) or 0.0),
        rainfall=bool(lap.get("rainfall", False)),
        field_pits_last_5=int(lap.get("field_pits_last_5", 0) or 0),
    )


def caution_from_track_status(track_status) -> float:
    """FastF1 gives a per-lap flag code string: "1" green, "2" yellow,
    "4" safety car, "5" red, "6"/"7" VSC. A lap can concatenate codes
    (e.g. "24") when conditions changed mid-lap. Collapse to a single
    binary "anything other than green" signal — one-hot over rare codes
    is mostly sparse noise.

    This is the SAME derivation models.prepare_features() uses at
    training time; keeping it in one importable place prevents the
    train/inference mismatch where is_caution silently becomes 0.
    """
    s = str(track_status) if track_status is not None else "1"
    return 0.0 if set(s) <= {"1"} else 1.0
