"""
urgency.py

Rule-based urgency classifier. Scores an incoming race situation on a
continuous [0, 1] scale (used as the priority key in the scheduler) and
also exposes a binary is_urgent() for the router's fast/smart decision.

Designed to start as pure rules (explainable, no training data needed,
matches "rule-based to start" from the brief) but built so a learned
model could later replace score_urgency() without touching the router
or scheduler — they only depend on this module's two functions.
"""

from dataclasses import dataclass
from typing import Optional

# ---- Tunable thresholds -----------------------------------------------
# These are starting points based on domain reasoning, not fit to data.
# Worth citing in your report as "expert-defined thresholds" and flagging
# as a place a learned classifier could improve on rules later.
GAP_CRITICAL_SEC = 1.0        # DRS/overtake threat range in F1
GAP_WARNING_SEC = 2.0
TYRE_LIFE_HIGH_LAPS = 20      # laps into a stint where degradation risk rises
FIELD_PIT_PRESSURE_HIGH = 3   # rivals pitting recently -> undercut/overcut pressure

URGENCY_THRESHOLD = 0.5       # >= this -> classified "urgent" for routing


@dataclass
class RaceSituation:
    """The inputs the urgency classifier needs for one driver at one lap.
    Matches the feature dict shape used by predict_service.py so the same
    lap record can feed both the urgency classifier and the ML models."""
    gap_ahead: Optional[float]      # seconds; None if leading
    gap_behind: Optional[float]     # seconds; None if last
    tyre_life: float                # laps on current tyre
    is_caution: float                # 0.0/1.0 — safety car / VSC / red flag active
    rainfall: bool
    laps_since_field_pit: int       # rivals pitting recently


def score_urgency(situation: RaceSituation) -> float:
    """
    Returns a continuous urgency score in [0, 1]. Higher = more urgent.
    Used directly as the priority key in the scheduler (finer-grained than
    a binary label, so ties are rare and ordering is meaningful).
    """
    score = 0.0

    # Rival about to overtake, or about to be overtaken — the single
    # clearest "decide now" signal in racing.
    if situation.gap_behind is not None:
        if situation.gap_behind < GAP_CRITICAL_SEC:
            score += 0.35
        elif situation.gap_behind < GAP_WARNING_SEC:
            score += 0.15

    # Undercut opportunity ahead — slightly less urgent than a defensive
    # threat from behind, so weighted lower.
    if situation.gap_ahead is not None:
        if situation.gap_ahead < GAP_CRITICAL_SEC:
            score += 0.15
        elif situation.gap_ahead < GAP_WARNING_SEC:
            score += 0.05

    # Safety car / VSC / red flag — pit windows compress hard under
    # caution, decisions become extremely time-sensitive.
    if situation.is_caution:
        score += 0.30

    # Rain changes tyre choice instantly and unpredictably.
    if situation.rainfall:
        score += 0.20

    # Tyres past a degradation threshold — pit decision becomes pressing.
    if situation.tyre_life >= TYRE_LIFE_HIGH_LAPS:
        score += 0.15

    # Rivals pitting recently — undercut/overcut pressure builds.
    if situation.laps_since_field_pit >= FIELD_PIT_PRESSURE_HIGH:
        score += 0.10

    return min(score, 1.0)


def is_urgent(situation: RaceSituation, threshold: float = URGENCY_THRESHOLD) -> bool:
    return score_urgency(situation) >= threshold


def situation_from_lap_dict(lap: dict) -> RaceSituation:
    """
    Convenience adapter: builds a RaceSituation from the same lap dict
    shape predict_service.py's predict_fast()/predict_smart() expect, so
    one lap record can drive urgency scoring AND model inference without
    reshaping data twice.
    """
    return RaceSituation(
        gap_ahead=lap.get("gap_ahead"),
        gap_behind=lap.get("gap_behind"),
        tyre_life=lap.get("tyre_life", 0.0),
        is_caution=lap.get("is_caution", 0.0),
        rainfall=bool(lap.get("rainfall", False)),
        laps_since_field_pit=lap.get("laps_since_field_pit", 0),
    )
