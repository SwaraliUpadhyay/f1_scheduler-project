"""
router.py

Given a scored request, decides:
  - which MODEL answers it (fast/local vs smart/cloud)
  - which DATA SOURCE it reads from (Redis cache vs Cassandra)

This is the resource allocation decision — kept separate from the
scheduler's ORDERING decision (see scheduler.py), per the project brief:
"resource allocation... is a distinct OS concept from scheduling (order)
— both need to be addressed and evaluated separately."

Depends only on urgency.py and predict_service.py's callable interface —
does not know about FastModel/SmartModel internals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from urgency import RaceSituation, score_urgency, is_urgent, URGENCY_THRESHOLD


class ModelChoice(Enum):
    FAST_LOCAL = "fast_local"
    SMART_CLOUD = "smart_cloud"


class DataSource(Enum):
    REDIS_CACHE = "redis_cache"
    CASSANDRA = "cassandra"


@dataclass
class RoutingDecision:
    model: ModelChoice
    data_source: DataSource
    urgency_score: float


def route(situation: RaceSituation, threshold: float = URGENCY_THRESHOLD) -> RoutingDecision:
    """
    Pure decision logic — no I/O, no model calls. Kept separate from
    execution (see Router.handle below) so this function alone can be
    unit tested and is the citable "routing algorithm" for the paper/patent.
    """
    score = score_urgency(situation)
    if score >= threshold:
        # Urgent: minimize latency. Fast local model + in-memory cache,
        # which is presumably already warm for the current race state.
        return RoutingDecision(ModelChoice.FAST_LOCAL, DataSource.REDIS_CACHE, score)
    else:
        # Not urgent: spend the extra latency budget on the more accurate
        # model, reading from the durable telemetry store.
        return RoutingDecision(ModelChoice.SMART_CLOUD, DataSource.CASSANDRA, score)


class Router:
    """
    Wraps the pure routing decision with actual execution — calling the
    ML lead's predict_fast/predict_smart, and (placeholder for now) a data
    source lookup the DB/Cloud lead will wire up to real Redis/Cassandra
    clients. Model functions are injected so this class is testable
    without needing trained model files or a live database.
    """

    def __init__(
        self,
        predict_fast: Callable[[Dict], Dict],
        predict_smart: Callable[[List[Dict]], Dict],
        data_source_lookup: Optional[Callable[[DataSource, str], Dict]] = None,
        threshold: float = URGENCY_THRESHOLD,
    ):
        self.predict_fast = predict_fast
        self.predict_smart = predict_smart
        # DB/Cloud lead plugs their real Redis/Cassandra client in here.
        # Defaults to a no-op so this module runs standalone until then.
        self.data_source_lookup = data_source_lookup or (lambda source, key: {})
        self.threshold = threshold

    def handle(self, situation: RaceSituation, lap: Dict, history: Optional[List[Dict]] = None) -> Dict:
        """
        lap: the current lap dict (same shape predict_service.py expects).
        history: last-N-lap dicts for the smart model; only needed if
        the decision routes to SMART_CLOUD. Falls back to [lap] if omitted,
        which predict_smart auto-pads — degraded but functional.
        """
        decision = route(situation, self.threshold)

        _ = self.data_source_lookup(decision.data_source, lap.get("driver", ""))

        if decision.model == ModelChoice.FAST_LOCAL:
            result = self.predict_fast(lap)
        else:
            result = self.predict_smart(history if history else [lap])

        return {
            "prediction": result,
            "model_used": decision.model.value,
            "data_source_used": decision.data_source.value,
            "urgency_score": decision.urgency_score,
        }
