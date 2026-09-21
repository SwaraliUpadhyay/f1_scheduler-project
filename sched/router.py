"""
router.py  —  OS roles 2 and 4

Given a scored situation, decides two things:
  - which MODEL answers it        (fast/local vs smart/cloud)
  - which DATA SOURCE it reads    (Redis cache vs Cassandra)

This is RESOURCE ALLOCATION. It is deliberately kept separate from the
scheduler's ORDERING decision, because they are distinct OS concepts and
the evaluation reports them separately: the scheduler decides *when* a
request runs, the router decides *where* it runs.

DESIGN NOTE FOR THE REPORT — why urgent goes to the FAST model
This looks backwards at first ("the important request gets the worse
model"). The justification is a deadline argument, not a quality one: an
overtake or a caution-period pit call has a decision window of roughly
one to two seconds. A prediction that arrives after the window closes has
zero value regardless of its accuracy, so under deadline pressure the
correct allocation maximises probability-of-answering-in-time, not
expected accuracy. Non-urgent requests have slack, so they spend it on
the more accurate model. State this explicitly — a marker will ask.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from config import (
    URGENCY_THRESHOLD, FAST_SERVICE_MS, SMART_COMPUTE_MS,
    SMART_NETWORK_MS, REDIS_LOOKUP_MS, CASSANDRA_LOOKUP_MS,
)
from sched.urgency import RaceSituation, score_urgency


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

    @property
    def service_time_ms(self) -> float:
        """Total cost of servicing this request on the allocated resources:
        model inference + the data-source read it needs. The scheduler
        consumes this as the request's service time, so the allocation
        decision feeds the scheduling experiment directly."""
        if self.model == ModelChoice.FAST_LOCAL:
            compute = FAST_SERVICE_MS
        else:
            compute = SMART_COMPUTE_MS + SMART_NETWORK_MS
        io = (REDIS_LOOKUP_MS if self.data_source == DataSource.REDIS_CACHE
              else CASSANDRA_LOOKUP_MS)
        return compute + io


def route(situation: RaceSituation, threshold: float = URGENCY_THRESHOLD) -> RoutingDecision:
    """Pure decision logic — no I/O, no model calls. Kept callable on its
    own so it can be unit tested and cited as *the* routing algorithm."""
    score = score_urgency(situation)
    if score >= threshold:
        # Urgent: minimise latency. Local model, in-memory cache that is
        # already warm for the current race state.
        return RoutingDecision(ModelChoice.FAST_LOCAL, DataSource.REDIS_CACHE, score)
    # Not urgent: spend the latency budget on accuracy, read from the
    # durable telemetry store.
    return RoutingDecision(ModelChoice.SMART_CLOUD, DataSource.CASSANDRA, score)


class Router:
    """Wraps the pure decision with execution: calls the ML lead's
    predict_fast/predict_smart and the DB layer's store facade.

    Everything is injected, so this class runs in tests with no trained
    model files and no live database.
    """

    def __init__(
        self,
        predict_fast: Callable[[Dict], Dict],
        predict_smart: Callable[[List[Dict]], Dict],
        store=None,
        threshold: float = URGENCY_THRESHOLD,
    ):
        self.predict_fast = predict_fast
        self.predict_smart = predict_smart
        self.store = store          # data.store.RaceStore, or None to skip I/O
        self.threshold = threshold

    def handle(self, situation: RaceSituation, race_id: str, driver: str,
               lap_number: int, lap: Optional[Dict] = None) -> Dict:
        """Route and execute one request end to end.

        If a store is attached, the lap record and history are READ FROM IT
        via the source the router chose — that is what makes the Redis vs
        Cassandra allocation real rather than decorative. `lap` is only a
        fallback for running without a database.
        """
        decision = route(situation, self.threshold)

        if self.store is not None:
            if decision.model == ModelChoice.FAST_LOCAL:
                lap = self.store.get_lap(race_id, driver, lap_number,
                                         prefer_cache=True)
                history = None
            else:
                history = self.store.get_recent_laps(race_id, driver, lap_number,
                                                     prefer_cache=False)
                lap = history[-1] if history else lap
        else:
            history = [lap] if lap else None

        if lap is None:
            raise ValueError(f"no lap record for {race_id}/{driver}/{lap_number}")

        if decision.model == ModelChoice.FAST_LOCAL:
            result = self.predict_fast(lap)
        else:
            result = self.predict_smart(history if history else [lap])

        out = {
            "prediction": result,
            "model_used": decision.model.value,
            "data_source_used": decision.data_source.value,
            "urgency_score": decision.urgency_score,
            "service_time_ms": decision.service_time_ms,
        }

        if self.store is not None:
            # Shared evaluation data for the whole team (DB role 2).
            self.store.log_prediction(
                race_id=race_id, driver=driver, lap_number=lap_number,
                urgency_score=decision.urgency_score,
                model_used=decision.model.value,
                data_source=decision.data_source.value,
                pit=result.get("pit"), pit_prob=result.get("pit_prob"),
                compound=result.get("compound"),
            )
        return out
