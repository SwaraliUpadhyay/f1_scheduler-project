"""
store.py  —  DB role 6

THE interface. The ML lead and the OS lead import from this file and
nothing else in data/. Neither of them ever sees a CQL string, a Redis
key, a connection object, or a cache-miss code path.

    from data.store import RaceStore

    store = RaceStore()
    lap     = store.get_lap("2022_Monaco_Grand_Prix", "VER", 34)
    history = store.get_recent_laps("2022_Monaco_Grand_Prix", "VER", 34)
    store.log_prediction(...)

Cache policy lives here, in one place:
  prefer_cache=True   -> Redis, fall through to Cassandra on miss, then
                         backfill Redis so the next read hits. This is the
                         urgent path the router selects.
  prefer_cache=False  -> Cassandra directly. The non-urgent path; it has
                         latency slack and reading through deliberately
                         keeps cache space for requests that need it.

The fall-through on a miss is what makes the cache an optimisation rather
than a correctness dependency: a cold or evicted Redis never produces a
wrong answer, only a slower one.
"""

import datetime as _dt
import uuid
from typing import Dict, List, Optional

from config import CASSANDRA_KEYSPACE


class RaceStore:
    def __init__(self, cassandra=None, cache=None, run_id: Optional[str] = None,
                 window: int = 8):
        # Imported lazily so that code which only needs the OS simulation
        # does not require the driver packages to be installed.
        if cassandra is None:
            from data.cassandra_client import CassandraClient
            cassandra = CassandraClient(keyspace=CASSANDRA_KEYSPACE)
        if cache is None:
            from data.redis_client import RedisCache
            cache = RedisCache()

        self.db = cassandra
        self.cache = cache
        self.window = window
        self.run_id = run_id or uuid.uuid4().hex[:12]

    # -- ingest -------------------------------------------------------
    def ingest_laps(self, laps: List[Dict], warm_cache: bool = True):
        """Write-through: Cassandra is the durable record, Redis is warmed
        in the same call so the read path is never cold for a lap that has
        already been ingested."""
        self.db.insert_laps_batched(laps)
        if warm_cache:
            self.cache.put_many(laps)

    # -- read path ----------------------------------------------------
    def get_lap(self, race_id: str, driver: str, lap_number: int,
                prefer_cache: bool = True) -> Optional[Dict]:
        if prefer_cache:
            lap = self.cache.get_lap(race_id, driver, lap_number)
            if lap is not None:
                return lap
        lap = self.db.get_lap(race_id, driver, lap_number)
        if lap is not None and prefer_cache:
            self.cache.put_lap(lap)   # backfill: next read is a hit
        return lap

    def get_recent_laps(self, race_id: str, driver: str, lap_number: int,
                        window: Optional[int] = None,
                        prefer_cache: bool = False) -> List[Dict]:
        """Up to `window` laps ending at lap_number, OLDEST FIRST — the
        exact order and shape predict_smart() expects, so the ML lead can
        pass the result straight through with no reshaping."""
        window = window or self.window
        if prefer_cache:
            laps = self.cache.get_recent_laps(race_id, driver, lap_number, window)
            if laps:
                return laps
        laps = self.db.get_recent_laps(race_id, driver, lap_number, window)
        if laps and prefer_cache:
            self.cache.put_many(laps)
        return laps

    # -- prediction log (DB role 2) -----------------------------------
    def log_prediction(self, race_id: str, driver: str, lap_number: int,
                       urgency_score: float, model_used: str, data_source: str,
                       pit=None, pit_prob=None, compound=None,
                       scheduling_policy: str = "", wait_time_ms=None,
                       response_time_ms=None, request_id: Optional[str] = None):
        self.db.log_prediction(
            run_id=self.run_id,
            race_id=race_id,
            request_ts=_dt.datetime.now(_dt.timezone.utc),
            request_id=request_id or uuid.uuid4().hex[:16],
            driver=driver, lap_number=lap_number,
            urgency_score=float(urgency_score),
            model_used=model_used, data_source=data_source,
            scheduling_policy=scheduling_policy,
            pit=int(pit) if pit is not None else None,
            pit_prob=float(pit_prob) if pit_prob is not None else None,
            compound=compound,
            wait_time_ms=wait_time_ms, response_time_ms=response_time_ms,
        )

        # -- DB ENHANCEMENT PASS (additive) ---------------------------
        # Every prediction that already gets logged also updates the
        # live leaderboard (TTL) and the event counters (counter
        # columns), so those NoSQL features are exercised by the real
        # pipeline every time it runs, not just by a standalone demo.
        # Nothing above this comment changed; this call adds two extra
        # writes but does not change this method's signature or return
        # value (still None), so router.py/scheduler.py need no changes.
        self.db.update_leaderboard(race_id, driver, lap_number, float(urgency_score))
        self.db.increment_counter(race_id, f"predictions_{model_used}")
        if pit == 1:
            self.db.increment_counter(race_id, "pit_stops")

    # -- instrumentation ----------------------------------------------
    def stats(self) -> Dict:
        return {
            "run_id": self.run_id,
            "cache": self.cache.stats(),
            "cassandra_latency": self.db.latency_stats(),
        }

    def reset_stats(self):
        self.cache.reset_stats()
        self.db.reset_latencies()

    # ===================================================================
    # DB ENHANCEMENT PASS — new facade methods. The ML/OS leads still
    # never need to touch these; they exist for the DB report + demo.
    # ===================================================================

    # -- UDF/UDA-backed reads --
    def get_lap_enriched(self, race_id: str, driver: str, lap_number: int) -> Optional[Dict]:
        return self.db.get_lap_enriched(race_id, driver, lap_number)

    def avg_urgency_for_race(self, race_id: str) -> Optional[float]:
        return self.db.avg_urgency_for_race(self.run_id, race_id)

    def response_time_bucket_counts(self, race_id: str) -> Dict[str, int]:
        return self.db.response_time_bucket_counts(self.run_id, race_id)

    # -- B1: TTL --
    def get_leaderboard(self, race_id: str) -> List[Dict]:
        return self.db.get_leaderboard(race_id)

    # -- B2: counters --
    def get_counter(self, race_id: str, counter_name: str) -> int:
        return self.db.get_counter(race_id, counter_name)

    # -- B3: collections --
    def add_compound_used(self, race_id: str, driver: str, compound: str):
        self.db.add_compound_used(race_id, driver, compound)

    def append_stint(self, race_id: str, driver: str, stint_label: str):
        self.db.append_stint(race_id, driver, stint_label)

    def get_driver_summary(self, race_id: str, driver: str) -> Optional[Dict]:
        return self.db.get_driver_summary(race_id, driver)

    # -- B4: materialized view --
    def get_predictions_by_driver(self, driver: str, limit: int = 50) -> List[Dict]:
        return self.db.get_predictions_by_driver(driver, limit)

    # -- B5: lightweight transaction --
    def upsert_driver_if_not_exists(self, race_id: str, driver: str, max_lap: int) -> bool:
        return self.db.upsert_driver_if_not_exists(race_id, driver, max_lap)

    # ===================================================================
    # END DB ENHANCEMENT PASS
    # ===================================================================

    def close(self):
        self.db.close()
