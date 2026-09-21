"""
redis_client.py  —  DB role 3

In-memory cache in front of Cassandra, serving the urgent/time-critical
read path.

KEY DESIGN
  f1:lap:{race_id}:{driver}:{lap_number}   -> JSON string, one lap
  f1:win:{race_id}:{driver}                -> sorted set, score = lap_number

The sorted set is the important one. predict_smart() needs "the last 8
laps", and ZREVRANGEBYSCORE gives exactly that in one O(log N + k) round
trip. Storing the window as 8 separate GETs would be 8 round trips, which
on the urgent path is the entire latency budget. Say this in the report:
the cache is not "Redis in front of Cassandra", it is a data structure
chosen to match the query shape.

Every access is counted so cache hit rate is a measured number, not a
claim (DB role 5).
"""

import json
from typing import Dict, List, Optional

import redis

from config import REDIS_HOST, REDIS_PORT, REDIS_TTL_SEC


def _lap_key(race_id: str, driver: str, lap_number: int) -> str:
    return f"f1:lap:{race_id}:{driver}:{lap_number}"


def _window_key(race_id: str, driver: str) -> str:
    return f"f1:win:{race_id}:{driver}"


class RedisCache:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, ttl=REDIS_TTL_SEC):
        self.r = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except redis.ConnectionError:
            return False

    # -- writes -------------------------------------------------------
    def put_lap(self, lap: Dict):
        """Write-through: called whenever a lap is ingested, so the cache
        is warm before the first request for that lap arrives. A cold
        cache on the urgent path defeats the entire point of having one."""
        race_id, driver = lap["race_id"], lap["driver"]
        lap_number = int(lap["lap_number"])
        blob = json.dumps(lap, default=str)

        pipe = self.r.pipeline()
        pipe.setex(_lap_key(race_id, driver, lap_number), self.ttl, blob)
        pipe.zadd(_window_key(race_id, driver), {blob: lap_number})
        pipe.expire(_window_key(race_id, driver), self.ttl)
        # Keep the window bounded — we never need more than the model's
        # window, and unbounded sorted sets are how a cache turns into a
        # memory leak over a full race weekend.
        pipe.zremrangebyrank(_window_key(race_id, driver), 0, -33)
        pipe.execute()

    def put_many(self, laps: List[Dict]):
        pipe = self.r.pipeline()
        for lap in laps:
            race_id, driver = lap["race_id"], lap["driver"]
            lap_number = int(lap["lap_number"])
            blob = json.dumps(lap, default=str)
            pipe.setex(_lap_key(race_id, driver, lap_number), self.ttl, blob)
            pipe.zadd(_window_key(race_id, driver), {blob: lap_number})
            pipe.expire(_window_key(race_id, driver), self.ttl)
        pipe.execute()

    # -- reads --------------------------------------------------------
    def get_lap(self, race_id: str, driver: str, lap_number: int) -> Optional[Dict]:
        blob = self.r.get(_lap_key(race_id, driver, lap_number))
        if blob is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(blob)

    def get_recent_laps(self, race_id: str, driver: str, lap_number: int,
                        window: int = 8) -> Optional[List[Dict]]:
        """Up to `window` laps ending at lap_number, OLDEST FIRST.
        Returns None on a miss so the caller can fall through to Cassandra;
        an empty list would be indistinguishable from "this driver has no
        laps", which is a different thing."""
        blobs = self.r.zrevrangebyscore(
            _window_key(race_id, driver), max=lap_number, min="-inf",
            start=0, num=window,
        )
        if not blobs:
            self.misses += 1
            return None
        self.hits += 1
        return [json.loads(b) for b in reversed(blobs)]

    # -- benchmark support --------------------------------------------
    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }

    def flush(self):
        for key in self.r.scan_iter("f1:*"):
            self.r.delete(key)
