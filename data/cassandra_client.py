"""
cassandra_client.py  —  DB roles 1, 2

Thin typed wrapper over the Cassandra driver. Everything the rest of the
system needs is a prepared statement executed here; nobody outside this
module writes CQL.

Prepared statements matter for the benchmark, not just for tidiness: an
unprepared statement is parsed server-side on every call and is routed to
a coordinator that may not own the partition. A prepared statement is
parsed once and lets the driver's token-aware policy send the request
straight to a replica, removing a network hop. Benchmarking with
unprepared statements would understate Cassandra badly.
"""

import datetime as _dt
import time
from typing import Dict, List, Optional

from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy, TokenAwarePolicy
from cassandra.query import BatchStatement, ConsistencyLevel

from config import CASSANDRA_HOSTS, CASSANDRA_KEYSPACE, CASSANDRA_PORT
from data.columns import LAP_COLUMNS


class CassandraClient:
    def __init__(self, hosts=None, port=None, keyspace=CASSANDRA_KEYSPACE):
        self.cluster = Cluster(
            hosts or CASSANDRA_HOSTS,
            port=port or CASSANDRA_PORT,
            # Token-aware routing: send each query to a node that actually
            # owns the partition, skipping the coordinator hop.
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
        )
        self.session = self.cluster.connect(keyspace)
        self.latencies_ms: List[float] = []
        self._prepare()
        self._prepare_enhancements()

    def _prepare(self):
        cols = ", ".join(LAP_COLUMNS)
        placeholders = ", ".join(["?"] * len(LAP_COLUMNS))
        self.ps_insert_lap = self.session.prepare(
            f"INSERT INTO telemetry_laps ({cols}) VALUES ({placeholders})")
        self.ps_get_lap = self.session.prepare(
            "SELECT * FROM telemetry_laps "
            "WHERE race_id=? AND driver=? AND lap_number=?")
        # lap_number clusters DESC, so "<= N LIMIT k" is a head read of the
        # partition: k contiguous rows, no sort, no scan.
        self.ps_get_window = self.session.prepare(
            "SELECT * FROM telemetry_laps "
            "WHERE race_id=? AND driver=? AND lap_number<=? LIMIT ?")
        self.ps_insert_pred = self.session.prepare(
            "INSERT INTO predictions_log (run_id, race_id, request_ts, request_id, "
            "driver, lap_number, urgency_score, model_used, data_source, "
            "scheduling_policy, pit, pit_prob, compound, wait_time_ms, "
            "response_time_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        self.ps_upsert_driver = self.session.prepare(
            "INSERT INTO race_drivers (race_id, driver, max_lap) VALUES (?,?,?)")

    # ===================================================================
    # DB ENHANCEMENT PASS — everything below is new. Nothing above this
    # line was changed. Requires data/schema_extensions.cql to have been
    # run first.
    # ===================================================================
    def _prepare_enhancements(self):
        # -- UDF/UDA-backed reads --
        self.ps_lap_enriched = self.session.prepare(
            "SELECT race_id, driver, lap_number, gap_ahead, gap_behind, tyre_life, "
            "urgency_score_udf(gap_ahead, gap_behind, tyre_life, is_caution, "
            "rainfall, field_pits_last_5) AS computed_urgency, "
            "pit_window_flag(tyre_life) AS in_pit_window, "
            "gap_threat_level(gap_ahead, gap_behind) AS threat_level "
            "FROM telemetry_laps WHERE race_id=? AND driver=? AND lap_number=?")
        self.ps_avg_urgency_by_race = self.session.prepare(
            "SELECT avg_urgency_by_race(urgency_score) AS avg_urgency "
            "FROM predictions_log WHERE run_id=? AND race_id=?")
        self.ps_response_time_buckets = self.session.prepare(
            "SELECT response_time_bucket(response_time_ms) AS bucket "
            "FROM predictions_log WHERE run_id=? AND race_id=?")

        # -- B1: TTL --
        self.ps_leaderboard_upsert = self.session.prepare(
            "INSERT INTO live_leaderboard (race_id, driver, lap_number, "
            "urgency, updated_at) VALUES (?, ?, ?, ?, ?) USING TTL ?")
        self.ps_leaderboard_get = self.session.prepare(
            "SELECT * FROM live_leaderboard WHERE race_id=?")

        # -- B2: counters --
        self.ps_counter_incr = self.session.prepare(
            "UPDATE race_event_counters SET count = count + ? "
            "WHERE race_id=? AND counter_name=?")
        self.ps_counter_get = self.session.prepare(
            "SELECT count FROM race_event_counters "
            "WHERE race_id=? AND counter_name=?")

        # -- B3: collections --
        self.ps_add_compound = self.session.prepare(
            "UPDATE driver_race_summary SET compounds_used = compounds_used + ? "
            "WHERE race_id=? AND driver=?")
        self.ps_append_stint = self.session.prepare(
            "UPDATE driver_race_summary SET stint_history = stint_history + ? "
            "WHERE race_id=? AND driver=?")
        self.ps_get_driver_summary = self.session.prepare(
            "SELECT * FROM driver_race_summary WHERE race_id=? AND driver=?")

        # -- B4: materialized view (read-only; Cassandra writes it for us) --
        self.ps_predictions_by_driver = self.session.prepare(
            "SELECT * FROM predictions_by_driver WHERE driver=? LIMIT ?")

        # -- B5: lightweight transaction (compare-and-set) --
        self.ps_upsert_driver_lwt = self.session.prepare(
            "INSERT INTO race_drivers (race_id, driver, max_lap) "
            "VALUES (?, ?, ?) IF NOT EXISTS")

    # -- UDF/UDA-backed reads ------------------------------------------
    def get_lap_enriched(self, race_id: str, driver: str, lap_number: int) -> Optional[Dict]:
        """Same point read as get_lap(), but urgency score, pit-window
        flag, and threat level are computed INSIDE Cassandra by the UDFs
        in schema_extensions.cql, not in Python."""
        rows = self._timed(self.ps_lap_enriched, [race_id, driver, lap_number])
        row = rows.one()
        return dict(row._asdict()) if row else None

    def avg_urgency_for_race(self, run_id: str, race_id: str) -> Optional[float]:
        """User-Defined Aggregate — averages urgency_score server-side
        across a whole partition in one round trip."""
        rows = self._timed(self.ps_avg_urgency_by_race, [run_id, race_id])
        row = rows.one()
        return row.avg_urgency if row else None

    def response_time_bucket_counts(self, run_id: str, race_id: str) -> Dict[str, int]:
        rows = self._timed(self.ps_response_time_buckets, [run_id, race_id])
        counts = {"fast": 0, "medium": 0, "slow": 0, "unknown": 0}
        for r in rows:
            counts[r.bucket] = counts.get(r.bucket, 0) + 1
        return counts

    # -- B1: TTL ---------------------------------------------------------
    def update_leaderboard(self, race_id: str, driver: str, lap_number: int,
                           urgency: float, ttl_sec: int = 600):
        """Row self-expires after ttl_sec — no cleanup job needed."""
        self._timed(self.ps_leaderboard_upsert, [
            race_id, driver, lap_number, urgency,
            _dt.datetime.now(_dt.timezone.utc), ttl_sec,
        ])

    def get_leaderboard(self, race_id: str) -> List[Dict]:
        rows = self._timed(self.ps_leaderboard_get, [race_id])
        return [dict(r._asdict()) for r in rows]

    # -- B2: counters ------------------------------------------------------
    def increment_counter(self, race_id: str, counter_name: str, amount: int = 1):
        self._timed(self.ps_counter_incr, [amount, race_id, counter_name])

    def get_counter(self, race_id: str, counter_name: str) -> int:
        rows = self._timed(self.ps_counter_get, [race_id, counter_name])
        row = rows.one()
        return row.count if row else 0

    # -- B3: collections -----------------------------------------------
    def add_compound_used(self, race_id: str, driver: str, compound: str):
        self._timed(self.ps_add_compound, [{compound}, race_id, driver])

    def append_stint(self, race_id: str, driver: str, stint_label: str):
        self._timed(self.ps_append_stint, [[stint_label], race_id, driver])

    def get_driver_summary(self, race_id: str, driver: str) -> Optional[Dict]:
        rows = self._timed(self.ps_get_driver_summary, [race_id, driver])
        row = rows.one()
        return dict(row._asdict()) if row else None

    # -- B4: materialized view -------------------------------------------
    def get_predictions_by_driver(self, driver: str, limit: int = 50) -> List[Dict]:
        rows = self._timed(self.ps_predictions_by_driver, [driver, limit])
        return [dict(r._asdict()) for r in rows]

    # -- B5: lightweight transaction --------------------------------------
    def upsert_driver_if_not_exists(self, race_id: str, driver: str, max_lap: int) -> bool:
        """Compare-and-set write via Paxos: only inserts if this
        (race_id, driver) doesn't already exist. Returns True if this
        call created the row, False if it already existed — a real
        linearizable guarantee, unlike Cassandra's normal
        eventually-consistent writes (see upsert_driver() above, which
        would silently overwrite)."""
        rows = self._timed(self.ps_upsert_driver_lwt, [race_id, driver, max_lap])
        row = rows.one()
        return bool(row.applied) if row else False

    # ===================================================================
    # END DB ENHANCEMENT PASS
    # ===================================================================

    # -- timing helper ------------------------------------------------
    def _timed(self, statement, params):
        t0 = time.perf_counter()
        rows = self.session.execute(statement, params)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        return rows

    # -- writes -------------------------------------------------------
    def insert_lap(self, lap: Dict):
        self._timed(self.ps_insert_lap, [lap.get(c) for c in LAP_COLUMNS])

    def insert_laps_batched(self, laps: List[Dict], batch_size: int = 50):
        """Batches are ONLY safe here because we group by partition key.
        A Cassandra batch spanning many partitions is an anti-pattern: the
        coordinator must fan out and hold the whole batch, which is slower
        than independent async writes. Same-partition batches are atomic
        and genuinely cheaper, so we group first."""
        by_partition: Dict = {}
        for lap in laps:
            by_partition.setdefault((lap["race_id"], lap["driver"]), []).append(lap)

        for _, group in by_partition.items():
            for i in range(0, len(group), batch_size):
                batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
                for lap in group[i:i + batch_size]:
                    batch.add(self.ps_insert_lap, [lap.get(c) for c in LAP_COLUMNS])
                t0 = time.perf_counter()
                self.session.execute(batch)
                self.latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    def log_prediction(self, **kw):
        self._timed(self.ps_insert_pred, [
            kw["run_id"], kw["race_id"], kw["request_ts"], kw["request_id"],
            kw["driver"], kw["lap_number"], kw["urgency_score"],
            kw["model_used"], kw["data_source"], kw.get("scheduling_policy", ""),
            kw.get("pit"), kw.get("pit_prob"), kw.get("compound"),
            kw.get("wait_time_ms"), kw.get("response_time_ms"),
        ])

    def upsert_driver(self, race_id: str, driver: str, max_lap: int):
        self._timed(self.ps_upsert_driver, [race_id, driver, max_lap])

    # -- reads --------------------------------------------------------
    def get_lap(self, race_id: str, driver: str, lap_number: int) -> Optional[Dict]:
        rows = self._timed(self.ps_get_lap, [race_id, driver, lap_number])
        row = rows.one()
        return dict(row._asdict()) if row else None

    def get_recent_laps(self, race_id: str, driver: str, lap_number: int,
                        window: int = 8) -> List[Dict]:
        """Returns up to `window` laps ending at lap_number, OLDEST FIRST —
        which is the order predict_smart() requires. The table stores them
        newest-first, so we reverse here rather than making every caller
        remember to."""
        rows = self._timed(self.ps_get_window,
                           [race_id, driver, lap_number, window])
        out = [dict(r._asdict()) for r in rows]
        return list(reversed(out))

    # -- benchmark support --------------------------------------------
    def reset_latencies(self):
        self.latencies_ms = []

    def latency_stats(self) -> Dict:
        if not self.latencies_ms:
            return {"n": 0}
        s = sorted(self.latencies_ms)
        return {
            "n": len(s),
            "avg_ms": sum(s) / len(s),
            "p50_ms": s[len(s) // 2],
            "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))],
            "p99_ms": s[min(len(s) - 1, int(0.99 * len(s)))],
            "max_ms": s[-1],
        }

    def close(self):
        self.cluster.shutdown()
