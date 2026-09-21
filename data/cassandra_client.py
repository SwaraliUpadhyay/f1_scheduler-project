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
