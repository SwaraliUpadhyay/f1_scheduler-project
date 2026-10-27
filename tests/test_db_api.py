"""
test_db_api.py — checks the HTTP adapter in db_api.py WITHOUT a Cassandra
server, by injecting an in-memory fake with the same method names as
data.store.RaceStore. It verifies the adapter's wiring and output shape;
it does NOT prove the CQL/UDFs work — that needs the real cluster
(see demo_db_features.py).

    python tests/test_db_api.py
"""
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

import api
import db_api


class FakeDB:
    def __init__(self, store):
        self.s = store
        self.lock = threading.Lock()

    def get_lap(self, race, driver, lap):
        return {"gap_ahead": 0.8, "gap_behind": 0.5, "tyre_life": 22.0,
                "is_caution": 0.0, "rainfall": False, "field_pits_last_5": 3}

    def increment_counter(self, race, name, amount=1):
        with self.lock:
            k = (race, name)
            self.s.counters[k] = self.s.counters.get(k, 0) + amount

    def delete_driver(self, race, driver):
        self.s.drivers.discard((race, driver))


class FakeStore:
    def __init__(self):
        self.run_id = "x"
        self.counters, self.preds, self.board = {}, [], {}
        self.drivers, self.summary = {("R", "VER")}, {}
        self.db = FakeDB(self)

    def log_prediction(self, race_id, driver, lap_number, urgency_score, model_used,
                       data_source, pit=None, pit_prob=None, compound=None,
                       response_time_ms=None, **_):
        self.preds.append((self.run_id, race_id, driver, urgency_score, response_time_ms))
        self.board[(race_id, driver)] = {"driver": driver, "lap_number": lap_number,
                                         "urgency": urgency_score, "ttl_remaining_sec": 599}
        self.db.increment_counter(race_id, f"predictions_{model_used}")
        if pit == 1:
            self.db.increment_counter(race_id, "pit_stops")

    def get_lap_enriched(self, race, driver, lap):
        return {"gap_ahead": 0.8, "gap_behind": 0.5, "tyre_life": 22.0,
                "computed_urgency": 0.35 + 0.15 + 0.15 + 0.10, "in_pit_window": True,
                "threat_level": "under_attack"}

    def _mine(self, race):
        return [p for p in self.preds if p[0] == self.run_id and p[1] == race]

    def avg_urgency_for_race(self, race):
        v = [p[3] for p in self._mine(race)]
        return sum(v) / len(v) if v else None

    def urgency_values_for_race(self, race):
        return [p[3] for p in self._mine(race)]

    def response_time_bucket_counts(self, race):
        c = {"fast": 0, "medium": 0, "slow": 0, "unknown": 0}
        for p in self._mine(race):
            ms = p[4]
            c["fast" if ms < 10 else "medium" if ms < 60 else "slow"] += 1
        return c

    def get_leaderboard_with_ttl(self, race):
        return [v for (r, _), v in self.board.items() if r == race]

    def get_counter(self, race, name):
        return self.counters.get((race, name), 0)

    def add_compound_used(self, race, driver, c):
        self.summary.setdefault("compounds_used", set()).add(c)

    def append_stint(self, race, driver, label):
        self.summary.setdefault("stint_history", []).append(label)

    def get_driver_summary(self, race, driver):
        return dict(self.summary) or None

    def get_predictions_by_driver(self, driver, limit=50):
        return [{"race_id": p[1], "urgency_score": p[3], "response_time_ms": p[4]}
                for p in self.preds if p[2] == driver][:limit]

    def upsert_driver_if_not_exists(self, race, driver, max_lap):
        if (race, driver) in self.drivers:
            return False
        self.drivers.add((race, driver))
        return True

    def close(self):
        pass


def main():
    store = FakeStore()
    db_api._store_factory = lambda: store
    client = TestClient(api.app)
    body = {"race": "R", "driver": "VER", "lap": 20}

    assert client.get("/api/db/status").json()["connected"] is True
    out = client.post("/api/db/run-all", json=body).json()
    res = {r["key"]: r for r in out["results"]}
    assert len(res) == 10, "all 10 features must run"
    assert all(r["ok"] for r in res.values()), {k: r.get("error") for k, r in res.items() if not r["ok"]}

    assert res["urgency_score_udf"]["result"]["matches_python"] is True
    assert res["pit_window_flag"]["result"]["in_pit_window"] is True
    assert res["gap_threat_level"]["result"]["threat_level"] == "under_attack"
    assert res["response_time_bucket"]["result"]["bucket_counts"] == {
        "fast": 2, "medium": 1, "slow": 1, "unknown": 0}
    assert res["avg_urgency_by_race"]["result"]["matches"] is True
    assert res["ttl"]["result"]["rows"][0]["ttl_remaining_sec"] > 0
    ct = res["counters"]["result"]["concurrency_test"]
    assert ct["no_lost_updates"] and ct["after"] - ct["before"] == 20
    assert res["collections"]["result"]["compounds_used (SET)"] == ["MEDIUM", "SOFT"]
    assert res["materialized_view"]["result"]["rows_found"] == 4
    lwt = res["lwt"]["result"]
    assert lwt["first_insert_applied"] is True and lwt["second_insert_applied"] is False

    # a missing telemetry row becomes a readable error, not a 500
    store.get_lap_enriched = lambda *a: None
    r = client.post("/api/db/run/pit_window_flag", json=body).json()
    assert r["ok"] is False and "No telemetry row" in r["error"]

    assert client.post("/api/db/run/nope", json=body).status_code == 404
    print("PASS  db_api adapter: 10 features, error handling")


if __name__ == "__main__":
    main()
