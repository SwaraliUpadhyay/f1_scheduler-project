"""
db_api.py  —  HTTP adapter for the DB enhancement pass (Part A + Part B).

Mounted by api.py. Each of the 10 features has one runner that calls the
existing RaceStore facade (data/store.py) and returns the result together
with the CQL that produced it, so the dashboard can show that the work
happens inside Cassandra rather than in Python.

Nothing here reimplements database logic; it only calls RaceStore and
formats the output.

Endpoints
    GET  /api/db/status                  Cassandra reachable? + feature list
    GET  /api/db/options?race=...        drivers / lap range from the parquet
    POST /api/db/seed                    start a fresh run_id, log demo predictions
    POST /api/db/run/{feature}           run one feature
    POST /api/db/run-all                 seed + run all 10
"""
import datetime as _dt
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import CASSANDRA_KEYSPACE, DATASET_PATH

router = APIRouter(prefix="/api/db", tags=["database"])


# ---------------------------------------------------------------------
# Store management
# ---------------------------------------------------------------------
class _NoCache:
    """The DB features only touch Cassandra; this keeps the demo from
    needing Redis to be up."""
    def stats(self): return {}
    def reset_stats(self): pass


_lock = threading.Lock()
_state: Dict = {"store": None, "seeded": False, "seed_info": None}

# Tests (or a different deployment) can replace this to inject a store.
_store_factory: Callable = lambda: _default_store()


def _default_store():
    from data.cassandra_client import CassandraClient
    from data.store import RaceStore
    return RaceStore(cassandra=CassandraClient(keyspace=CASSANDRA_KEYSPACE),
                     cache=_NoCache())


def _get_store():
    with _lock:
        if _state["store"] is None:
            try:
                _state["store"] = _store_factory()
            except Exception as exc:  # driver import error, NoHostAvailable, ...
                raise HTTPException(
                    status_code=503,
                    detail=("Cassandra is not reachable. Start it (docker compose up -d), "
                            "then run data/schema.cql and data/schema_extensions.cql. "
                            f"Driver said: {type(exc).__name__}: {exc}"),
                )
        return _state["store"]


def _reset_store():
    with _lock:
        s, _state["store"] = _state["store"], None
        _state["seeded"] = False
        _state["seed_info"] = None
    if s is not None:
        try:
            s.close()
        except Exception:
            pass


class DbRequest(BaseModel):
    race: str
    driver: str
    lap: int


def _clean(v):
    """Make Cassandra values JSON friendly (sets, datetimes, float32 noise)."""
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


# ---------------------------------------------------------------------
# Demo predictions (same four-ish rows demo_db_features.py logs)
# ---------------------------------------------------------------------
_SEED_ROWS = [
    # urgency, model_used,    data_source,   pit, pit_prob, compound, response_ms
    (0.72, "fast_local",  "redis_cache", 1, 0.81, "MEDIUM", 7.4),
    (0.20, "smart_cloud", "cassandra",   0, 0.05, "MEDIUM", 48.9),
    (0.55, "fast_local",  "redis_cache", 0, 0.30, "MEDIUM", 5.1),
    (0.90, "smart_cloud", "cassandra",   1, 0.92, "HARD",  96.5),
]


def _seed(req: DbRequest) -> Dict:
    """Fresh run_id + a handful of logged predictions. log_prediction()
    also feeds the live leaderboard (TTL) and event counters as a side
    effect, exactly as in the real pipeline."""
    import uuid
    store = _get_store()
    store.run_id = uuid.uuid4().hex[:12]
    for i, (u, model, src, pit, pp, comp, ms) in enumerate(_SEED_ROWS):
        store.log_prediction(
            race_id=req.race, driver=req.driver, lap_number=req.lap + i,
            urgency_score=u, model_used=model, data_source=src, pit=pit,
            pit_prob=pp, compound=comp, response_time_ms=ms,
        )
    info = {"run_id": store.run_id, "predictions_logged": len(_SEED_ROWS),
            "race": req.race, "driver": req.driver}
    _state["seeded"], _state["seed_info"] = True, info
    return info


def _ensure_seeded(req: DbRequest):
    """Seed on first use, and again if the user switched race/driver."""
    info = _state["seed_info"] or {}
    if not _state["seeded"] or info.get("race") != req.race or info.get("driver") != req.driver:
        _seed(req)


# ---------------------------------------------------------------------
# Part A — User-Defined Functions
# ---------------------------------------------------------------------
def _enriched(store, req: DbRequest) -> Dict:
    row = store.get_lap_enriched(req.race, req.driver, req.lap)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(f"No telemetry row for {req.race} / {req.driver} / lap {req.lap}. "
                    "Load it first: python -m data.load_dataset --races " + req.race),
        )
    return row


def a1_urgency(req: DbRequest) -> Dict:
    store = _get_store()
    row = _enriched(store, req)
    out = {
        "inputs": {k: row.get(k) for k in ("gap_ahead", "gap_behind", "tyre_life")},
        "computed_urgency": row["computed_urgency"],
    }
    # Parity check against the Python rule set the UDF was ported from.
    try:
        from sched.urgency import RaceSituation, score_urgency
        raw = store.db.get_lap(req.race, req.driver, req.lap)
        py = score_urgency(RaceSituation(
            gap_ahead=raw["gap_ahead"], gap_behind=raw["gap_behind"],
            tyre_life=raw["tyre_life"] or 0.0, is_caution=raw["is_caution"] or 0.0,
            rainfall=bool(raw["rainfall"]), field_pits_last_5=raw["field_pits_last_5"] or 0))
        out["python_urgency"] = round(py, 4)
        out["matches_python"] = abs(py - float(row["computed_urgency"] or 0)) < 1e-3
        out["inputs"].update({k: raw.get(k) for k in
                              ("is_caution", "rainfall", "field_pits_last_5")})
    except Exception as exc:
        out["parity_check"] = f"skipped ({type(exc).__name__}: {exc})"
    return {
        "cql": ("SELECT urgency_score_udf(gap_ahead, gap_behind, tyre_life, is_caution,\n"
                "       rainfall, field_pits_last_5) AS computed_urgency\n"
                "FROM telemetry_laps WHERE race_id=? AND driver=? AND lap_number=?"),
        "result": out,
    }


def a2_pit_window(req: DbRequest) -> Dict:
    row = _enriched(_get_store(), req)
    return {
        "cql": ("SELECT tyre_life, pit_window_flag(tyre_life) AS in_pit_window\n"
                "FROM telemetry_laps WHERE race_id=? AND driver=? AND lap_number=?"),
        "result": {"tyre_life": row["tyre_life"], "in_pit_window": row["in_pit_window"],
                   "rule": "tyre_life >= 20 laps"},
    }


def a3_threat(req: DbRequest) -> Dict:
    row = _enriched(_get_store(), req)
    return {
        "cql": ("SELECT gap_ahead, gap_behind, gap_threat_level(gap_ahead, gap_behind)\n"
                "       AS threat_level\n"
                "FROM telemetry_laps WHERE race_id=? AND driver=? AND lap_number=?"),
        "result": {"gap_ahead": row["gap_ahead"], "gap_behind": row["gap_behind"],
                   "threat_level": row["threat_level"],
                   "rule": "behind<1s: under_attack · behind<2s or ahead<1s: pressure · else safe"},
    }


def a4_buckets(req: DbRequest) -> Dict:
    store = _get_store()
    _ensure_seeded(req)
    return {
        "cql": ("SELECT response_time_bucket(response_time_ms) AS bucket\n"
                "FROM predictions_log WHERE run_id=? AND race_id=?"),
        "result": {"run_id": store.run_id,
                   "bucket_counts": store.response_time_bucket_counts(req.race),
                   "rule": "<10 ms fast · <60 ms medium · else slow"},
    }


def a5_uda(req: DbRequest) -> Dict:
    store = _get_store()
    _ensure_seeded(req)
    server = store.avg_urgency_for_race(req.race)
    values = store.urgency_values_for_race(req.race)
    client = sum(values) / len(values) if values else None
    return {
        "cql": ("SELECT avg_urgency_by_race(urgency_score) AS avg_urgency\n"
                "FROM predictions_log WHERE run_id=? AND race_id=?"),
        "result": {"run_id": store.run_id,
                   "server_side_avg": server,
                   "rows_folded": len(values),
                   "client_side_check": None if client is None else round(client, 4),
                   "matches": (server is not None and client is not None
                               and abs(server - client) < 1e-3)},
    }


# ---------------------------------------------------------------------
# Part B — NoSQL-native features
# ---------------------------------------------------------------------
def b1_ttl(req: DbRequest) -> Dict:
    store = _get_store()
    _ensure_seeded(req)
    rows = store.get_leaderboard_with_ttl(req.race)
    return {
        "cql": ("INSERT INTO live_leaderboard (race_id, driver, lap_number, urgency, updated_at)\n"
                "VALUES (?, ?, ?, ?, ?) USING TTL 600;\n\n"
                "SELECT driver, lap_number, urgency, TTL(urgency) AS ttl_remaining_sec\n"
                "FROM live_leaderboard WHERE race_id=?"),
        "result": {"ttl_seconds_at_write": 600, "rows": rows},
    }


def b2_counters(req: DbRequest) -> Dict:
    store = _get_store()
    _ensure_seeded(req)
    name = "demo_concurrent"
    before = store.get_counter(req.race, name)
    n = 20
    # 20 writers in parallel; a read-modify-write column would lose updates.
    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda _: store.db.increment_counter(req.race, name), range(n)))
    after = store.get_counter(req.race, name)
    return {
        "cql": ("UPDATE race_event_counters SET count = count + 1\n"
                "WHERE race_id=? AND counter_name=?"),
        "result": {
            "event_counters": {c: store.get_counter(req.race, c) for c in
                               ("predictions_fast_local", "predictions_smart_cloud", "pit_stops")},
            "concurrency_test": {"parallel_increments": n, "before": before, "after": after,
                                 "expected": before + n, "no_lost_updates": after == before + n},
        },
    }


def b3_collections(req: DbRequest) -> Dict:
    store = _get_store()
    store.add_compound_used(req.race, req.driver, "MEDIUM")
    store.add_compound_used(req.race, req.driver, "SOFT")
    store.add_compound_used(req.race, req.driver, "SOFT")     # duplicate on purpose
    label = f"MEDIUM:1-{req.lap}"
    existing = store.get_driver_summary(req.race, req.driver) or {}
    if label not in (existing.get("stint_history") or []):
        store.append_stint(req.race, req.driver, label)
    summary = store.get_driver_summary(req.race, req.driver) or {}
    return {
        "cql": ("UPDATE driver_race_summary SET compounds_used = compounds_used + {'SOFT'}\n"
                "WHERE race_id=? AND driver=?;   -- run twice, stored once\n"
                "UPDATE driver_race_summary SET stint_history = stint_history + ['MEDIUM:1-N']\n"
                "WHERE race_id=? AND driver=?"),
        "result": {"compounds_used (SET)": summary.get("compounds_used"),
                   "stint_history (LIST)": summary.get("stint_history"),
                   "note": "SOFT was added twice; the SET keeps one copy."},
    }


def b4_mview(req: DbRequest) -> Dict:
    store = _get_store()
    _ensure_seeded(req)
    rows = store.get_predictions_by_driver(req.driver, limit=5)
    slim = [{k: r.get(k) for k in ("race_id", "lap_number", "urgency_score",
                                   "model_used", "response_time_ms", "request_ts")} for r in rows]
    return {
        "cql": ("-- predictions_log is partitioned by (run_id, race_id).\n"
                "-- The view re-keys it by driver, maintained by Cassandra:\n"
                "SELECT * FROM predictions_by_driver WHERE driver=? LIMIT 5"),
        "result": {"driver": req.driver, "rows_found": len(slim), "rows": slim},
    }


def b5_lwt(req: DbRequest) -> Dict:
    store = _get_store()
    scratch = f"{req.race}__lwt_demo_{store.run_id}"
    first = store.upsert_driver_if_not_exists(scratch, req.driver, req.lap)
    second = store.upsert_driver_if_not_exists(scratch, req.driver, req.lap)
    try:                                           # leave no demo rows behind
        store.db.delete_driver(scratch, req.driver)
    except Exception:
        pass
    existing = store.upsert_driver_if_not_exists(req.race, req.driver, req.lap)
    return {
        "cql": "INSERT INTO race_drivers (race_id, driver, max_lap)\nVALUES (?, ?, ?) IF NOT EXISTS",
        "result": {"first_insert_applied": first, "second_insert_applied": second,
                   "expected": "True then False",
                   "real_row_applied": existing,
                   "real_row_note": ("False means the dataset loader had already created "
                                     "this (race, driver) row — LWT refused to overwrite it."
                                     if not existing else
                                     "This (race, driver) row did not exist yet, so it was created.")},
    }


FEATURES = [
    # key, part, number, title, runner
    ("urgency_score_udf",     "A", 1, "urgency_score_udf",     a1_urgency),
    ("pit_window_flag",       "A", 2, "pit_window_flag",       a2_pit_window),
    ("gap_threat_level",      "A", 3, "gap_threat_level",      a3_threat),
    ("response_time_bucket",  "A", 4, "response_time_bucket",  a4_buckets),
    ("avg_urgency_by_race",   "A", 5, "avg_urgency_by_race",   a5_uda),
    ("ttl",                   "B", 1, "TTL",                   b1_ttl),
    ("counters",              "B", 2, "Counter columns",       b2_counters),
    ("collections",           "B", 3, "Collections",           b3_collections),
    ("materialized_view",     "B", 4, "Materialized view",     b4_mview),
    ("lwt",                   "B", 5, "Lightweight transaction", b5_lwt),
]
_RUNNERS = {k: (p, n, t, fn) for k, p, n, t, fn in FEATURES}


def _hint_for(exc: Exception) -> Optional[str]:
    msg = str(exc).lower()
    if "unconfigured table" in msg or "unconfigured columnfamily" in msg or "does not exist" in msg:
        return "Run data/schema_extensions.cql (and schema.cql) against the f1 keyspace."
    if "user defined function" in msg or "unknown function" in msg or "materialized" in msg:
        return ("Enable user_defined_functions and materialized_views in cassandra.yaml "
                "(see docker-compose.yml) and re-run data/schema_extensions.cql.")
    return None


def _run_feature(key: str, req: DbRequest) -> Dict:
    part, num, title, fn = _RUNNERS[key]
    t0 = perf_counter()
    base = {"key": key, "part": part, "number": num, "title": title}
    try:
        payload = fn(req)
        return {**base, "ok": True, "cql": payload["cql"],
                "result": _clean(payload["result"]),
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1)}
    except HTTPException as exc:
        return {**base, "ok": False, "error": exc.detail, "status": exc.status_code,
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1)}
    except Exception as exc:
        return {**base, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "hint": _hint_for(exc),
                "elapsed_ms": round((perf_counter() - t0) * 1000, 1)}


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@router.get("/status")
def db_status():
    features = [{"key": k, "part": p, "number": n, "title": t} for k, p, n, t, _ in FEATURES]
    try:
        _get_store()
        return {"connected": True, "keyspace": CASSANDRA_KEYSPACE, "features": features,
                "seeded": _state["seeded"], "seed_info": _state["seed_info"]}
    except HTTPException as exc:
        return {"connected": False, "error": exc.detail, "features": features}


@router.post("/reconnect")
def db_reconnect():
    _reset_store()
    return db_status()


@router.get("/options")
def db_options(race: Optional[str] = None):
    if not Path(DATASET_PATH).exists():
        raise HTTPException(status_code=404, detail="f1_dataset.parquet not found.")
    df = pd.read_parquet(DATASET_PATH, columns=["race_id", "driver", "lap_number"])
    races = sorted(df["race_id"].dropna().unique().tolist())
    race = race if race in races else (races[0] if races else None)
    sub = df[df["race_id"] == race]
    return {
        "races": races,
        "race": race,
        "drivers": sorted(sub["driver"].dropna().unique().tolist()),
        "min_lap": int(sub["lap_number"].min()) if len(sub) else 1,
        "max_lap": int(sub["lap_number"].max()) if len(sub) else 1,
    }


@router.post("/seed")
def db_seed(req: DbRequest):
    try:
        return _seed(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@router.post("/run/{feature}")
def db_run(feature: str, req: DbRequest):
    if feature not in _RUNNERS:
        raise HTTPException(status_code=404, detail=f"Unknown feature '{feature}'.")
    _get_store()                       # surface a clean 503 if Cassandra is down
    return _run_feature(feature, req)


@router.post("/run-all")
def db_run_all(req: DbRequest):
    _get_store()
    try:
        seed = _seed(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Seeding failed: {type(exc).__name__}: {exc}")
    return {"seed": seed, "results": [_run_feature(k, req) for k, *_ in FEATURES]}
