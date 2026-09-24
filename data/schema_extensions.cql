-- schema_extensions.cql  —  DB enhancement pass
--
-- Adds 5 User-Defined Functions (+ 1 User-Defined Aggregate) and 5
-- NoSQL-native features on top of the tables already created by
-- schema.cql. Nothing here modifies or drops an existing table.
--
-- Run AFTER schema.cql:
--     cqlsh -f data/schema.cql
--     cqlsh -f data/schema_extensions.cql
--
-- REQUIRES two cluster settings OFF by default (security defaults for a
-- multi-tenant cluster; safe to enable on a single-node dev cluster).
-- In cassandra.yaml, set:
--     enable_user_defined_functions: true
--     enable_materialized_views: true
-- then restart Cassandra before running this file.

USE f1;

-- =======================================================================
-- PART A — User-Defined Functions (DB role: UDF pass)
-- =======================================================================

-- A1. urgency_score_udf — a CQL port of sched/urgency.py:score_urgency().
-- Weights are literal copies of the constants in config.py. If config.py
-- changes, this must be edited to match — a real, worth-noting limitation
-- of duplicating one rule set across Python and Java, and worth a
-- sentence in the report.
CREATE OR REPLACE FUNCTION urgency_score_udf(
    gap_ahead float, gap_behind float, tyre_life float,
    is_caution float, rainfall boolean, field_pits_last_5 int
)
CALLED ON NULL INPUT
RETURNS float
LANGUAGE java AS $$
    double score = 0.0;
    if (gap_behind != null) {
        if (gap_behind < 1.0) score += 0.35;
        else if (gap_behind < 2.0) score += 0.15;
    }
    if (gap_ahead != null) {
        if (gap_ahead < 1.0) score += 0.15;
        else if (gap_ahead < 2.0) score += 0.05;
    }
    if (is_caution != null && is_caution >= 1.0) score += 0.30;
    if (rainfall != null && rainfall) score += 0.20;
    if (tyre_life != null && tyre_life >= 20.0) score += 0.15;
    if (field_pits_last_5 != null && field_pits_last_5 >= 3) score += 0.10;
    return (float) Math.min(score, 1.0);
$$;

-- A2. pit_window_flag — is this lap inside the "tyres are worn enough to
-- pit" window.
CREATE OR REPLACE FUNCTION pit_window_flag(tyre_life float)
CALLED ON NULL INPUT
RETURNS boolean
LANGUAGE java AS $$
    if (tyre_life == null) return false;
    return tyre_life >= 20.0;
$$;

-- A3. gap_threat_level — classifies defensive pressure from the gaps.
CREATE OR REPLACE FUNCTION gap_threat_level(gap_ahead float, gap_behind float)
CALLED ON NULL INPUT
RETURNS text
LANGUAGE java AS $$
    double behind = (gap_behind == null) ? 999.0 : gap_behind;
    double ahead  = (gap_ahead  == null) ? 999.0 : gap_ahead;
    if (behind < 1.0) return "under_attack";
    if (behind < 2.0 || ahead < 1.0) return "pressure";
    return "safe";
$$;

-- A4. response_time_bucket — buckets a prediction's latency for reporting
-- straight out of predictions_log, no Python post-processing needed.
CREATE OR REPLACE FUNCTION response_time_bucket(response_time_ms float)
CALLED ON NULL INPUT
RETURNS text
LANGUAGE java AS $$
    if (response_time_ms == null) return "unknown";
    if (response_time_ms < 10.0) return "fast";
    if (response_time_ms < 60.0) return "medium";
    return "slow";
$$;

-- A5. avg_urgency_by_race — a real User-Defined AGGREGATE (one step above
-- a scalar UDF). SFUNC folds each row into a running (count, total) pair;
-- FINALFUNC turns that into the average once the fold is done.
CREATE OR REPLACE FUNCTION avg_urgency_state(state tuple<int, float>, next_val float)
CALLED ON NULL INPUT
RETURNS tuple<int, float>
LANGUAGE java AS $$
    if (next_val == null) return state;
    int count = state.getInt(0) + 1;
    float total = state.getFloat(1) + next_val;
    state.setInt(0, count);
    state.setFloat(1, total);
    return state;
$$;

CREATE OR REPLACE FUNCTION avg_urgency_final(state tuple<int, float>)
CALLED ON NULL INPUT
RETURNS float
LANGUAGE java AS $$
    int count = state.getInt(0);
    if (count == 0) return null;
    return state.getFloat(1) / count;
$$;

CREATE OR REPLACE AGGREGATE avg_urgency_by_race(float)
SFUNC avg_urgency_state
STYPE tuple<int, float>
FINALFUNC avg_urgency_final
INITCOND (0, 0.0);
-- Usage:  SELECT avg_urgency_by_race(urgency_score) FROM predictions_log
--         WHERE run_id=? AND race_id=?;


-- =======================================================================
-- PART B — NoSQL-native features (things a relational baseline has no
-- direct equivalent for, or handles very differently)
-- =======================================================================

-- B1. TTL — a self-expiring "live leaderboard" mirror. Rows written here
-- vanish on their own after their TTL; nothing has to run a cleanup job.
-- (relational_baseline.py has no equivalent — SQLite would need an
-- external DELETE job on a schedule.)
CREATE TABLE IF NOT EXISTS live_leaderboard (
    race_id     text,
    driver      text,
    lap_number  int,
    urgency     float,
    updated_at  timestamp,
    PRIMARY KEY (race_id, driver)
);

-- B2. Counter columns — atomic, coordinator-side increments. No
-- read-modify-write race between concurrent writers, unlike
-- `UPDATE ... SET n = n + 1` on a plain column under concurrency.
CREATE TABLE IF NOT EXISTS race_event_counters (
    race_id       text,
    counter_name  text,   -- e.g. 'predictions_fast_local', 'pit_stops'
    count         counter,
    PRIMARY KEY (race_id, counter_name)
);

-- B3. Collections — denormalized nested data in one row instead of a
-- join table. compounds_used is a SET (auto-dedups); stint_history is a
-- LIST (preserves tyre-stint order).
CREATE TABLE IF NOT EXISTS driver_race_summary (
    race_id         text,
    driver          text,
    compounds_used  set<text>,
    stint_history   list<text>,   -- e.g. "SOFT:1-12", "MEDIUM:13-30"
    PRIMARY KEY (race_id, driver)
);

-- B4. Materialized View — Cassandra maintains this automatically from
-- predictions_log; nothing ever writes to it directly. Lets the same
-- data be queried "by driver" even though predictions_log is partitioned
-- by (run_id, race_id) for the OS/write path.
CREATE MATERIALIZED VIEW IF NOT EXISTS predictions_by_driver AS
    SELECT run_id, race_id, driver, request_ts, request_id, lap_number,
           urgency_score, model_used, data_source, scheduling_policy,
           pit, pit_prob, compound, wait_time_ms, response_time_ms
    FROM predictions_log
    WHERE run_id IS NOT NULL AND race_id IS NOT NULL
      AND driver IS NOT NULL AND request_ts IS NOT NULL
      AND request_id IS NOT NULL
    PRIMARY KEY (driver, run_id, race_id, request_ts, request_id)
    WITH CLUSTERING ORDER BY (
        run_id ASC,
        race_id ASC,
        request_ts DESC,
        request_id ASC
    );

-- B5. Lightweight Transaction (Paxos-based compare-and-set) — no new
-- table needed. This is a query-pattern addition against the EXISTING
-- race_drivers table: `INSERT ... IF NOT EXISTS` gives a real
-- linearizable guarantee, unlike Cassandra's normal eventually-consistent
-- writes. Implemented in cassandra_client.py as
-- upsert_driver_if_not_exists(); no schema change required for this one.
