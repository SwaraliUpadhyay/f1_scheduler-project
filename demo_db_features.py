"""
demo_db_features.py  —  DB enhancement pass demo

Exercises all 5 User-Defined Functions (+ 1 User-Defined Aggregate) and
all 5 NoSQL-native features added in:
    data/schema_extensions.cql
    cassandra_client.py   (new methods, existing ones untouched)
    store.py               (new methods + 2 extra writes inside
                             log_prediction)

This script does NOT touch sched/, models.py, train.py,
predict_service.py, or fastf1_pipeline.py — it only exercises the DB
layer, standalone.

SETUP (run once, in order):
    cqlsh -f data/schema.cql
    cqlsh -f data/schema_extensions.cql
    python data/load_dataset.py --races 2022_Monaco_Grand_Prix

USAGE:
    python demo_db_features.py --race 2022_Monaco_Grand_Prix --driver VER --lap 20
"""
import argparse

from data.store import RaceStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--race", required=True, help="race_id, e.g. 2022_Monaco_Grand_Prix")
    ap.add_argument("--driver", required=True, help="driver code, e.g. VER")
    ap.add_argument("--lap", type=int, required=True, help="lap_number to read/write")
    args = ap.parse_args()

    store = RaceStore()

    print("\n=== UDFs: enriched lap read ===")
    print("(urgency score, pit-window flag, threat level all computed INSIDE Cassandra)")
    print(store.get_lap_enriched(args.race, args.driver, args.lap))

    # log a couple of predictions so there's something for the UDA /
    # bucket / leaderboard / counter reads below to show
    store.log_prediction(
        race_id=args.race, driver=args.driver, lap_number=args.lap,
        urgency_score=0.72, model_used="fast_local", data_source="redis_cache",
        pit=1, pit_prob=0.81, compound="MEDIUM", response_time_ms=7.4,
    )
    store.log_prediction(
        race_id=args.race, driver=args.driver, lap_number=args.lap + 1,
        urgency_score=0.20, model_used="smart_cloud", data_source="cassandra",
        pit=0, pit_prob=0.05, compound="MEDIUM", response_time_ms=48.9,
    )

    print("\n=== UDA: average urgency logged for this run/race ===")
    print(store.avg_urgency_for_race(args.race))

    print("\n=== UDF: response-time bucket counts (fast/medium/slow) ===")
    print(store.response_time_bucket_counts(args.race))

    print("\n=== NoSQL B1 (TTL): live leaderboard (rows self-expire after 10 min) ===")
    print(store.get_leaderboard(args.race))

    print("\n=== NoSQL B2 (counters): event counts ===")
    for name in ("predictions_fast_local", "predictions_smart_cloud", "pit_stops"):
        print(f"  {name}: {store.get_counter(args.race, name)}")

    print("\n=== NoSQL B3 (collections): driver summary (set + list) ===")
    store.add_compound_used(args.race, args.driver, "MEDIUM")
    store.add_compound_used(args.race, args.driver, "SOFT")   # duplicate compound OK, set dedups
    store.append_stint(args.race, args.driver, f"MEDIUM:1-{args.lap}")
    print(store.get_driver_summary(args.race, args.driver))

    print("\n=== NoSQL B4 (materialized view): predictions by driver, auto-maintained ===")
    print(store.get_predictions_by_driver(args.driver, limit=5))

    print("\n=== NoSQL B5 (lightweight transaction): upsert-if-not-exists ===")
    created_first = store.upsert_driver_if_not_exists(args.race, args.driver, args.lap)
    created_second = store.upsert_driver_if_not_exists(args.race, args.driver, args.lap)
    print(f"  first call created row:  {created_first}  (expect True)")
    print(f"  second call created row: {created_second}  (expect False — row already exists)")

    store.close()


if __name__ == "__main__":
    main()
