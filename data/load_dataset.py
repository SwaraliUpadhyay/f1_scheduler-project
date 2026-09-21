"""
load_dataset.py  —  DB role 1, ingest

Bulk-loads f1_dataset.parquet into Cassandra and warms Redis.

    python -m data.load_dataset                       # all races
    python -m data.load_dataset --races 2022_Monaco_Grand_Prix
    python -m data.load_dataset --no-cache            # skip cache warming
"""

import argparse

from sched import replay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", nargs="*", default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    from data.store import RaceStore

    df = replay.load_races(race_ids=args.races)
    laps = [replay.lap_to_dict(row) for _, row in df.iterrows()]
    print(f"Ingesting {len(laps)} laps across {df['race_id'].nunique()} race(s) ...")

    store = RaceStore()
    store.ingest_laps(laps, warm_cache=not args.no_cache)

    # race_drivers powers cache prefetching before a replay run.
    for (race_id, driver), g in df.groupby(["race_id", "driver"]):
        store.db.upsert_driver(race_id, driver, int(g["lap_number"].max()))

    print("Done.", store.stats())
    store.close()


if __name__ == "__main__":
    main()
