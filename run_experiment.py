"""
run_experiment.py  —  the OS core experiment (roles 5 + 6)

    python run_experiment.py                      # all 22 races
    python run_experiment.py --races 2022_Monaco_Grand_Prix
    python run_experiment.py --smart-workers 3    # cloud elasticity variant
    python run_experiment.py --threshold 0.4      # urgency sensitivity sweep

Writes CSVs and figures into results/.
"""

import argparse
import json
import os

from config import (FAST_WORKERS, SMART_WORKERS, RESULTS_DIR,
                    URGENCY_THRESHOLD, ROUTING_MODE)
from sched import metrics, replay
from sched.scheduler import compare_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", nargs="*", default=None,
                    help="race_id values to replay; default = all")
    ap.add_argument("--fast-workers", type=int, default=FAST_WORKERS)
    ap.add_argument("--smart-workers", type=int, default=SMART_WORKERS)
    ap.add_argument("--threshold", type=float, default=URGENCY_THRESHOLD)
    ap.add_argument("--mode", choices=["partitioned", "two_stage"],
                    default=ROUTING_MODE,
                    help="routing architecture; see README")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    print("Loading replay data ...")
    df = replay.load_races(race_ids=args.races)
    print(f"  {len(df)} lap records across {df['race_id'].nunique()} race(s)")

    print("Building request stream ...")
    requests = replay.build_request_stream(df, seed=args.seed,
                                           threshold=args.threshold,
                                           mode=args.mode)
    print(f"  mode={args.mode}  {len(requests)} requests")
    summary = replay.stream_summary(requests)

    print("Running PRIORITY and FCFS over the identical stream ...")
    results = compare_policies(requests,
                               fast_workers=args.fast_workers,
                               smart_workers=args.smart_workers)

    comparison = metrics.build_comparison(results)
    metrics.print_report(comparison, summary)

    written = metrics.write_csvs(results, comparison)
    if not args.no_plots:
        written += metrics.plot(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "run_config.json")
    with open(cfg_path, "w") as f:
        json.dump({"args": vars(args), "workload": summary}, f, indent=2)
    written.append(cfg_path)

    print("\nWrote:")
    for p in written:
        print("  ", p)


if __name__ == "__main__":
    main()
