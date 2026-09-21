# F1 Strategy Scheduler

Two-tier F1 race-strategy prediction system with urgency-aware scheduling
and resource allocation, backed by a NoSQL telemetry store.

## Architecture

```
                         ┌──────────────────────────────┐
   f1_dataset.parquet ──▶│  sched/replay.py             │  OS role 5
   (22 races, 23k laps)  │  lap-by-lap → request stream │
                         │  bursts at overtakes /       │
                         │  pit windows / cautions      │
                         └──────────────┬───────────────┘
                                        │ Request(arrival, urgency,
                                        │         resource, service_ms)
                         ┌──────────────▼───────────────┐
                         │  sched/urgency.py            │  OS role 1
                         │  rule-based score ∈ [0,1]    │
                         └──────────────┬───────────────┘
                                        │
             ┌──────────────────────────▼──────────────────────────┐
             │  sched/router.py          WHERE it runs (allocation)│  OS roles 2,4
             │  urgent  → fast_local  + Redis                      │
             │  calm    → smart_cloud + Cassandra                  │
             └──────────────────────────┬──────────────────────────┘
                                        │
             ┌──────────────────────────▼──────────────────────────┐
             │  sched/scheduler.py       WHEN it runs (ordering)   │  OS role 3
             │  per-resource ready queues                          │
             │  PRIORITY vs FCFS over the identical stream         │
             └──────────────────────────┬──────────────────────────┘
                                        │
                         ┌──────────────▼───────────────┐
                         │  sched/metrics.py            │  OS role 6
                         │  response time / overhead /  │
                         │  utilisation → CSV + plots   │
                         └──────────────────────────────┘

   Data plane (DB workstream), reached only through data/store.py:

        data/store.py  ──┬──▶  data/redis_client.py      (cache, urgent path)
         RaceStore       └──▶  data/cassandra_client.py  (durable, calm path)
                                       │
                                       └── predictions_log (shared team data)

        data/relational_baseline.py    SQLite, same workload, for the
                                       "why NoSQL" benchmark
        cloud/lambda_handler.py        smart model as a cloud function,
                                       so network latency is measured
```

**The central design point:** allocation (*where*) and scheduling (*when*)
are separate modules, measured separately. That separation is the OS-course
contribution and the patent claim.

## Two architectures — run both

`config.ROUTING_MODE`, or `run_experiment.py --mode`.

**`partitioned`** is the original brief: urgent → fast/local, calm →
smart/cloud. It does not work, and knowing *why* is worth more than the
result would have been. Because resource assignment is a pure function of
urgency, each queue ends up urgency-homogeneous:

```
smart_cloud  n=715  urgency 0.00-0.45
fast_local   n=438  urgency 0.50-1.00
```

A priority queue holding only high-priority items has nothing to reorder,
so FCFS is already optimal and PRIORITY measures slightly worse
(urgent p95 5.13 ms → 6.59 ms, +28.5%). Tuning does not rescue it: the
routing threshold sweep (0.5/0.3/0.2) bottoms out at −3.5%, and raising
cloud network latency changes nothing because it only touches the tier
that has no urgent work in it. **The allocation decision had silently
absorbed the entire benefit the scheduler was meant to demonstrate.**

**`two_stage`** (default) fixes it architecturally. Every lap gets a
fast-path answer on the local tier, so that tier carries the full urgency
range; laps with latency slack additionally fire an asynchronous cloud
refinement. One contended resource now sees mixed priorities, which is
the precondition for priority scheduling to do anything at all.

All 22 races, 43,468 requests:

| | FCFS | PRIORITY | change |
|---|---|---|---|
| urgent avg response | 23.07 ms | 9.25 ms | **−59.9%** |
| urgent p95 response | 113.59 ms | 43.05 ms | **−62.1%** |
| urgent p99 response | 141.99 ms | 80.50 ms | −43.3% |
| non-urgent avg response | 35.44 ms | 38.00 ms | +7.2% |
| non-urgent p95 response | 119.92 ms | 126.77 ms | +5.7% |
| scheduling overhead | — | 0.62 µs/req | 0.017% of service |

Report both halves. The non-urgent regression is priority scheduling's
textbook starvation cost, and `metrics.py` prints `max_wait_ms` so it is
visible rather than buried. A result showing only the win reads as
selective.

This is also a stronger patent claim than the original. "Urgent requests
skip the queue" is obvious. "A single latency-bounded local tier serves
all requests under urgency-ordered scheduling, with an asynchronous cloud
tier refining only those answers whose decision window is wide enough for
the round trip" is a specific mechanism with a measured effect.

## Layout

| Path | Role | What it is |
|---|---|---|
| `config.py` | both | every tunable; an experiment = this file + the dataset |
| `sched/urgency.py` | OS 1 | rule-based urgency score |
| `sched/router.py` | OS 2,4 | model + data-source allocation |
| `sched/scheduler.py` | OS 3,6 | discrete-event sim, PRIORITY vs FCFS |
| `sched/replay.py` | OS 5 | historical race → timed request stream |
| `sched/metrics.py` | OS 6 | tables, CSVs, plots |
| `run_experiment.py` | OS 5,6 | entrypoint |
| `data/schema.cql` | DB 1,2 | Cassandra DDL + key-design rationale |
| `data/cassandra_client.py` | DB 1,2 | prepared statements, timing |
| `data/redis_client.py` | DB 3 | cache, hit/miss counters |
| `data/store.py` | DB 6 | **the** interface for ML and OS leads |
| `data/load_dataset.py` | DB 1 | bulk ingest + cache warm |
| `data/relational_baseline.py` | DB 5 | SQLite comparison |
| `data/bench_db.py` | DB 5 | throughput / latency / hit rate |
| `cloud/lambda_handler.py` | DB 4 | smart model as cloud function |
| `cloud/measure_rtt.py` | DB 4 | measures real network latency |
| `tests/test_sched.py` | — | guards the scheduler's correctness |

ML files (`models.py`, `train.py`, `predict_service.py`,
`fastf1_pipeline.py`) stay at the root, untouched.

## Running it

```bash
pip install -r requirements.txt

python tests/test_sched.py                  # 13 checks, all should pass
python run_experiment.py --mode partitioned --races 2022_Monaco_Grand_Prix
python run_experiment.py --mode two_stage                  # all 22 races

docker compose up -d                        # Cassandra + Redis
sleep 75
docker compose exec cassandra cqlsh -f /schema.cql
python -m data.load_dataset
python run_db_benchmark.py
```

## Before you report any number

1. **`config.FAST_SERVICE_MS` / `SMART_COMPUTE_MS` are placeholders.**
   Replace them with real `benchmark_speed()` output.
2. **`config.SMART_NETWORK_MS` is 0.0.** Until `cloud/lambda_handler.py`
   is deployed and `cloud/measure_rtt.py` has been run, the fast/smart gap
   is understated by orders of magnitude.
3. **`REDIS_LOOKUP_MS` / `CASSANDRA_LOOKUP_MS` are placeholders.** Fill
   from `run_db_benchmark.py`.
4. **SQLite will probably beat single-node Cassandra on this dataset.**
   23k rows fits in page cache; Cassandra's advantages are write
   throughput under concurrency, horizontal scale, and bounded-partition
   time-series reads at sizes this dataset never reaches. Report that
   honestly and argue the NoSQL case on the access pattern and on scaling,
   not on a point-read latency number you will lose.
