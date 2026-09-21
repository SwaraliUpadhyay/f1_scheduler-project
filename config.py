"""
config.py

Single source of truth for every tunable in the system. Both the OS
workstream (scheduler/router) and the DB workstream (Cassandra/Redis)
read from here, so an experiment is fully described by this file plus
the dataset — which is what makes runs reproducible for the report.
"""

import os as _os  # stdlib os — safe now that no local package shadows it

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = _os.path.dirname(_os.path.abspath(__file__))
DATASET_PATH = _os.path.join(ROOT, "f1_dataset.parquet")
FAST_MODEL_PATH = _os.path.join(ROOT, "fast_model.pkl")
SMART_MODEL_PATH = _os.path.join(ROOT, "smart_model.pkl")
RESULTS_DIR = _os.path.join(ROOT, "results")

# ---------------------------------------------------------------------
# Urgency classifier thresholds (OS role 1)
# ---------------------------------------------------------------------
GAP_CRITICAL_SEC = 1.0
GAP_WARNING_SEC = 2.0
TYRE_LIFE_HIGH_LAPS = 20
FIELD_PIT_PRESSURE_HIGH = 3
URGENCY_THRESHOLD = 0.5       # >= this counts as "urgent" for REPORTING

# ---------------------------------------------------------------------
# Routing architecture (OS roles 2, 4) — see README "Two architectures"
#
# "partitioned" : urgent -> fast_local, calm -> smart_cloud.
#                 This is the original brief. Each queue ends up
#                 urgency-homogeneous, so PRIORITY has nothing to reorder
#                 and measures no better than FCFS. Keep it — it is the
#                 motivating negative result.
#
# "two_stage"   : every request gets a fast-path answer on the local
#                 tier; requests with latency slack ALSO get an
#                 asynchronous smart-path refinement. One contended
#                 resource now sees the full urgency range, which is the
#                 precondition for priority scheduling to do anything.
# ---------------------------------------------------------------------
ROUTING_MODE = "two_stage"

# In two_stage mode, requests at or below this score are cheap enough in
# deadline terms to be worth a cloud refinement. Above it the decision
# window has already closed by the time the cloud replies.
REFINEMENT_MAX_URGENCY = 0.5

# ---------------------------------------------------------------------
# Service times (OS roles 4-6)
#
# Fill FAST_SERVICE_MS / SMART_COMPUTE_MS from an actual
# `python -m ml.benchmark` run. The defaults below are placeholders and
# MUST be replaced before you report numbers.
#
# SMART_NETWORK_MS is the round-trip to the deployed cloud function.
# Set it to 0.0 while the smart model runs locally; set it to the real
# measured RTT once cloud/lambda_handler.py is deployed (DB role 4).
# ---------------------------------------------------------------------
FAST_SERVICE_MS = 0.5152510001789778
SMART_COMPUTE_MS = 33.6
SMART_NETWORK_MS = 119.8

# Data-source access cost, added to service time by the router's
# allocation decision. Measure these with data/bench_db.py (DB role 5).
REDIS_LOOKUP_MS = 0.25
CASSANDRA_LOOKUP_MS = 3.50

# ---------------------------------------------------------------------
# Resource pools (OS role 4)
# ---------------------------------------------------------------------
FAST_WORKERS = 1      # one local process/thread
SMART_WORKERS = 1     # raise to model cloud elasticity as a second experiment

# ---------------------------------------------------------------------
# Replay simulation (OS role 5)
# ---------------------------------------------------------------------
LAP_DURATION_MS = 90_000.0     # ~90s green-flag lap, the replay clock tick
NORMAL_JITTER_MS = 400.0       # drivers stagger their requests on a calm lap
BURST_SPREAD_MS = 5.0          # high-pressure: everyone fires near-simultaneously
BURST_TYRE_LIFE = 18           # tyre age that counts as "in the pit window"
BURST_GAP_SEC = 1.0            # gap that counts as an overtake threat

# ---------------------------------------------------------------------
# Cassandra (DB roles 1-3)
# ---------------------------------------------------------------------
CASSANDRA_HOSTS = _os.environ.get("CASSANDRA_HOSTS", "127.0.0.1").split(",")
CASSANDRA_PORT = int(_os.environ.get("CASSANDRA_PORT", 9042))
CASSANDRA_KEYSPACE = "f1"

REDIS_HOST = _os.environ.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(_os.environ.get("REDIS_PORT", 6379))
REDIS_TTL_SEC = 300            # a lap is stale well before 5 minutes

RELATIONAL_DB_PATH = _os.path.join(ROOT, "results", "relational_baseline.db")

# ---------------------------------------------------------------------
# Cloud function (DB role 4)
# ---------------------------------------------------------------------
SMART_ENDPOINT_URL = _os.environ.get("SMART_ENDPOINT_URL", "")
