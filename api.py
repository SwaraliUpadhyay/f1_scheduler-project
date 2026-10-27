
from pathlib import Path
from time import perf_counter
import subprocess
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DATASET_PATH, FAST_WORKERS, SMART_WORKERS
from sched import metrics, replay
from sched.scheduler import compare_policies


app = FastAPI(title="F1 Strategy Scheduler API")

# Allow the Vite development server to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


from db_api import router as db_router  # Part A / Part B database features
app.include_router(db_router)


class RunRequest(BaseModel):
    races: list[str] = Field(default_factory=list)
    mode: str = "two_stage"
    threshold: float = Field(default=0.5, ge=0, le=1)
    trials: int = Field(default=1, ge=1, le=5)
    fast_workers: int = Field(default=FAST_WORKERS, ge=1, le=16)
    smart_workers: int = Field(default=SMART_WORKERS, ge=1, le=16)
    seed: int = 42


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/races")
def get_races():
    if not Path(DATASET_PATH).exists():
        raise HTTPException(
            status_code=404,
            detail="f1_dataset.parquet was not found in the project root.",
        )

    try:
        df = pd.read_parquet(DATASET_PATH)
        races = sorted(df["race_id"].dropna().unique().tolist())
        return {"races": races}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/run")
def run_scheduler(request: RunRequest):
    if request.mode not in ("two_stage", "partitioned"):
        raise HTTPException(status_code=400, detail="Invalid routing mode.")

    if not Path(DATASET_PATH).exists():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    try:
        # Load only the selected races; an empty selection means all races.
        df = replay.load_races(race_ids=request.races or None)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="No lap records found for the selected races.",
            )

        # Generate the same workload for both scheduling policies.
        request_stream = replay.build_request_stream(
            df,
            seed=request.seed,
            threshold=request.threshold,
            mode=request.mode,
        )

        summary = replay.stream_summary(request_stream)

        # Execute the existing scheduler; do not reimplement its logic.
        durations = []
        results = None

        for trial in range(request.trials):
            start = perf_counter()
            results = compare_policies(
                request_stream,
                fast_workers=request.fast_workers,
                smart_workers=request.smart_workers,
            )
            durations.append((perf_counter() - start) * 1000)

        comparison = metrics.build_comparison(results)

        def policy_summary(name):
            policy = results[name]
            return {
                "overall": policy["overall"],
                "urgent": policy["urgent"],
                "non_urgent": policy["non_urgent"],
                "overhead": {
                    "us_per_request": policy["scheduling_overhead"][
                        "us_per_request"
                    ],
                },
                "busy_utilization": policy["busy_utilization"],
            }

        # Return a limited sample so the dashboard stays responsive.
        sample = []
        for r in results["priority"]["requests"][:300]:
            sample.append({
                "id": r.id,
                "race_id": r.race_id,
                "driver": r.driver,
                "lap_number": r.lap_number,
                "urgency_score": r.urgency_score,
                "resource": r.resource,
                "wait_ms": r.wait_time_ms,
                "response_ms": r.response_time_ms,
                "is_urgent": r.urgency_score >= request.threshold,
            })

        urgent_p95 = next(
            row for row in comparison["rows"]
            if row["group"] == "urgent"
            and row["metric"] == "p95_response_ms"
        )

        return {
            "race_count": int(df["race_id"].nunique()),
            "lap_records": len(df),
            "requests_total": len(request_stream),
            "workload": {
                **summary,
                "total_requests": len(request_stream),
                "urgent_requests": sum(
                    r.urgency_score >= request.threshold
                    for r in request_stream
                ),
                "urgent_fraction": (
                    sum(r.urgency_score >= request.threshold
                        for r in request_stream) / len(request_stream)
                    if request_stream else 0
                ),
            },
            "policies": {
                "priority": policy_summary("priority"),
                "fcfs": policy_summary("fcfs"),
            },
            "headline": {
                "urgent_p95_fcfs_ms": urgent_p95["fcfs"],
                "urgent_p95_priority_ms": urgent_p95["priority"],
                "change_pct": urgent_p95["pct_change"],
            },
            "runtime": {
                "trials": request.trials,
                "average_ms": sum(durations) / len(durations),
            },
            "requests_sample": sample,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/tests")
def run_tests():
    # Execute the repository's existing scheduler tests.
    root = Path(__file__).resolve().parent
    test_file = root / "tests" / "test_sched.py"

    if not test_file.exists():
        raise HTTPException(status_code=404, detail="Scheduler tests not found.")

    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout + result.stderr

        # test_sched.py prints individual checks rather than pytest summaries.
        passed = sum(
            1 for line in output.splitlines()
            if line.strip().startswith("PASS")
        )

        return {
            "executed": True,
            "status": "passed" if result.returncode == 0 else "failed",
            "passed": passed,
            "total": 13,
            "output": output or "Test process finished without console output.",
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Tests timed out.")
