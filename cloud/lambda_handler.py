"""
cloud/lambda_handler.py  —  DB role 4

Deploys the smart (GRU) model as a cloud function so the team measures
REAL network latency instead of a constant you invented.

Works unmodified on both targets:
  * AWS Lambda   -> handler = lambda_handler
  * Cloud Run    -> the Flask app at the bottom, via the Dockerfile

WHY THIS MATTERS FOR THE OS EXPERIMENT
config.SMART_NETWORK_MS is currently 0.0. Until this is deployed and that
constant is replaced with a measured round-trip, the fast-vs-smart
latency gap in your scheduling results is understated by roughly two
orders of magnitude, and the router's allocation decision looks far less
consequential than it is. Deploy this, measure the RTT, then re-run
run_experiment.py.

Cold starts: Lambda will cold-start on the first call and load a ~70KB
pickle plus torch. Measure warm and cold separately and report both —
a cold start is not noise, it is a real property of the deployment and
it is precisely the kind of tail latency a race-strategy system cares
about.
"""

import json
import os
import time

MODEL_PATH = os.environ.get("SMART_MODEL_PATH", "/opt/ml/smart_model.pkl")
_model = None


def _load():
    global _model
    if _model is None:
        import pickle
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _predict(laps):
    """laps: list of lap dicts, oldest first."""
    import numpy as np
    from predict_service import _lap_dict_to_row
    from models import WINDOW_SIZE

    model = _load()
    rows = [_lap_dict_to_row(lap, model.feature_cols) for lap in laps]
    window = np.stack(rows)
    n, f = window.shape
    if n < WINDOW_SIZE:
        window = np.vstack([np.zeros((WINDOW_SIZE - n, f), dtype=np.float32), window])
    elif n > WINDOW_SIZE:
        window = window[-WINDOW_SIZE:]
    return model.predict(window)


def lambda_handler(event, context):
    t0 = time.perf_counter()
    try:
        body = event.get("body")
        payload = json.loads(body) if isinstance(body, str) else (body or event)
        laps = payload["laps"]
        result = _predict(laps)
        # server_ms lets the client subtract compute from RTT and isolate
        # the network component — which is the number config.py needs.
        result["server_ms"] = (time.perf_counter() - t0) * 1000.0
        return {"statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result, default=str)}
    except Exception as e:
        return {"statusCode": 500,
                "body": json.dumps({"error": f"{type(e).__name__}: {e}"})}


# --- Cloud Run / local container entrypoint --------------------------
try:
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/predict/smart")
    def predict_smart_http():
        t0 = time.perf_counter()
        result = _predict(request.get_json()["laps"])
        result["server_ms"] = (time.perf_counter() - t0) * 1000.0
        return jsonify(result)

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
except ImportError:
    app = None
