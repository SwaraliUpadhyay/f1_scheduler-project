"""
predict_service.py

Exposes both trained models behind a stable, simple interface for the
Systems/OS lead's router — no need to know about FastModel/SmartModel
internals, scalers, feature encoding, or windowing.

TWO WAYS TO USE THIS:

1. Direct Python import (fastest, if the router is also Python):

    from predict_service import predict_fast, predict_smart

    result = predict_fast({
        "tyre_life": 12, "compound": "MEDIUM", "gap_ahead": 1.8,
        "gap_behind": 2.4, "gap_ahead_delta": -0.1, "air_temp": 29.0,
        "track_temp": 41.0, "rainfall": False, "laps_since_field_pit": 2,
    })
    # -> {"pit": 0, "pit_prob": 0.12, "compound": None}

    # predict_smart needs the last WINDOW_SIZE laps as a list of dicts,
    # oldest first, same keys as above:
    result = predict_smart(list_of_last_8_lap_dicts)

2. HTTP API (if the router is in another language, or runs as a separate
   service/container):

    uvicorn predict_service:app --host 0.0.0.0 --port 8000

    POST /predict/fast   body: single lap dict (as above)
    POST /predict/smart  body: {"laps": [lap_dict, lap_dict, ...]}  (last N laps)
    GET  /health          -> {"status": "ok"}
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from models import prepare_features, WINDOW_SIZE, NUMERIC_FEATURES, COMPOUND_CLASSES

FAST_MODEL_PATH = "fast_model.pkl"
SMART_MODEL_PATH = "smart_model.pkl"

_fast_model = None
_smart_model = None


def load_models(fast_path: str = FAST_MODEL_PATH, smart_path: str = SMART_MODEL_PATH):
    """Loads both trained models into memory once. Called lazily on first use,
    or call explicitly at process startup to fail fast if files are missing."""
    global _fast_model, _smart_model
    if _fast_model is None:
        with open(fast_path, "rb") as f:
            _fast_model = pickle.load(f)
    if _smart_model is None:
        with open(smart_path, "rb") as f:
            _smart_model = pickle.load(f)
    return _fast_model, _smart_model


def _lap_dict_to_row(lap: Dict, feature_cols: List[str]) -> np.ndarray:
    """
    Converts a single lap dict (raw, human-readable keys) into the exact
    numeric feature vector the models were trained on — same one-hot
    compound encoding, same column order, missing values filled with 0.
    """
    df = pd.DataFrame([lap])
    if "compound" not in df.columns:
        df["compound"] = "MEDIUM"  # harmless default if caller omits it
    df["rainfall"] = df.get("rainfall", False)
    df["rainfall"] = df["rainfall"].astype(float)
    df = pd.get_dummies(df, columns=["compound"], prefix="cmp")

    row = pd.DataFrame(columns=feature_cols)
    row = pd.concat([row, df], ignore_index=True)
    row = row.reindex(columns=feature_cols, fill_value=0)
    row[feature_cols] = row[feature_cols].fillna(0)
    return row[feature_cols].values.astype(np.float32)[0]


def predict_fast(lap: Dict) -> Dict:
    """
    lap: a single dict with keys tyre_life, compound, gap_ahead, gap_behind,
    gap_ahead_delta, air_temp, track_temp, rainfall, laps_since_field_pit.
    Returns {"pit": 0/1, "pit_prob": float, "compound": str or None}.
    """
    fast_model, _ = load_models()
    x_row = _lap_dict_to_row(lap, fast_model.feature_cols)
    return fast_model.predict(x_row)


def predict_smart(laps: List[Dict]) -> Dict:
    """
    laps: the last up-to-WINDOW_SIZE lap dicts for one driver, OLDEST FIRST,
    same keys as predict_fast. Fewer than WINDOW_SIZE is fine early in a
    race (auto zero-padded, matching how the model was trained) — pass
    whatever history you have so far.
    Returns {"pit": 0/1, "pit_prob": float, "compound": str or None}.
    """
    _, smart_model = load_models()
    rows = [_lap_dict_to_row(lap, smart_model.feature_cols) for lap in laps]
    window = np.stack(rows)  # (n_laps, n_features)

    n, f = window.shape
    if n < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - n, f), dtype=np.float32)
        window = np.vstack([pad, window])
    elif n > WINDOW_SIZE:
        window = window[-WINDOW_SIZE:]

    return smart_model.predict(window)


# ---------------------------------------------------------------------------
# Optional HTTP API — only used if the router isn't Python or runs separately.
# Requires: pip install fastapi uvicorn
# ---------------------------------------------------------------------------
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="F1 Strategy Prediction Service")

    class LapFeatures(BaseModel):
        tyre_life: float
        compound: str = "MEDIUM"
        gap_ahead: Optional[float] = None
        gap_behind: Optional[float] = None
        gap_ahead_delta: float = 0.0
        air_temp: float = 25.0
        track_temp: float = 35.0
        rainfall: bool = False
        laps_since_field_pit: int = 0

    class SmartRequest(BaseModel):
        laps: List[LapFeatures]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict/fast")
    def http_predict_fast(lap: LapFeatures):
        try:
            return predict_fast(lap.dict())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/smart")
    def http_predict_smart(req: SmartRequest):
        try:
            return predict_smart([l.dict() for l in req.laps])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

except ImportError:
    app = None  # fastapi/pydantic not installed — direct Python import still works fine


if __name__ == "__main__":
    # Quick smoke test using load_models() to fail fast if model files are missing.
    load_models()
    sample_lap = {
        "tyre_life": 15, "compound": "MEDIUM", "gap_ahead": 1.2, "gap_behind": 3.0,
        "gap_ahead_delta": -0.2, "air_temp": 30.0, "track_temp": 42.0,
        "rainfall": False, "laps_since_field_pit": 1,
    }
    print("predict_fast:", predict_fast(sample_lap))
    print("predict_smart (1 lap, auto-padded):", predict_smart([sample_lap]))
    print("predict_smart (8 laps):", predict_smart([sample_lap] * 8))
