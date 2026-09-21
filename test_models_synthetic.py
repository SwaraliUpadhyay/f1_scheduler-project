"""
Sanity-checks models.py against synthetic data shaped like fastf1_pipeline.py's
output, so bugs surface before you burn time on a real FastF1 pull.
"""
import numpy as np
import pandas as pd
from models import (
    prepare_features, FastModel, SmartModel, make_sequences,
    benchmark_speed, evaluate_pit_predictions, COMPOUND_CLASSES
)

np.random.seed(0)
N_RACES = 3
LAPS_PER_RACE = 50
DRIVERS = ["VER", "HAM"]

rows = []
for race in range(N_RACES):
    for driver in DRIVERS:
        for lap in range(1, LAPS_PER_RACE + 1):
            rows.append({
                "race_id": f"race_{race}",
                "driver": driver,
                "lap_number": lap,
                "tyre_life": lap % 20,
                "compound": np.random.choice(["SOFT", "MEDIUM", "HARD"]),
                "stint": lap // 20 + 1,
                "gap_ahead": np.random.uniform(0.5, 5),
                "gap_behind": np.random.uniform(0.5, 5),
                "gap_ahead_delta": np.random.uniform(-0.5, 0.5),
                "air_temp": 28 + np.random.randn(),
                "track_temp": 40 + np.random.randn(),
                "rainfall": False,
                "track_status": "1",
                "laps_since_field_pit": np.random.randint(0, 5),
                "pit_next_3": np.random.choice([0, 1], p=[0.85, 0.15]),
                "compound_next": np.random.choice(COMPOUND_CLASSES[:3]),
            })
df = pd.DataFrame(rows)

print("=== prepare_features ===")
df2, X, y_pit, y_compound, feature_cols = prepare_features(df)
print("X shape:", X.shape, "| feature_cols:", feature_cols)

print("\n=== FastModel ===")
fast = FastModel().fit(X, y_pit, y_compound, feature_cols)
sample = X[0]
pred = fast.predict(sample)
print("Single prediction:", pred)
ms = benchmark_speed(fast, sample)
print(f"Fast model avg inference: {ms:.4f} ms")

pits_pred, _ = fast.predict_batch(X)
print("FastModel eval:", evaluate_pit_predictions(y_pit, pits_pred))

print("\n=== SmartModel (GRU) ===")
groups = (df2["race_id"] + "_" + df2["driver"]).values
X_seq, idx_map = make_sequences(X, groups, window=8)
y_pit_seq = y_pit[idx_map]

# map compound strings to indices, -1 where pit==0
cmp_to_idx = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
y_cmp_idx = np.array([
    cmp_to_idx.get(c, -1) if y_pit_seq[i] == 1 else -1
    for i, c in enumerate(y_compound[idx_map])
])

smart = SmartModel(n_features=X.shape[1], epochs=5)  # few epochs, just checking it runs
smart.fit(X_seq, y_pit_seq, y_cmp_idx, feature_cols)

sample_window = X_seq[0]
pred = smart.predict(sample_window)
print("Single GRU prediction:", pred)
ms = benchmark_speed(smart, sample_window)
print(f"Smart model avg inference: {ms:.4f} ms")

print("\nAll checks passed.")
