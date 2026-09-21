"""
train.py

End-to-end run: load f1_dataset.parquet (built by fastf1_pipeline.py),
split by race, train FastModel + SmartModel, and print the accuracy/speed
benchmark table that feeds directly into:
  - your results section (fast vs smart accuracy trade-off)
  - the Systems/OS lead's routing thresholds (how many ms each path costs)
"""
import numpy as np
import pandas as pd
import pickle

from fastf1_pipeline import build_train_test_split, TEST_RACES
from models import (
    prepare_features, FastModel, SmartModel, make_sequences,
    benchmark_speed, evaluate_pit_predictions, COMPOUND_CLASSES
)

DATA_PATH = "f1_dataset.parquet"
# TEST_RACES is imported from fastf1_pipeline.py so the train/test split
# used here always matches the one the dataset was documented with —
# edit it in one place (fastf1_pipeline.py), not both.


def main():
    df = pd.read_parquet(DATA_PATH)
    train_df, test_df = build_train_test_split(df, TEST_RACES)
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)} "
          f"| Train races: {train_df['race_id'].nunique()} | Test races: {test_df['race_id'].nunique()}")

    train_df2, X_train, y_pit_train, y_cmp_train, feature_cols = prepare_features(train_df)
    test_df2, X_test, y_pit_test, y_cmp_test, _ = prepare_features(test_df)
    # align test columns to train columns (in case a compound is missing in test split)
    test_df2 = test_df2.reindex(columns=train_df2.columns, fill_value=0)
    X_test = test_df2[feature_cols].values.astype(np.float32)

    # ---------- Fast model ----------
    print("\nTraining FastModel (logistic regression)...")
    fast = FastModel().fit(X_train, y_pit_train, y_cmp_train, feature_cols)
    pits_pred, _ = fast.predict_batch(X_test)
    fast_metrics = evaluate_pit_predictions(y_pit_test, pits_pred)
    fast_ms = benchmark_speed(fast, X_test[0])
    print("FastModel metrics:", fast_metrics)
    print(f"FastModel avg latency: {fast_ms:.4f} ms")

    # ---------- Smart model ----------
    print("\nBuilding sequences for SmartModel (GRU)...")
    groups_train = (train_df2["race_id"] + "_" + train_df2["driver"]).values
    groups_test = (test_df2["race_id"] + "_" + test_df2["driver"]).values

    X_seq_train, idx_train = make_sequences(X_train, groups_train)
    X_seq_test, idx_test = make_sequences(X_test, groups_test)
    y_pit_seq_train = y_pit_train[idx_train]
    y_pit_seq_test = y_pit_test[idx_test]

    cmp_to_idx = {c: i for i, c in enumerate(COMPOUND_CLASSES)}

    def to_cmp_idx(y_compound, y_pit_seq, idx_map):
        out = []
        for i, orig_i in enumerate(idx_map):
            if y_pit_seq[i] == 1 and pd.notna(y_compound[orig_i]):
                out.append(cmp_to_idx.get(y_compound[orig_i], -1))
            else:
                out.append(-1)
        return np.array(out)

    y_cmp_idx_train = to_cmp_idx(y_cmp_train, y_pit_seq_train, idx_train)

    print("Training SmartModel (GRU)...")
    smart = SmartModel(n_features=X_train.shape[1])  # uses default epochs (60)
    smart.fit(X_seq_train, y_pit_seq_train, y_cmp_idx_train, feature_cols)

    smart_preds = []
    for window in X_seq_test:
        smart_preds.append(smart.predict(window)["pit"])
    smart_preds = np.array(smart_preds)
    smart_metrics = evaluate_pit_predictions(y_pit_seq_test, smart_preds)
    smart_ms = benchmark_speed(smart, X_seq_test[0])
    print("SmartModel metrics:", smart_metrics)
    print(f"SmartModel avg latency: {smart_ms:.4f} ms")

    # ---------- Summary table for the paper / OS lead ----------
    print("\n=== Benchmark summary (feeds routing thresholds + results section) ===")
    print(f"{'Model':<12}{'Acc':<8}{'Prec':<8}{'Recall':<8}{'F1':<8}{'Latency(ms)':<12}")
    print(f"{'Fast':<12}{fast_metrics['accuracy']:<8.3f}{fast_metrics['precision']:<8.3f}"
          f"{fast_metrics['recall']:<8.3f}{fast_metrics['f1']:<8.3f}{fast_ms:<12.4f}")
    print(f"{'Smart':<12}{smart_metrics['accuracy']:<8.3f}{smart_metrics['precision']:<8.3f}"
          f"{smart_metrics['recall']:<8.3f}{smart_metrics['f1']:<8.3f}{smart_ms:<12.4f}")

    with open("fast_model.pkl", "wb") as f:
        pickle.dump(fast, f)

    # Save the *whole* SmartModel (scaler + feature_cols + net weights), not
    # just the raw network state_dict — predict() needs the fitted scaler
    # too, or a later reload will silently feed unscaled input into a
    # scaled-trained network and give wrong predictions.
    smart.net.to("cpu")  # portable across machines without a GPU
    smart.device = "cpu"
    with open("smart_model.pkl", "wb") as f:
        pickle.dump(smart, f)

    print("\nSaved fast_model.pkl and smart_model.pkl")


if __name__ == "__main__":
    main()