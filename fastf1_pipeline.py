"""
fastf1_pipeline.py

Pulls race data via FastF1, builds a per-(driver, lap) feature table with
Option-A "imitation" labels (predict what the team actually did, not what
was theoretically optimal), and produces a race-level train/test split.

Output schema (one row per driver per lap):
    race_id, driver, lap_number, tyre_life, compound, stint,
    gap_ahead, gap_behind, gap_ahead_delta,
    air_temp, track_temp, rainfall, track_status,
    laps_since_field_pit,
    pit_next_3          <- label 1: will this car pit within the next 3 laps
    compound_next        <- label 2: compound fitted at that pit stop (NaN if no pit)

Usage:
    from fastf1_pipeline import build_dataset, RACE_CALENDAR
    df = build_dataset(RACE_CALENDAR)
    df.to_parquet("f1_dataset.parquet")
"""

import fastf1
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from typing import List

os.makedirs("fastf1_cache", exist_ok=True)  # FastF1 requires this dir to exist first
fastf1.Cache.enable_cache("fastf1_cache")  # avoid re-downloading on every run

PIT_HORIZON = 3         # "pit_next_3": will car pit within next N laps
FIELD_PIT_WINDOW = 5    # laps to look back for "how many rivals just pitted"


@dataclass
class RaceRef:
    year: int
    gp: str          # e.g. "Bahrain", "Monza" — matches FastF1 event names
    session: str = "R"  # Race session


# Seasons to pull. Currently set to 2022 only to fit a tighter timeline —
# 2022 alone is ~22-23 races, still plenty for a working baseline and a
# real train/test split. Add 2023/2024 back to this list later if time
# allows; nothing else in the pipeline needs to change to do that.
SEASONS: List[int] = [2022]

# Races held out entirely for testing — never trained on. Chosen for a mix
# of circuit types (street/technical/high-speed) plus late-season races
# (tests generalization to "later" races, mirroring real deployment).
# Edit this list once you've pulled the full schedule below and can see
# exact event names — FastF1 event names sometimes differ slightly from
# what you'd guess (e.g. "São Paulo" vs "Brazil").
TEST_RACES: List[str] = [
    "2022_Monaco_Grand_Prix", "2022_Italian_Grand_Prix",
    "2022_Singapore_Grand_Prix", "2022_Abu_Dhabi_Grand_Prix",
]


def get_full_calendar(seasons: List[int]) -> List[RaceRef]:
    """
    Pulls every race weekend FastF1 has for the given seasons (excludes
    pre-season testing events, which aren't real races and have no
    strategy signal).
    """
    races = []
    for year in seasons:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            races.append(RaceRef(year, event["EventName"]))
    return races


# Populated by calling get_full_calendar() in __main__ / when you're ready.
# Left as an empty placeholder here so importing this module doesn't hit
# the network.
RACE_CALENDAR: List[RaceRef] = []


def _load_session(race: RaceRef):
    session = fastf1.get_session(race.year, race.gp, race.session)
    session.load(telemetry=False, weather=True, laps=True)
    return session


def _merge_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach nearest-in-time weather reading to each lap."""
    weather = weather.sort_values("Time")
    laps = laps.sort_values("Time")
    merged = pd.merge_asof(
        laps, weather[["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall"]],
        on="Time", direction="nearest"
    )
    return merged


def _compute_gaps(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate gap-ahead/gap-behind in seconds using cumulative race time
    per driver at each lap. This is an approximation (FastF1 doesn't give a
    clean per-lap 'gap to car ahead' field directly) — good enough for
    strategy-relevant features, note this simplification in your report.
    """
    laps = laps.copy()
    laps["CumRaceTime"] = laps.groupby("Driver")["LapTime"].cumsum()

    gap_ahead, gap_behind = [], []
    for lap_num, group in laps.groupby("LapNumber"):
        g = group.sort_values("Position")
        cum_times = g["CumRaceTime"].values
        idx_by_driver = {d: i for i, d in enumerate(g["Driver"].values)}
        for i, drv in enumerate(g["Driver"].values):
            ahead = cum_times[i] - cum_times[i - 1] if i > 0 else np.nan
            behind = cum_times[i + 1] - cum_times[i] if i < len(cum_times) - 1 else np.nan
            gap_ahead.append((lap_num, drv, ahead))
            gap_behind.append((lap_num, drv, behind))

    gap_ahead_df = pd.DataFrame(gap_ahead, columns=["LapNumber", "Driver", "gap_ahead"])
    gap_behind_df = pd.DataFrame(gap_behind, columns=["LapNumber", "Driver", "gap_behind"])
    laps = laps.merge(gap_ahead_df, on=["LapNumber", "Driver"])
    laps = laps.merge(gap_behind_df, on=["LapNumber", "Driver"])
    return laps


def _build_labels(laps: pd.DataFrame) -> pd.DataFrame:
    """Option A imitation labels: what the team actually did."""
    laps = laps.sort_values(["Driver", "LapNumber"]).copy()

    # A pit happened on a lap if PitInTime is not null on that lap row.
    laps["pit_this_lap"] = laps["PitInTime"].notna().astype(int)

    def label_group(g):
        g = g.sort_values("LapNumber").reset_index(drop=True)
        pit_next = np.zeros(len(g), dtype=int)
        compound_next = [np.nan] * len(g)
        for i in range(len(g)):
            window = g.iloc[i + 1: i + 1 + PIT_HORIZON]
            pit_rows = window[window["pit_this_lap"] == 1]
            if len(pit_rows) > 0:
                pit_next[i] = 1
                compound_next[i] = pit_rows.iloc[0]["Compound"]
        g["pit_next_3"] = pit_next
        g["compound_next"] = compound_next
        return g

    laps = laps.groupby("Driver", group_keys=False)[laps.columns].apply(label_group)
    return laps


def _field_pit_pressure(laps: pd.DataFrame) -> pd.DataFrame:
    """How many cars pitted in the last FIELD_PIT_WINDOW laps (undercut/overcut signal)."""
    laps = laps.sort_values("LapNumber").copy()
    pit_counts = laps.groupby("LapNumber")["pit_this_lap"].sum()
    rolling = pit_counts.rolling(FIELD_PIT_WINDOW, min_periods=1).sum()
    laps["laps_since_field_pit"] = laps["LapNumber"].map(rolling)
    return laps


def build_dataset(races: List[RaceRef]) -> pd.DataFrame:
    all_rows = []
    skipped = []
    for race in races:
        race_id = f"{race.year}_{race.gp}".replace(" ", "_")
        print(f"Loading {race_id} ...")
        try:
            session = _load_session(race)

            laps = session.laps.copy()
            laps["LapTime"] = laps["LapTime"].dt.total_seconds()
            laps = laps[laps["LapTime"].notna()]  # drop in/out laps w/o valid time if needed later

            if len(laps) == 0:
                raise ValueError("no valid laps after filtering")

            laps = _merge_weather(laps, session.weather_data)
            laps = _compute_gaps(laps)
            laps = _build_labels(laps)
            laps = _field_pit_pressure(laps)

            laps["gap_ahead_delta"] = laps.groupby("Driver")["gap_ahead"].diff().fillna(0)
            laps["race_id"] = race_id

            keep_cols = [
                "race_id", "Driver", "LapNumber", "TyreLife", "Compound", "Stint",
                "gap_ahead", "gap_behind", "gap_ahead_delta",
                "AirTemp", "TrackTemp", "Rainfall", "TrackStatus",
                "laps_since_field_pit", "pit_next_3", "compound_next",
            ]
            all_rows.append(laps[keep_cols])

        except Exception as e:
            # Some historical sessions have incomplete/corrupt upstream timing
            # data (a known FastF1 quirk, not a bug in this pipeline) — skip
            # and keep going rather than losing the whole multi-race pull.
            print(f"  SKIPPED {race_id}: {type(e).__name__}: {e}")
            skipped.append(race_id)
            continue

    if len(all_rows) == 0:
        raise RuntimeError(
            "Every race failed to load — check your internet connection and "
            "FastF1 installation before re-running."
        )

    if skipped:
        print(f"\n{len(skipped)} race(s) skipped due to upstream data issues: {skipped}")
        print("This is expected occasionally — some historical FastF1 sessions have "
              "incomplete timing feeds. Note the count in your report's data section.")

    df = pd.concat(all_rows, ignore_index=True)
    df = df.rename(columns={
        "Driver": "driver", "LapNumber": "lap_number", "TyreLife": "tyre_life",
        "Compound": "compound", "Stint": "stint", "AirTemp": "air_temp",
        "TrackTemp": "track_temp", "Rainfall": "rainfall", "TrackStatus": "track_status",
    })
    return df


def build_train_test_split(df: pd.DataFrame, test_races: List[str]):
    """
    Split by race_id, NOT by row — random row-level splits leak
    same-race weather/track-evolution context between train and test.
    """
    test_df = df[df["race_id"].isin(test_races)].copy()
    train_df = df[~df["race_id"].isin(test_races)].copy()
    return train_df, test_df


if __name__ == "__main__":
    calendar = get_full_calendar(SEASONS)
    print(f"Found {len(calendar)} race weekends across seasons {SEASONS}")

    df = build_dataset(calendar)
    df.to_parquet("f1_dataset.parquet")

    print(df.head())
    print(f"\nTotal rows: {len(df)}  |  pit_next_3 positive rate: {df['pit_next_3'].mean():.3f}")
    print(f"Distinct race_ids: {sorted(df['race_id'].unique())}")
    print("\n--> Check the race_id list above against TEST_RACES at the top of this "
          "file — race_id format is '<year>_<EventName with spaces as underscores>'. "
          "Update TEST_RACES if any names don't match exactly.")