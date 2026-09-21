"""
columns.py

The lap column list, in ONE place. Both the Cassandra client and the
SQLite baseline import it, which is what guarantees the benchmark
compares identical rows — if the two stacks drifted to different column
sets the comparison would be meaningless and nothing would error.
"""

LAP_COLUMNS = [
    "race_id", "driver", "lap_number", "tyre_life", "compound", "stint",
    "gap_ahead", "gap_behind", "gap_ahead_delta", "air_temp", "track_temp",
    "rainfall", "track_status", "is_caution", "field_pits_last_5",
]
