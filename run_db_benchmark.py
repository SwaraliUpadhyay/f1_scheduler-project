"""
run_db_benchmark.py  —  the DB evidence run (DB role 5)

    python run_db_benchmark.py --skip-cassandra          # SQLite only, no cluster
    python run_db_benchmark.py --races 2022_Monaco_Grand_Prix

Writes results/db_benchmark.json.
"""
from data.bench_db import main

if __name__ == "__main__":
    main()
