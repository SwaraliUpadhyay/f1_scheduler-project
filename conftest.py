"""Makes the repo root importable so `python -m pytest tests/` works
without installing the project as a package."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
