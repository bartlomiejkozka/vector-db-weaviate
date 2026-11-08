"""Search benchmark (milestone 2): warm-up, 3 passes, recall and p50/p95/p99 per sweep setting -> search.csv.

Not implemented yet. The sweep parameter and values are already configured per
profile (sweep_param / sweep_values in config/profiles.toml).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


if __name__ == "__main__":
    print("benchmark: not implemented yet (milestone 2)", file=sys.stderr)
    sys.exit(2)
