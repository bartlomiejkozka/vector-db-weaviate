"""Load step: insert all train vectors (chunks of ~50,000 rows). Requires the DDL step.

    docker compose run --rm benchmark python scripts/load.py glove-200-cosine
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvbench import cliargs, pipeline  # noqa: E402


def run() -> None:
    args = cliargs.parse(cliargs.parser(__doc__))
    with pipeline.Run(cliargs.resolved(args)) as run_:
        pipeline.step_load(run_)


if __name__ == "__main__":
    cliargs.main(run)
