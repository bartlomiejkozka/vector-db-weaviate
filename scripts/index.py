"""Index step: wait until the vector index is built; record build time and index size.

    docker compose run --rm benchmark python scripts/index.py glove-200-cosine
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvbench import cliargs, pipeline  # noqa: E402


def run() -> None:
    args = cliargs.parse(cliargs.parser(__doc__))
    with pipeline.Run(cliargs.resolved(args)) as run_:
        pipeline.step_index(run_)


if __name__ == "__main__":
    cliargs.main(run)
