"""Full population in the baseline order: schema -> load -> index -> check, then build.csv.

    docker compose run --rm setup glove-200-cosine
    docker compose run --rm setup glove-200-cosine --param ingest.mode=server_side
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wvbench import cliargs, pipeline  # noqa: E402


def run() -> None:
    args = cliargs.parse(cliargs.parser(__doc__))
    with pipeline.Run(cliargs.resolved(args)) as run_:
        pipeline.step_schema(run_)
        pipeline.step_load(run_)
        pipeline.step_index(run_)
        pipeline.step_check(run_)
        pipeline.finalize(run_)


if __name__ == "__main__":
    cliargs.main(run)
