"""Download step: fetch <dataset>.hdf5 into /data and print its size and sha256.

    docker compose run --rm download glove-200-cosine
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from wvbench import settings  # noqa: E402
from wvbench.cliargs import main  # noqa: E402
from wvbench.download import download  # noqa: E402


def run() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", nargs="?", default=settings.default_dataset())
    args = p.parse_args()
    if not args.dataset:
        p.error("dataset is required (argument or $WVB_DATASET)")
    download(args.dataset)


if __name__ == "__main__":
    main(run)
