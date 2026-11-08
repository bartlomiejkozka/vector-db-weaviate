"""db <goal> <params...>

Goals:
  1  KNN   db 1 <query_id> <k>   k nearest neighbours of test vector <query_id>

Output: one result per line as `id distance`, nearest first. Errors go to
stderr with a non-zero exit code. The dataset comes from --dataset or $WVB_DATASET.
"""

from __future__ import annotations

import argparse
import sys

from wvbench import settings
from wvbench.cliargs import main as run_main
from wvbench.goals import knn

GOALS = {
    1: ("KNN", knn.USAGE, knn.run),
}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="db", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("goal", type=int, help="goal number")
    p.add_argument("params", nargs="*", help="goal parameters")
    p.add_argument("--dataset", default=settings.default_dataset(), help="dataset name (default: $WVB_DATASET)")
    args = p.parse_intermixed_args(argv)

    if args.goal not in GOALS:
        known = "; ".join(f"{n}: {name} ({usage})" for n, (name, usage, _) in GOALS.items())
        print(f"error: unknown goal {args.goal}. Known goals: {known}", file=sys.stderr)
        sys.exit(2)
    if not args.dataset:
        print("error: no dataset; pass --dataset or set WVB_DATASET", file=sys.stderr)
        sys.exit(2)

    _, _, fn = GOALS[args.goal]

    def go() -> None:
        for rid, dist in fn(args.params, args.dataset):
            print(f"{rid} {dist:.6f}")

    run_main(go)


if __name__ == "__main__":
    main()
