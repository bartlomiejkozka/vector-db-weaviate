"""Print the resolved configuration of a dataset (profile, parameters, memory estimate).

Does not touch the database; works before the file is downloaded (uses config/datasets.toml).

    python scripts/show_config.py arxiv-nomic-768-normalized
    python scripts/show_config.py hotpotqa-harrier-640-normalized --param oversize.policy=fallback
    python scripts/show_config.py --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from wvbench import cliargs, dataset as ds, schema  # noqa: E402
from wvbench.resolve import OversizeError  # noqa: E402


def show(args, name: str) -> bool:
    args.dataset = name
    try:
        r = cliargs.resolved(args, require_file=False)
    except OversizeError as e:
        print(f"{name}: REJECTED\n  {e}\n")
        return False
    print(f"{name}  ({r.info.rows:,} x {r.info.dim}, {r.info.distance} -> {r.info.metric}, source: {r.info.source})")
    print(f"  collection: {r.info.collection}   (baseline table {r.info.table})")
    print(f"  profile:    {r.settings.profile}  [{r.settings.profile_reason}]")
    print(f"  estimate:   {r.estimate.summary()}")
    for note in r.notes:
        print(f"  note:       {note}")
    print(f"  index:      {json.dumps(schema.index_definition(r))}")
    for sec in ("schema", "ingest", "timing", "metrics"):
        print(f"  {sec + ':':<11} {json.dumps(r.section(sec))}")
    if args.json:
        print(json.dumps(r.as_dict(), indent=2, default=str))
    print()
    return True


def run() -> None:
    p = cliargs.parser(__doc__, dataset_required=False)
    p.add_argument("--all", action="store_true", help="every dataset in config/datasets.toml")
    p.add_argument("--json", action="store_true", help="also print the full resolved configuration")
    args = p.parse_intermixed_args()
    names = list(ds.registry()["datasets"]) if args.all else [args.dataset]
    if names == [None]:
        p.error("dataset or --all is required")
    ok = [show(args, n) for n in names]
    if not args.all and not all(ok):
        sys.exit(3)


if __name__ == "__main__":
    cliargs.main(run)
