"""Shared command-line arguments and error handling for scripts/*.py."""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Callable

from wvbench import settings
from wvbench.resolve import Resolved, resolve
from wvbench.settings import ConfigError


def parser(description: str, dataset_required: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("dataset", nargs=None if dataset_required else "?", default=settings.default_dataset(),
                   help="dataset name, e.g. glove-200-cosine (default: $WVB_DATASET)")
    p.add_argument("--profile", help="use this profile instead of the automatic choice")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="override a setting, e.g. max_connections=32 or ingest.mode=server_side (repeatable)")
    p.add_argument("--allow-oversize", action="store_true",
                   help="run even if the memory estimate exceeds the budget")
    return p


def parse(p: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    # Allow `script.py --param x=1 glove` as well as `script.py glove --param x=1`.
    args = p.parse_intermixed_args(argv)
    if not args.dataset:
        p.error("dataset is required (argument or $WVB_DATASET)")
    return args


def resolved(args: argparse.Namespace, require_file: bool = True) -> Resolved:
    return resolve(args.dataset, args.profile, args.param, args.allow_oversize, require_file=require_file)


def main(fn: Callable[[], None]) -> None:
    """Run fn; configuration errors exit with 2, everything else with 1."""
    try:
        fn()
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        traceback.print_exc()
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
