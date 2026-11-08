"""DDL: drop and (re)create the collection - SCHEMA PART + INDEX PART.

The two parts are defined in src/wvbench/schema.py (marked "=== SCHEMA PART ===" and
"=== INDEX PART ==="). Weaviate needs the index definition when the collection is
created, so both parts are applied by one create call; see README "Database design".

    docker compose run --rm benchmark python scripts/ddl.py glove-200-cosine
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

from wvbench import cliargs, pipeline, schema  # noqa: E402


def run() -> None:
    args = cliargs.parse(cliargs.parser(__doc__))
    r = cliargs.resolved(args)
    print("=== SCHEMA PART ===")
    print(json.dumps(schema.schema_definition(r), indent=2))
    print("=== INDEX PART ===")
    print(json.dumps(schema.index_definition(r), indent=2))
    with pipeline.Run(r) as run_:
        pipeline.step_schema(run_)


if __name__ == "__main__":
    cliargs.main(run)
