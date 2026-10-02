from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.action_path_regeneration import (
    CONTRACT_AFFECTED_BLUEPRINTS,
    DEFAULT_OUT_DIR,
    run_action_path_regeneration,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Action-path based SS regeneration")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--blueprints",
        type=str,
        default=",".join(sorted(CONTRACT_AFFECTED_BLUEPRINTS)),
        help="Comma-separated blueprint ids to regenerate",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    bp_ids = {x.strip() for x in args.blueprints.split(",") if x.strip()}
    run_action_path_regeneration(
        source_jsonl=args.source,
        out_dir=args.out_dir,
        blueprint_ids=bp_ids,
        verbose=not args.quiet,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
