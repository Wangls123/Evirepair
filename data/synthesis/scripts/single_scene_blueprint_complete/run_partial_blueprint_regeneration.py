from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.partial_blueprint_regeneration import (
    DEFAULT_OUT_DIR,
    DEFAULT_SOURCE_JSONL,
    run_partial_blueprint_regeneration,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Partial SS repair for four defective blueprints")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_partial_blueprint_regeneration(
        source_jsonl=args.source,
        out_dir=args.out_dir,
        verbose=not args.quiet,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
