from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.action_path_rebalance import (
    DEFAULT_OUT_DIR,
    run_action_path_rebalance,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Path-quota rebalance for SS corpus")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_OUT_DIR / "single_scene_samples.jsonl",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skew-threshold", type=float, default=2.0, help="Rebalance if max/min path count exceeds this")
    parser.add_argument("--strict", action="store_true", help="Fail if any path deficit cannot be filled")
    parser.add_argument("--max-attempts-multiplier", type=int, default=5, help="Attempt budget = deficit_slots * multiplier (capped)")
    parser.add_argument("--max-attempts-cap", type=int, default=3000, help="Hard cap on synthesis attempts per blueprint")
    parser.add_argument("--progress-interval", type=int, default=100, help="Log progress every N attempts (0=off)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable per-blueprint checkpoint writes")
    parser.add_argument("--blueprints", type=str, default="", help="Optional comma-separated blueprint filter")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    bp_ids = {x.strip() for x in args.blueprints.split(",") if x.strip()} or None
    run_action_path_rebalance(
        source_jsonl=args.source,
        out_dir=args.out_dir,
        blueprint_ids=bp_ids,
        skew_threshold=args.skew_threshold,
        allow_partial=not args.strict,
        max_attempts_multiplier=args.max_attempts_multiplier,
        max_attempts_cap=args.max_attempts_cap,
        progress_interval=args.progress_interval,
        checkpoint_each_blueprint=not args.no_checkpoint,
        verbose=not args.quiet,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
