from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.multi_action_frozen_v1.b0_output_merge import run_ss_repair_b0_merge
from smarthome_mdf.multi_action_frozen_v1.config import (
    FORMAL_B0_SS_REPAIR_DIR,
    MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES,
    SS_REPAIR_SAMPLES,
)

def main() -> int:
    parser = argparse.ArgumentParser(description="Merge formal B0 into synthesis samples (b0_output field)")
    parser.add_argument("--ss-samples", type=Path, default=SS_REPAIR_SAMPLES)
    parser.add_argument("--ss-b0", type=Path, default=FORMAL_B0_SS_REPAIR_DIR / "single_scene_b0_outputs.jsonl")
    parser.add_argument("--ma-samples", type=Path, default=MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES)
    parser.add_argument("--ma-b0", type=Path, default=FORMAL_B0_SS_REPAIR_DIR / "multi_action_b0_outputs.jsonl")
    parser.add_argument("--out-ss", type=Path, default=None, help="Default: overwrite --ss-samples")
    parser.add_argument("--out-ma", type=Path, default=None, help="Default: overwrite --ma-samples")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run_ss_repair_b0_merge(
        ss_samples=args.ss_samples,
        ss_b0=args.ss_b0,
        ma_samples=args.ma_samples,
        ma_b0=args.ma_b0,
        out_ss=args.out_ss,
        out_ma=args.out_ma,
        backup=not args.no_backup,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
