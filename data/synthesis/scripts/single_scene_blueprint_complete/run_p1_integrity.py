from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR
from smarthome_mdf.single_scene_blueprint_complete.p1_coordinator import run_p1_repair
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import V1_DIR_NAME, V2_DIR_NAME

def main() -> int:
    parser = argparse.ArgumentParser(description="P1 integrity audit + targeted repair")
    parser.add_argument("--v1", type=str, default=str(OUTPUT_DIR / V1_DIR_NAME))
    parser.add_argument("--v2", type=str, default=str(OUTPUT_DIR / V2_DIR_NAME))
    parser.add_argument("--audit-only", action="store_true", help="Audit v1 only, no v2 repair")
    args = parser.parse_args()

    if args.audit_only:
        from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import (
            run_full_integrity_audit,
            write_audit_artifacts,
            VERDICT_BLOCKED,
        )
        from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl
        import json

        v1 = Path(args.v1)
        samples = read_jsonl(v1 / "single_scene_samples.jsonl")
        audit = run_full_integrity_audit(samples)
        write_audit_artifacts(v1, audit)
        print(f"Audit complete: invalid={audit['invalid_count']} verdict={VERDICT_BLOCKED}")
        return 0

    manifest = run_p1_repair(v1_dir=Path(args.v1), v2_dir=Path(args.v2))
    return 0 if manifest.get("verdict") == "READY_FOR_SINGLE_SCENE_FREEZE" else 1

if __name__ == "__main__":
    raise SystemExit(main())
