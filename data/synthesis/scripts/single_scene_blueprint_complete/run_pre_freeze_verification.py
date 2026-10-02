from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR
from smarthome_mdf.single_scene_blueprint_complete.p1_integrity_audit import V2_DIR_NAME
from smarthome_mdf.single_scene_blueprint_complete.pre_freeze_verification import run_pre_freeze_verification

def main() -> int:
    report = run_pre_freeze_verification(v2_dir=Path(OUTPUT_DIR / V2_DIR_NAME))
    print(f"FINAL VERDICT: {report['FINAL_VERDICT']}")
    h1 = report["HARD_RESULT_1_POST_SANITIZE"]
    h2 = report["HARD_RESULT_2_TEMPORAL"]
    h3 = report["HARD_RESULT_3_CAPACITY"]
    print(f"HARD1 sanitize_retained={h1['sanitize_retained']} MATCH_OK={h1['MATCH_OK']}/{h1['sanitize_retained']} FAIL={h1['MATCH_FAIL']}")
    print(f"HARD2 temporal_near_dup={h2['temporal_near_dup']} only_ts={h2['only_timestamp_shift']} effective_unique={h2['effective_unique_samples_without_timestamp_only_duplicates']}")
    print(f"HARD3 inst_covered={h3['full_instance_capacity_covered']} bind_covered={h3['full_binding_capacity_covered']}")
    if report["blockers"]:
        print("BLOCKERS:", report["blockers"])
    return 0 if report["FINAL_VERDICT"] == "READY_FOR_SINGLE_SCENE_FREEZE" else 1

if __name__ == "__main__":
    raise SystemExit(main())
