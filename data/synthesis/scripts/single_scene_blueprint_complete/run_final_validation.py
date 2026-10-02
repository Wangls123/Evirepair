from __future__ import annotations

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR

from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import V3_CLEAN_OUTPUT_DIR_NAME

from smarthome_mdf.single_scene_blueprint_complete.final_validation import run_final_validation

def main() -> int:

    report = run_final_validation(OUTPUT_DIR / V3_CLEAN_OUTPUT_DIR_NAME)

    print(f"FINAL VERDICT: {report['FINAL_VERDICT']}")

    print(f"Samples: {report['file_integrity']['FINAL_SAMPLE_COUNT']}")

    rm = report["runtime_rematch"]

    print(f"Runtime match: {rm['POST_GENERATION_RUNTIME_MATCH']}/{rm['total']} FAIL={rm['MATCH_FAIL']}")

    dup = report["duplicates"]

    print(

        f"Duplicates: exact={dup['EXACT_DUPLICATE']} canonical={dup['CANONICAL_DUPLICATE']} "

        f"padding={dup['SAME_SOURCE_DUPLICATE_PADDING']} ts_only={dup['TIMESTAMP_ONLY_DUPLICATE']}"

    )

    if report["blockers"]:

        print("BLOCKERS:")

        for b in report["blockers"]:

            print(f"  - {b}")

    return 0 if report["FINAL_VERDICT"] == "READY_FOR_SINGLE_SCENE_FREEZE" else 1

if __name__ == "__main__":

    raise SystemExit(main())
