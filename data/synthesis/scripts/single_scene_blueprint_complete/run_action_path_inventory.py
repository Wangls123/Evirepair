from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    build_blueprint_action_inventory,
    write_action_path_artifacts,
)
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR

def main() -> int:
    parser = argparse.ArgumentParser(description="Build executable action path inventory")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--samples", type=Path, default=None, help="Optional samples jsonl for coverage audit")
    parser.add_argument("--print-summary", action="store_true")
    args = parser.parse_args()

    paths = write_action_path_artifacts(args.output, samples_jsonl=args.samples)
    if args.print_summary:
        inv = build_blueprint_action_inventory()
        for bp in inv["blueprints"]:
            print(f"{bp['blueprint_id']}: {bp['executable_action_count']} paths — {', '.join(bp['unique_services'])}")
        print(f"\nTotal paths: {inv['total_executable_paths']}")
        print(f"Inventory: {paths['executable_action_inventory']}")
        if "action_path_coverage_audit" in paths:
            cov = json.loads(paths["action_path_coverage_audit"].read_text(encoding="utf-8"))
            print(f"Coverage: {cov['overall_coverage_ratio']:.1%} ({cov['total_covered_paths']}/{cov['total_executable_paths']})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
