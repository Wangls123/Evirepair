from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.final_coordinator import run_final_regeneration
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import FINAL_OUTPUT_DIR_NAME

def main() -> int:
    parser = argparse.ArgumentParser(description="Final single-scene regeneration")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / FINAL_OUTPUT_DIR_NAME))
    args = parser.parse_args()
    out = Path(args.output)
    report = run_final_regeneration(out)
    return 0 if report.get("verdict") == "READY_FOR_SINGLE_SCENE_FREEZE" else 1

if __name__ == "__main__":
    raise SystemExit(main())
