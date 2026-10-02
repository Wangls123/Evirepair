from __future__ import annotations

import json

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs

from smarthome_mdf.single_scene_blueprint_complete.blueprint_capacity import build_capacity_row

from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, PROJECT_SCENES

from smarthome_mdf.single_scene_blueprint_complete.diversity_source_selector import DiversitySourceSelector

from smarthome_mdf.single_scene_blueprint_complete.final_coordinator import _load_checkpoint

from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (

    GROUNDED_CAPABLE_BLUEPRINTS,

    V3_CLEAN_OUTPUT_DIR_NAME,

)

from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler

from smarthome_mdf.single_scene_blueprint_complete.grounded_instance_cache import resolve_grounded_instances

from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool

from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry

def main() -> None:

    out = OUTPUT_DIR / V3_CLEAN_OUTPUT_DIR_NAME

    samples = _load_checkpoint(out / "single_scene_samples.jsonl")

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    bp_state = {r["blueprint_id"]: r for r in manifest["scheduler_state"]["blueprints"]}

    grounded_raw = json.loads((out / "grounded_instance_candidates.json").read_text(encoding="utf-8"))

    inst_idx = {

        bp: [

            {

                "automation_instance_id": c["automation_instance_id"],

                "blueprint_id": c["blueprint_id"],

                "scene": c["scene"],

                "blueprint_inputs": c.get("input_bindings") or {},

                "bound_entities": c.get("entity_bindings") or [],

            }

            for c in cands

        ]

        for bp, cands in grounded_raw.items()

    }

    reg = build_registry()

    pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)

    resolve_grounded_instances(out, pool, reg, clean_run=True, resume_requested=True, verbose=False)

    specs = build_all_behavior_specs(reg, instance_overrides=inst_idx)

    blueprint_budget = json.loads((out / "blueprint_budget.json").read_text(encoding="utf-8"))

    scheduler = FinalGenerationScheduler(specs, blueprint_budget.get("blueprints") or [], instance_templates=inst_idx)

    selector = DiversitySourceSelector()

    selector.restore_from_samples(samples)

    rows: list[dict] = []

    for bp_id in GROUNDED_CAPABLE_BLUEPRINTS:

        st = bp_state.get(bp_id) or {}

        scene = st.get("scene") or (inst_idx.get(bp_id) or [{}])[0].get("scene", "")

        rows.append(

            build_capacity_row(

                blueprint_id=bp_id,

                scene=scene,

                samples=samples,

                scheduler=scheduler,

                selector=selector,

                pool=pool,

                inst_list=inst_idx.get(bp_id) or [],

                grounded_raw=grounded_raw.get(bp_id) or [],

                reg=reg,

                scheduler_stop_reason=st.get("stop_reason", "UNKNOWN"),

            )

        )

    report_path = out / "blueprint_grounded_capacity_report.json"

    report_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    headers = [

        "blueprint_id",

        "scene",

        "blueprint_state",

        "accepted_count",

        "current_grounded_instance_count",

        "current_grounded_instances_covered",

        "reachable_behavior_count",

        "covered_behavior_count",

        "compatible_source_count",

        "used_source_count",

        "grounded_behavior_source_capacity",

        "accepted_behavior_source_pairs",

        "unused_behavior_source_pairs",

        "remaining_grounded_pairs",

        "scheduler_stop_reason",

    ]

    print("\t".join(headers))

    for r in rows:

        print("\t".join(str(r[h]) for h in headers))

    print(f"\nWrote {report_path}", file=sys.stderr)

if __name__ == "__main__":

    main()
