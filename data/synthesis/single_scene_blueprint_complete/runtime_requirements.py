from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle
from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.local_registry import load_registry
from smarthome_mdf.single_scene_blueprint_complete.yaml_acquisition import resolve_canonical_path

def build_all_runtime_requirements(registry: dict | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    inst_doc = build_instance_templates()
    blueprints: list[dict] = []
    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            inst = ((inst_doc.get("instances") or {}).get(bp_id) or [{}])[0]
            nb = _normalize_from_instance(scene, inst, registry)
            ir = None
            path = resolve_canonical_path(bp_id)
            if path.is_file():
                ir = parse_blueprint_ir(bp_id, path, inst.get("blueprint_inputs") or {}, automation_instance_id=inst.get("automation_instance_id", ""))
            if nb.parse_status != "PARSE_OK":
                blueprints.append({"blueprint_id": bp_id, "scene": scene, "status": "IMPLEMENTATION_GAP", "requirements": []})
                continue
            bundle = extract_requirement_bundle(nb, ir=ir)
            reqs = [r.to_dict() for r in bundle.requirements]
            blueprints.append(
                {
                    "blueprint_id": bp_id,
                    "scene": scene,
                    "status": "OK",
                    "requirement_count": len(reqs),
                    "requirements": reqs,
                    "execution_structures": [s.to_dict() for s in bundle.execution_structures],
                }
            )
    return {"total": len(blueprints), "blueprints": blueprints}

def write_runtime_requirements(out_dir: Path | None = None) -> Path:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    doc = build_all_runtime_requirements()
    path = out / "blueprint_runtime_requirements.json"
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
