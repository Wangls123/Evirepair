from __future__ import annotations

from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.config import SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    GROUNDED_CAPABLE_BLUEPRINTS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
)
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path

def _complexity_features(spec: dict, bundle_req_count: int, ir_branch_count: int) -> dict[str, Any]:
    branches = spec.get("branches") or []
    templates = spec.get("semantic_action_templates") or []
    services = spec.get("action_services") or []
    return {
        "branch_count": len(branches),
        "action_path_count": len({t.get("service", "") + "|" + str(t.get("branch_id", "")) for t in templates}),
        "action_template_count": len(templates),
        "action_domain_size": len(spec.get("action_domain") or []),
        "action_service_count": len(services),
        "runtime_requirement_count": bundle_req_count,
        "entity_role_count": len(spec.get("entity_requirements") or []),
        "automation_instance_count": 1,
        "history_requirement": bool(spec.get("history_requirements")),
        "temporal_requirement": bool(spec.get("temporal_requirements")),
        "trigger_count": len(spec.get("triggers") or []),
        "ir_branch_count": ir_branch_count,
    }

def _score(features: dict[str, Any]) -> float:
    return (
        features["branch_count"] * 2.0
        + features["action_path_count"] * 1.5
        + features["action_template_count"] * 1.0
        + features["action_domain_size"] * 3.0
        + features["runtime_requirement_count"] * 1.2
        + features["entity_role_count"] * 0.5
        + features["trigger_count"] * 0.8
        + features["ir_branch_count"] * 0.5
        + (5.0 if features["history_requirement"] else 0.0)
        + (4.0 if features["temporal_requirement"] else 0.0)
    )

def build_blueprint_complexity(specs: list[dict], registry: dict | None = None) -> dict[str, Any]:
    registry = registry or build_registry()
    inst_doc = build_instance_templates()
    by_bp: dict[str, dict] = {}
    spec_by_bp = {s["blueprint_id"]: s for s in specs}

    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            if bp_id in by_bp:
                continue
            spec = spec_by_bp.get(bp_id) or {}
            inst = ((inst_doc.get("instances") or {}).get(bp_id) or [{}])[0]
            nb = _normalize_from_instance(scene, inst, registry)
            bundle_req_count = 0
            ir_branch_count = 0
            path = resolve_yaml_path(bp_id)
            if path and path.is_file() and nb.parse_status == "PARSE_OK":
                try:
                    ir = parse_blueprint_ir(bp_id, path, inst.get("blueprint_inputs") or {})
                    bundle = extract_requirement_bundle(nb, ir=ir)
                    bundle_req_count = len(bundle.requirements)
                    ir_branch_count = len(ir.branches or [])
                except Exception:
                    pass
            features = _complexity_features(spec, bundle_req_count, ir_branch_count)
            score = _score(features)
            by_bp[bp_id] = {
                "blueprint_id": bp_id,
                "scene": scene,
                "complexity_features": features,
                "complexity_score": round(score, 3),
                "grounded_capable": bp_id in GROUNDED_CAPABLE_BLUEPRINTS,
                "true_evidence_limited": bp_id in TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
            }

    rows = list(by_bp.values())
    total_score = sum(r["complexity_score"] for r in rows if r["grounded_capable"]) or 1.0
    for r in rows:
        if r["grounded_capable"]:
            r["budget_weight"] = round(r["complexity_score"] / total_score, 6)
        else:
            r["budget_weight"] = 0.0
    return {"blueprints": rows, "grounded_capable_count": len(GROUNDED_CAPABLE_BLUEPRINTS)}
