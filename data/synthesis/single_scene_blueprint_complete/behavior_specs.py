from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.behavior_spec import build_behavior_spec
from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.blueprint_parser import NormalizedBlueprint
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.local_registry import load_registry
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path, yaml_exists
from smarthome_mdf.synthesis_v3.blueprint_parser import parse_blueprint_yaml

def _normalize_from_instance(scene: str, instance: dict, registry: dict) -> NormalizedBlueprint:
    bp_id = instance["blueprint_id"]
    iid = instance["automation_instance_id"]
    meta = (registry.get("blueprints") or {}).get(bp_id) or {}
    path = resolve_yaml_path(bp_id)
    if not path or not path.is_file():
        return NormalizedBlueprint(
            blueprint_id=bp_id,
            source="missing_yaml",
            scene=scene,
            triggers=[],
            conditions=[],
            actions=[],
            entities=[],
            parameters={},
            temporal_requirements=[],
            state_requirements=[],
            required_runtime_evidence=[],
            provenance={"automation_instance_id": iid},
            parse_status="IMPLEMENTATION_GAP",
            rejection_reason="YAML_MISSING",
        )
    inputs = dict(instance.get("blueprint_inputs") or {})
    try:
        spec = parse_blueprint_yaml(bp_id, path, inputs)
        ir = parse_blueprint_ir(bp_id, path, inputs, automation_instance_id=iid)
        actions = sorted({ap.service for ap in spec.action_paths if ap.service})
        ir_actions = sorted({t.get("service") for t in (ir.action_templates or []) if t.get("service")})
        actions = sorted(set(actions) | set(ir_actions))
        ir_ok = ir.parse_status == "PARSE_OK" or bool(ir.root_sequence or ir.branches or ir.action_templates)
        if not path.is_file():
            status = "IMPLEMENTATION_GAP"
            reason = "YAML_MISSING"
        elif ir_ok or actions:
            status = "PARSE_OK"
            reason = None
        elif ir.parse_status == "UNSUPPORTED_SEMANTICS":
            status = "IMPLEMENTATION_GAP"
            reason = "UNSUPPORTED_SEMANTICS"
        else:
            status = "IMPLEMENTATION_GAP"
            reason = f"unsupported={spec.unsupported_constructs[:3]}"
        entities = list(instance.get("bound_entities") or [])
        return NormalizedBlueprint(
            blueprint_id=bp_id,
            source=str(path),
            scene=scene,
            triggers=list(spec.triggers or []),
            conditions=sorted({c for ap in spec.action_paths for c in (ap.conditions or [])}),
            actions=actions,
            entities=entities,
            parameters=inputs,
            temporal_requirements=[],
            state_requirements=[],
            required_runtime_evidence=[],
            provenance={
                "blueprint_path": str(path),
                "automation_instance_id": iid,
                "ir_branch_count": len(ir.branches or []),
                "ir_action_templates": len(ir.action_templates or []),
                "yaml_hash": spec.yaml_hash,
            },
            parse_status=status,
            rejection_reason=reason,
        )
    except Exception as exc:
        return NormalizedBlueprint(
            blueprint_id=bp_id,
            source=str(path),
            scene=scene,
            triggers=[],
            conditions=[],
            actions=[],
            entities=[],
            parameters=inputs,
            temporal_requirements=[],
            state_requirements=[],
            required_runtime_evidence=[],
            provenance={"automation_instance_id": iid},
            parse_status="IMPLEMENTATION_GAP",
            rejection_reason=str(exc)[:200],
        )

def build_all_behavior_specs(registry: dict | None = None, *, instance_overrides: dict[str, list[dict]] | None = None) -> list[dict]:
    registry = registry or load_registry()
    if instance_overrides is not None:
        inst_doc = {"instances": instance_overrides}
    else:
        inst_doc = build_instance_templates()
    specs: list[dict] = []
    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            for inst in (inst_doc.get("instances") or {}).get(bp_id) or []:
                nb = _normalize_from_instance(scene, inst, registry)
                yaml_spec = None
                if nb.parse_status == "PARSE_OK":
                    path = resolve_yaml_path(bp_id)
                    yaml_spec = parse_blueprint_yaml(bp_id, path, inst.get("blueprint_inputs") or {})
                bspec = build_behavior_spec(
                    nb,
                    automation_instance_id=inst["automation_instance_id"],
                    yaml_spec=yaml_spec,
                )
                row = bspec.to_dict()
                row["scene"] = scene
                row["parse_status"] = nb.parse_status
                row["yaml_exists"] = yaml_exists(bp_id)
                domains = sorted({(t.get("service") or "").split(".")[0] for t in row.get("semantic_action_templates") or [] if t.get("service")})
                row["action_domain"] = domains
                row["action_services"] = sorted({t.get("service") for t in row.get("semantic_action_templates") or [] if t.get("service")})
                specs.append(row)
    return specs

def build_action_domains(specs: list[dict]) -> dict[str, Any]:
    by_bp: dict[str, dict] = {}
    for s in specs:
        bid = s["blueprint_id"]
        if bid not in by_bp:
            by_bp[bid] = {
                "blueprint_id": bid,
                "scene": s["scene"],
                "action_domain": sorted(set(s.get("action_domain") or [])),
                "action_services": sorted(set(s.get("action_services") or [])),
                "instances": [],
            }
        by_bp[bid]["instances"].append(s["automation_instance_id"])
        by_bp[bid]["action_services"] = sorted(
            set(by_bp[bid]["action_services"]) | set(s.get("action_services") or [])
        )
    return {"blueprints": list(by_bp.values()), "total": len(by_bp)}

def run_parser_audit(registry: dict | None = None) -> dict[str, Any]:
    registry = registry or load_registry()
    inst_doc = build_instance_templates()
    rows: list[dict] = []
    gap = 0
    ok = 0
    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            inst = ((inst_doc.get("instances") or {}).get(bp_id) or [{}])[0]
            nb = _normalize_from_instance(scene, inst, registry)
            if nb.parse_status == "PARSE_OK":
                ok += 1
                st = "PARSE_OK"
            else:
                gap += 1
                st = "IMPLEMENTATION_GAP"
            rows.append(
                {
                    "blueprint_id": bp_id,
                    "scene": scene,
                    "yaml_exists": yaml_exists(bp_id),
                    "parse_status": st,
                    "rejection_reason": nb.rejection_reason,
                }
            )
    return {
        "total_blueprints": len(rows),
        "parse_ok": ok,
        "implementation_gap": gap,
        "implementation_gap_zero": gap == 0,
        "rows": rows,
    }

def write_behavior_artifacts(out_dir: Path | None = None) -> dict[str, Path]:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    specs = build_all_behavior_specs()
    domains = build_action_domains(specs)
    audit = run_parser_audit()
    paths = {
        "behavior_specs": out / "blueprint_behavior_specs.json",
        "action_domains": out / "blueprint_action_domains.json",
        "parser_audit": out / "blueprint_parser_support.json",
    }
    paths["behavior_specs"].write_text(json.dumps(specs, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["action_domains"].write_text(json.dumps(domains, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["parser_audit"].write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
