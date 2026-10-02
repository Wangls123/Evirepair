from __future__ import annotations

GROUNDED_INSTANCE_GENERATOR_VERSION = "grounded_instances_v3_config_sig"

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle, minimal_grounding_requirements
from smarthome_mdf.multi_action_vnext.source_adapters import SourcePool
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import satisfies_requirement
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.config import SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import default_inputs_for
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import TRUE_EVIDENCE_LIMITED_BLUEPRINTS
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import _default_inputs
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.semantic_capabilities import attach_capabilities
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path, yaml_exists

@dataclass
class GroundedInstanceCandidate:
    automation_instance_id: str
    blueprint_id: str
    scene: str
    input_bindings: dict[str, Any]
    entity_bindings: list[str]
    parameter_bindings: dict[str, Any]
    configuration_signature: str
    display_name: str = ""
    matcher_viable: bool = False
    supporting_source_count: int = 0

def _configuration_signature(blueprint_id: str, inputs: dict, entities: list[str]) -> str:
    payload = json.dumps({"blueprint_id": blueprint_id, "inputs": inputs, "entities": sorted(entities)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def _instance_id_from_signature(blueprint_id: str, sig: str) -> str:
    return f"{blueprint_id}__inst_{sig}"

def _variant_inputs(scene: str, blueprint_id: str, base: dict) -> list[dict[str, Any]]:
    variants = [dict(base)]
    alt = dict(base)
    if "open_delay_sec" in alt:
        alt["open_delay_sec"] = 45
    if "illuminace_level" in alt:
        alt["illuminace_level"] = 80
    if "start_threshold" in alt:
        alt["start_threshold"] = 15
    if "no_motion_wait" in alt:
        alt["no_motion_wait"] = 180
    if "motion_entity" in alt and scene == "advanced_lighting":
        alt["motion_entity"] = "binary_sensor.motion_hallway"
    if alt != base:
        variants.append(alt)
    return variants

def _entities_from_inputs(inputs: dict) -> list[str]:
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str) and "." in node:
            out.add(node)
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(inputs)
    return sorted(out)

def _instance_can_match(pool: SourcePool, scene: str, instance: dict, reg: dict, *, max_probe: int = 200) -> tuple[bool, int]:
    bp = instance["blueprint_id"]
    iid = instance["automation_instance_id"]
    nb = _normalize_from_instance(scene, instance, reg)
    if nb.parse_status != "PARSE_OK":
        return False, 0
    ypath = resolve_yaml_path(bp)
    ir = parse_blueprint_ir(bp, Path(ypath), dict(instance.get("blueprint_inputs") or {}), automation_instance_id=iid) if ypath else None
    bundle = extract_requirement_bundle(nb, ir=ir)
    grounding = minimal_grounding_requirements(bundle)
    viable = 0
    for cand in pool.candidates_for_scene(scene)[:max_probe]:
        obs = attach_capabilities(cand)
        ok_all = True
        for req in grounding:
            ok, _, _ = satisfies_requirement(req, obs)
            if not ok:
                ok_all = False
                break
        if ok_all:
            viable += 1
    return viable > 0, viable

def enumerate_grounded_instances(
    pool: SourcePool,
    *,
    reg: dict | None = None,
    max_probe: int = 200,
) -> dict[str, list[GroundedInstanceCandidate]]:
    reg = reg or build_registry()
    out: dict[str, list[GroundedInstanceCandidate]] = {}
    for scene, bp_ids in SCENE_BLUEPRINT_MAP.items():
        for bp_id in bp_ids:
            if bp_id in TRUE_EVIDENCE_LIMITED_BLUEPRINTS or not yaml_exists(bp_id):
                continue
            base = default_inputs_for(bp_id, scene) or _default_inputs(scene, bp_id)
            candidates: list[GroundedInstanceCandidate] = []
            seen_sig: set[str] = set()
            for inputs in _variant_inputs(scene, bp_id, base):
                entities = _entities_from_inputs(inputs)
                sig = _configuration_signature(bp_id, inputs, entities)
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                iid = _instance_id_from_signature(bp_id, sig)
                inst = {
                    "automation_instance_id": iid,
                    "blueprint_id": bp_id,
                    "scene": scene,
                    "blueprint_inputs": inputs,
                    "bound_entities": entities,
                }
                viable, count = _instance_can_match(pool, scene, inst, reg, max_probe=max_probe)
                candidates.append(
                    GroundedInstanceCandidate(
                        automation_instance_id=iid,
                        blueprint_id=bp_id,
                        scene=scene,
                        input_bindings=inputs,
                        entity_bindings=entities,
                        parameter_bindings={k: v for k, v in inputs.items() if not isinstance(v, (list, dict))},
                        configuration_signature=sig,
                        display_name=f"cfg_{sig[:8]}",
                        matcher_viable=viable,
                        supporting_source_count=count,
                    )
                )
            out[bp_id] = [c for c in candidates if c.matcher_viable] or candidates
    return out

def instances_to_templates(grounded: dict[str, list[GroundedInstanceCandidate]]) -> dict[str, list[dict]]:
    templates: dict[str, list[dict]] = {}
    for bp_id, cands in grounded.items():
        templates[bp_id] = [
            {
                "automation_instance_id": c.automation_instance_id,
                "blueprint_id": c.blueprint_id,
                "scene": c.scene,
                "description": c.display_name,
                "blueprint_inputs": c.input_bindings,
                "bound_entities": c.entity_bindings,
                "configuration_signature": c.configuration_signature,
                "yaml_available": True,
            }
            for c in cands
        ]
    return templates
