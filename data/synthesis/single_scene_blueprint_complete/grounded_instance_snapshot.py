from __future__ import annotations

from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import _normalize_from_instance
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path

def instance_from_sample(sample: dict) -> dict:

    bb = sample.get("blueprint_binding") or {}
    snap = bb.get("grounded_instance") or sample.get("synthesis_metadata", {}).get("grounded_instance") or {}
    bp = bb.get("blueprint_id", "")
    iid = bb.get("automation_instance_id", "")
    scene = sample.get("scene_type", "")
    if snap:
        return {
            "automation_instance_id": snap.get("automation_instance_id") or iid,
            "blueprint_id": snap.get("blueprint_id") or bp,
            "scene": snap.get("scene") or scene,
            "blueprint_inputs": dict(snap.get("blueprint_inputs") or {}),
            "bound_entities": list(snap.get("bound_entities") or bb.get("entities") or []),
            "configuration_signature": snap.get("configuration_signature"),
        }
    return {
        "automation_instance_id": iid,
        "blueprint_id": bp,
        "scene": scene,
        "blueprint_inputs": {},
        "bound_entities": list(bb.get("entities") or []),
    }

def build_ir_for_sample(sample: dict, reg: dict | None = None):
    reg = reg or build_registry()
    inst = instance_from_sample(sample)
    bp = inst["blueprint_id"]
    iid = inst["automation_instance_id"]
    ypath = resolve_yaml_path(bp)
    if not ypath or not Path(ypath).is_file():
        return None, inst, _normalize_from_instance(sample.get("scene_type", ""), inst, reg)
    ir = parse_blueprint_ir(bp, Path(ypath), dict(inst.get("blueprint_inputs") or {}), automation_instance_id=iid)
    nb = _normalize_from_instance(sample.get("scene_type", ""), inst, reg)
    return ir, inst, nb

def grounded_instance_snapshot(instance: dict) -> dict[str, Any]:
    return {
        "automation_instance_id": instance.get("automation_instance_id"),
        "blueprint_id": instance.get("blueprint_id"),
        "scene": instance.get("scene"),
        "configuration_signature": instance.get("configuration_signature"),
        "blueprint_inputs": dict(instance.get("blueprint_inputs") or {}),
        "bound_entities": list(instance.get("bound_entities") or []),
        "parameter_bindings": {
            k: v for k, v in (instance.get("blueprint_inputs") or {}).items() if not isinstance(v, (list, dict))
        },
    }
