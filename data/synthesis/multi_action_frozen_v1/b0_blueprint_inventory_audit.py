from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path
from smarthome_mdf.synthesis_v3.frozen_blueprint_executor_registry import (
    ExecutionMode,
    classify_blueprint,
    get_executor_profile,
    registry_audit,
)

CONSTRUCTION_LEAKAGE_TOKENS = (
    "action_path_signature",
    "behavior_target",
    "construction_trace",
    "component_action_schema",
    "selected_path",
)

def yaml_sha256(blueprint_id: str) -> str | None:
    try:
        ypath = resolve_yaml_path(blueprint_id)
    except KeyError:
        return None
    if not ypath or not ypath.is_file():
        return None
    return hashlib.sha256(ypath.read_bytes()).hexdigest()

def audit_blueprint_inventory() -> dict[str, Any]:
    rows = []
    scene_fallback = 0
    unsupported = 0
    not_implemented = 0
    for bp_id in sorted(ALL_BLUEPRINT_IDS):
        prof = get_executor_profile(bp_id)
        mode = classify_blueprint(bp_id)
        if mode == ExecutionMode.SCENE_FALLBACK_ONLY:
            scene_fallback += 1
        if mode == ExecutionMode.UNSUPPORTED:
            unsupported += 1
            not_implemented += 1
        try:
            ypath = resolve_yaml_path(bp_id)
            yaml_path = str(ypath) if ypath else None
        except KeyError:
            yaml_path = None
        rows.append(
            {
                "blueprint_id": bp_id,
                "scene": prof.scene if prof else "unknown",
                "yaml_path": yaml_path,
                "yaml_sha256": yaml_sha256(bp_id),
                "executor_implementation_path": "smarthome_mdf.synthesis_v3.blueprint_executor._execute_frozen_blueprint_id",
                "execution_mode": mode.value,
                "executor_kind": prof.executor_kind.value if prof else None,
                "trigger_support": prof.trigger_support if prof else False,
                "condition_support": prof.condition_support if prof else False,
                "branch_support": prof.branch_support if prof else False,
                "action_support": prof.action_support if prof else False,
                "parameter_support": prof.parameter_support if prof else False,
            }
        )
    reg = registry_audit()
    return {
        "blueprints": rows,
        "SCENE_FALLBACK_ONLY": scene_fallback,
        "UNSUPPORTED_BLUEPRINT": unsupported,
        "EXECUTOR_NOT_IMPLEMENTED": not_implemented,
        "registry_summary": reg,
    }

def audit_yaml_resolution(parent: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for comp in parent.get("automation_components") or []:
        bid = str(comp.get("blueprint_id") or "")
        try:
            ypath = resolve_yaml_path(bid)
            if not ypath or not ypath.is_file():
                mismatches.append({"blueprint_id": bid, "reason": "yaml_not_found"})
        except KeyError:
            mismatches.append({"blueprint_id": bid, "reason": "unknown_blueprint_id"})
    return {"B0_BLUEPRINT_YAML_MISMATCH": len(mismatches), "mismatches": mismatches}

def audit_construction_path_leakage() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    b0_paths = [
        root / "synthesis_v3" / "blueprint_executor.py",
        root / "synthesis_v3" / "generic_blueprint_executor.py",
        root / "multi_action_frozen_v1" / "b0_adapter.py",
        root / "multi_action_frozen_v1" / "frozen_runtime_projector.py",
        root / "multi_action_vnext" / "parent_executor.py",
    ]
    violations = []
    for path in b0_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if "action_path_signature" in text:
            if path.name == "frozen_runtime_projector.py":
                continue
            if re.search(r"action_path_signature[^\n]*(?:branch|decision|select|require_duration)", text):
                violations.append({"file": str(path), "token": "action_path_signature"})
        if re.search(r"component_action_schema[^\n]*(?:execute|branch|select)", text):
            violations.append({"file": str(path), "token": "component_action_schema"})
    return {"B0_CONSTRUCTION_PATH_LEAKAGE": len(violations), "violations": violations}

def audit_runtime_projection_neutrality() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "frozen_runtime_projector.py"
    text = path.read_text(encoding="utf-8")
    leakage = 0
    if "action_path_signature" in text and "must NOT" not in text:
        leakage += 1
    if re.search(r"action_sig\.startswith", text):
        leakage += 1
    return {"B0_RUNTIME_PROJECTION_DECISION_LEAKAGE": leakage}
