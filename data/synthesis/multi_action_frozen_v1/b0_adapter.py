from __future__ import annotations

import ast
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from smarthome_mdf.multi_action_frozen_v1.config import B0_ADAPTER_VERSION, EXPECTED_MULTI_ACTION_FROZEN_SHA256
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import resolve_frozen_instance
from smarthome_mdf.multi_action_frozen_v1.frozen_runtime_projector import project_component_runtime
from smarthome_mdf.multi_action_vnext.parent_executor import execute_parent_b0
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path

B0_FORBIDDEN_TOKENS = (
    "expected_actions",
    "expected_state",
    "conflict_label",
    "repair_actions",
    "gt_builder",
    "construct_sample_gt",
    "component_oracle_traces",
    "ars_evaluator",
    "llm_multi_action_label",
    "get_v3_b0_actions",
)

B0_FORBIDDEN_IMPORTS = (
    "smarthome_mdf.multi_action_vnext.gt_builder",
    "smarthome_mdf.labeling.llm_multi_action_label",
    "smarthome_mdf.multi_action_vnext.conflict_",
    "smarthome_mdf.compositional_repair",
)

def frozen_to_b0_parent(ma: dict[str, Any], frozen_index: dict[str, dict[str, Any]]) -> dict[str, Any]:

    components = ma.get("components") or []
    automation_components: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    order: list[str] = []
    for comp in components:
        cid = str(comp.get("component_id") or "")
        order.append(cid)
        sid = str(comp.get("single_scene_sample_id") or "")
        ss = frozen_index.get(sid, {})
        scene = str(comp.get("scene") or ss.get("scene_type") or "")
        iid = str(comp.get("grounded_instance") or (ss.get("blueprint_binding") or {}).get("automation_instance_id") or "")
        bp_id = str(comp.get("blueprint_id") or (ss.get("blueprint_binding") or {}).get("blueprint_id") or "")
        inst = resolve_frozen_instance(iid, blueprint_id=bp_id, scene=scene)
        rs = comp.get("runtime_state") or {}
        projected = project_component_runtime(
            scene=scene,
            single_scene_sample=ss,
            component_runtime=rs,
            blueprint_inputs=dict(inst.get("blueprint_inputs") or {}),
        )
        automation_components.append(
            {
                "component_id": cid,
                "blueprint_id": bp_id,
                "automation_instance_id": iid or inst.get("automation_instance_id"),
                "blueprint_inputs": dict(inst.get("blueprint_inputs") or {}),
                "scene_type": scene,
            }
        )
        contexts.append(
            {
                "component_id": cid,
                "scene_type": scene,
                "observed": projected["observed"],
                "derived_observation": projected["derived_observation"],
                "system_state": dict(rs.get("system_state") or ss.get("system_state") or {}),
                "entity_observations": projected["entity_observations"],
                "trigger_context": projected["trigger_context"],
                "entity_states": projected["entity_states"],
                "runtime_memory": projected["runtime_memory"],
                "synthetic_runtime_state": projected["synthetic_runtime_state"],
            }
        )
    return {
        "sample_id": ma.get("multi_action_id"),
        "schema_version": "multi_action_frozen_v1_1",
        "automation_components": automation_components,
        "component_runtime_contexts": contexts,
        "ordering_semantics": {"component_execution_order": order},
        "source_mode": ma.get("source_mode"),
        "composition_timeline": ma.get("composition_timeline"),
        "shared_entities": ma.get("shared_entities"),
        "entity_binding_by_component": {str(c["component_id"]): c.get("entity_binding") for c in components},
    }

def verify_yaml_resolution(parent: dict[str, Any]) -> tuple[list[str], list[dict[str, str]]]:

    resolved: list[str] = []
    failures: list[dict[str, str]] = []
    for comp in parent.get("automation_components") or []:
        bid = str(comp.get("blueprint_id") or "")
        try:
            ypath = resolve_yaml_path(bid)
        except KeyError:
            failures.append({"blueprint_id": bid, "component_id": str(comp.get("component_id") or ""), "reason": "unknown_blueprint_id"})
            continue
        if ypath and ypath.is_file():
            resolved.append(bid)
        else:
            failures.append({"blueprint_id": bid, "component_id": str(comp.get("component_id") or ""), "reason": "yaml_not_found"})
    return resolved, failures

@contextmanager
def blueprint_yaml_bridge() -> Iterator[None]:

    import smarthome_mdf.synthesis_v3.blueprint_registry as blueprint_registry

    orig_registry = blueprint_registry.get_blueprint

    def _patched_get_blueprint(blueprint_id: str) -> dict[str, Any]:
        ypath = resolve_yaml_path(blueprint_id)
        if ypath and ypath.is_file():
            return {"blueprint_id": blueprint_id, "blueprint_path": str(ypath)}
        return orig_registry(blueprint_id)

    blueprint_registry.get_blueprint = _patched_get_blueprint
    try:
        yield
    finally:
        blueprint_registry.get_blueprint = orig_registry

def execute_frozen_parent_b0(
    ma: dict[str, Any],
    frozen_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:

    parent = frozen_to_b0_parent(ma, frozen_index)
    _, yaml_failures = verify_yaml_resolution(parent)
    if yaml_failures:
        raise RuntimeError(f"B0_YAML_RESOLUTION_FAIL: {yaml_failures}")

    with blueprint_yaml_bridge():
        raw = execute_parent_b0(parent)

    semantic = raw.get("semantic_actions")
    if not isinstance(semantic, list):
        raise RuntimeError("B0 execution missing semantic_actions list")

    side_effects: list[dict[str, Any]] = []
    for act in semantic:
        prov = act.get("provenance") or {}
        if prov.get("source_branch") or prov.get("auxiliary") or str(act.get("service", "")).startswith("valid."):
            side_effects.append(
                {
                    "action_id": act.get("action_id"),
                    "service": act.get("service"),
                    "classification": "auxiliary_or_side_effect" if str(act.get("service", "")).startswith("valid.") else "branch_side_effect",
                    "provenance": prov,
                }
            )

    return {
        "multi_action_id": ma.get("multi_action_id"),
        "development_pilot_only": True,
        "multi_action_frozen_corpus_sha256": EXPECTED_MULTI_ACTION_FROZEN_SHA256,
        "adapter_version": B0_ADAPTER_VERSION,
        "canonical_b0": {
            "semantic_actions": semantic,
            "provenance": raw.get("provenance"),
            "merge_order": raw.get("merge_order"),
            "dedupe_log": raw.get("dedupe_log"),
            "side_effects_audit": side_effects,
        },
        "per_component_b0": raw.get("per_component_b0"),
        "execution_traces": raw.get("execution_traces"),
        "compatibility_debug": {
            "service_level_view": raw.get("service_level_view"),
            "service_level_view_is_canonical": False,
            "parent_b0": raw.get("parent_b0"),
        },
    }

def _scan_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports

def audit_b0_information_flow() -> dict[str, Any]:

    path = Path(__file__)
    imports = _scan_imports(path)
    violations: list[dict[str, str]] = []
    for imp in imports:
        for forbidden in B0_FORBIDDEN_IMPORTS:
            if imp.startswith(forbidden):
                violations.append({"token": forbidden, "kind": "import", "module": imp})
    return {
        "B0_INFORMATION_FLOW_VIOLATION": len(violations),
        "violations": violations,
    }

def audit_b0_y_coupling_from_b0_side() -> dict[str, Any]:

    imports = _scan_imports(Path(__file__))
    y_coupling = [i for i in imports if "llm" in i.lower() or "labeling" in i or "gt_builder" in i]
    return {"B0_Y_COUPLING_VIOLATION": len(y_coupling), "imports": y_coupling}
