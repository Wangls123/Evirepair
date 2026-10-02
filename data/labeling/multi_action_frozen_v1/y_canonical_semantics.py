from __future__ import annotations

from functools import lru_cache
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.y_yaml_services import extract_static_yaml_services
from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path
from smarthome_mdf.synthesis_v3.blueprint_parser import parse_blueprint_yaml

_FORBIDDEN_SEMANTICS_KEYS = frozenset(
    {
        "branch_id",
        "behavior_target",
        "expected_action",
        "expected_service",
        "action_path_signature",
        "strict_path_key",
        "multi_action_gt",
        "selected_runtime_path",
    }
)

def _sanitize_node(node: dict[str, Any]) -> dict[str, Any]:

    out: dict[str, Any] = {}
    ntype = node.get("node_type")
    if ntype:
        out["node_type"] = ntype
    if node.get("service"):
        out["service"] = node["service"]
    if node.get("target_expression"):
        out["target_expression"] = node["target_expression"]
    if node.get("data_expression"):
        out["data_expression"] = node["data_expression"]
    if node.get("conditions") is not None:
        out["conditions"] = node["conditions"]
    if node.get("expression") is not None:
        out["expression"] = node["expression"]
    if node.get("duration") is not None or node.get("duration_expression"):
        out["delay"] = node.get("duration") or node.get("duration_expression")
    if node.get("trigger_spec"):
        out["wait_for_trigger"] = node["trigger_spec"]
    if node.get("timeout") is not None:
        out["timeout"] = node["timeout"]
    if ntype == "ChooseNode":
        out["branches"] = [
            {
                "conditions": b.get("conditions"),
                "sequence": [_sanitize_node(n) for n in (b.get("sequence") or [])],
            }
            for b in (node.get("branches") or [])
        ]
        if node.get("default_sequence"):
            out["default_sequence"] = [_sanitize_node(n) for n in node["default_sequence"]]
    elif ntype == "SequenceNode":
        out["sequence"] = [_sanitize_node(n) for n in (node.get("sequence") or [])]
    elif ntype == "ConditionNode":
        out["then_sequence"] = [_sanitize_node(n) for n in (node.get("then_sequence") or [])]
        out["else_sequence"] = [_sanitize_node(n) for n in (node.get("else_sequence") or [])]
    elif ntype == "RepeatNode":
        out["repeat_type"] = node.get("repeat_type")
        out["sequence"] = [_sanitize_node(n) for n in (node.get("sequence") or [])]
    elif ntype == "ParallelNode":
        out["branches"] = [
            {"sequence": [_sanitize_node(n) for n in (b.get("sequence") or [])]}
            for b in (node.get("branches") or [])
        ]
    return out

def _sanitize_condition(cond: Any) -> Any:

    if isinstance(cond, list):
        return [_sanitize_condition(c) for c in cond]
    if not isinstance(cond, dict):
        return cond
    keep = (
        "condition",
        "trigger",
        "state",
        "numeric_state",
        "template",
        "time",
        "and",
        "or",
        "not",
        "id",
        "entity_id",
        "from",
        "to",
        "above",
        "below",
        "value",
        "value_template",
        "before",
        "after",
        "weekday",
    )
    out: dict[str, Any] = {}
    for k, v in cond.items():
        if k in _FORBIDDEN_SEMANTICS_KEYS:
            continue
        if k in keep:
            out[k] = _sanitize_condition(v) if k in ("and", "or", "not", "condition") else v
        elif k == "conditions":
            out["conditions"] = _sanitize_condition(v)
    return out if out else cond

def _extract_condition_semantics(triggers: list[dict], control_flow: list[dict]) -> list[dict]:

    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("conditions") is not None:
                found.append({"source": "control_flow", "conditions": _sanitize_condition(node["conditions"])})
            if node.get("condition"):
                found.append({"source": "node", "condition": _sanitize_condition(node["condition"])})
            for key in ("sequence", "then_sequence", "else_sequence", "default_sequence", "branches"):
                for child in node.get(key) or []:
                    if isinstance(child, dict) and child.get("sequence"):
                        for n in child["sequence"]:
                            walk(n)
                    else:
                        walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for t in triggers or []:
        if t.get("condition"):
            found.append({"source": "trigger", "condition": _sanitize_condition(t["condition"])})
    walk(control_flow)
    return found

def _sanitize_triggers(triggers: list[dict]) -> list[dict]:
    clean: list[dict] = []
    keep_keys = (
        "platform",
        "id",
        "trigger_id",
        "entity_id",
        "from",
        "to",
        "event",
        "event_type",
        "duration",
        "at",
        "offset",
        "value_template",
        "alias",
        "condition",
    )
    for t in triggers or []:
        if not isinstance(t, dict):
            continue
        row = {k: v for k, v in t.items() if k in keep_keys and k not in _FORBIDDEN_SEMANTICS_KEYS}
        if t.get("condition") and isinstance(t["condition"], dict):
            row["condition"] = _sanitize_condition(t["condition"])
        clean.append(row)
    return clean

@lru_cache(maxsize=64)
def _load_static_ir(blueprint_id: str) -> tuple[Any, Any]:
    ypath = resolve_yaml_path(blueprint_id)
    if not ypath or not ypath.is_file():
        return None, None
    spec = parse_blueprint_yaml(blueprint_id, ypath, {})
    ir = parse_blueprint_ir(blueprint_id, ypath, {})
    return spec, ir

def build_canonical_blueprint_semantics(
    blueprint_id: str,
    blueprint_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:

    inputs = dict(blueprint_inputs or {})
    yaml_meta = extract_static_yaml_services(blueprint_id)
    spec, ir = _load_static_ir(blueprint_id)

    semantics: dict[str, Any] = {
        "blueprint_id": blueprint_id,
        "blueprint_inputs": inputs,
        "static_declared_services": yaml_meta["services"],
        "dynamic_action_inputs": yaml_meta["dynamic_action_inputs"],
    }

    if spec is not None:
        semantics["canonical_trigger_semantics"] = _sanitize_triggers(list(spec.triggers or []))
        semantics["required_inputs"] = list(spec.required_inputs or [])
        semantics["input_defaults"] = dict(spec.input_defaults or {})
        semantics["input_selectors"] = dict(spec.input_selectors or {})

    if ir is not None:
        control_flow = [_sanitize_node(n) for n in (ir.root_sequence or [])]
        semantics["canonical_action_control_flow"] = control_flow
        semantics["unsupported_constructs"] = list(ir.unsupported_nodes or [])
        semantics["canonical_condition_semantics"] = _extract_condition_semantics(
            semantics.get("canonical_trigger_semantics") or [],
            control_flow,
        )

    return semantics
