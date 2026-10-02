from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.single_scene_y_labeling_view import (
    build_single_scene_y_labeling_view,
)
from smarthome_mdf.multi_action_frozen_v1.y_compact_prompt import compact_canonical_semantics
from smarthome_mdf.multi_action_frozen_v1.y_runtime_semantic_context import build_y_runtime_semantic_context

Y_V6_PROMPT_VERSION = "frozen_independent_y_v6"
Y_V6_SCHEMA_VERSION = "formal_y_minimal_v1"

Y_ALLOWED_FIELDS = [
    "scenario",
    "observation",
    "entity_observations",
    "trigger",
    "condition",
    "blueprint",
    "bindings",
    "automation_specification",
    "intended_action_definition",
]

Y_V6_FORBIDDEN_KEYS = frozenset(
    {
        "formal_b0",
        "b0_output",
        "b0",
        "b0_actions",
        "parent_b0",
        "semantic_b0_actions",
        "expected_action_ha",
        "execution_status",
        "execution_traces",
        "execution_outcome",
        "execution_resolution",
        "diagnostic_reason",
        "trigger_matched",
        "conditions_passed",
        "repair_result",
        "repaired_actions",
        "repair_actions",
        "repair_hints",
        "formal_y",
        "y_output",
        "y_label_status",
        "eval_y",
        "ground_truth",
        "behavior_target",
        "action_path_signature",
        "branch_id",
        "selected_runtime_path",
        "synthesis_construction_trace",
    }
)

Y_V6_FORBIDDEN_SUBSTRINGS = (
    "formal_b0",
    "b0_output",
    "execution_resolution",
    "trigger_matched",
    "conditions_passed",
    "repair_result",
    "repaired_actions",
    "formal_y",
    "expected_action_ha",
    "semantic_repair_success",
)

FROZEN_INDEPENDENT_Y_V6_SS_SYSTEM_PROMPT = """You label the correct smart-home automation behavior from the current scene state and automation semantics only.

Task:
Decide whether the automation should execute one action now, or execute no action.

Valid decisions: ACTIONS or NO_ACTION only.

Rules:
- Use trigger, condition, observation, entity states, environment, blueprint, and automation semantics.
- Do NOT treat "a trigger exists" as sufficient for ACTIONS.
- Do NOT treat "the blueprint declares an action" as sufficient for ACTIONS.
- If the target state is already satisfied, choose NO_ACTION.
- If execution conditions are met and a state change is actually required, choose ACTIONS.
- ACTIONS must contain exactly one action.
- NO_ACTION must contain zero actions.
- Do not abstain. Do not return UNKNOWN or UNCERTAIN.
- candidate services / entity catalog are the legal search space, not the answer.
- Ignore any baseline execution, B0, repair, or previous labels. They are not in the input.

Return JSON only. No markdown. No explanation.

NO_ACTION:
{"decision":"NO_ACTION","actions":[]}

ACTIONS:
{"decision":"ACTIONS","actions":[{"service":"...","target":"...","data":{}}]}

Include data only when parameters are required. Always set component_origin to c0.
"""

def _json_safe(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def _strip_forbidden(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): _strip_forbidden(v)
            for k, v in obj.items()
            if str(k) not in Y_V6_FORBIDDEN_KEYS
        }
    if isinstance(obj, list):
        return [_strip_forbidden(x) for x in obj]
    return obj

def _compact_entity_observations(obs: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in obs or []:
        if not isinstance(item, dict):
            continue
        attrs = dict(item.get("attributes") or {})
        attrs.pop("behavior_target", None)
        row = {
            "entity_id": item.get("entity_id"),
            "domain": item.get("domain"),
            "state": item.get("state"),
            "fusion_role": item.get("fusion_role"),
            "attributes": _strip_forbidden(attrs),
        }
        out.append({k: v for k, v in row.items() if v not in (None, "", [], {})})
    return out

def build_y_v6_payload(sample: dict[str, Any], view: dict[str, Any] | None = None) -> dict[str, Any]:

    view = view or build_single_scene_y_labeling_view(sample)
    bp = str(view.get("blueprint_id") or "")
    sem_full = (view.get("canonical_blueprint_semantics") or {}).get(bp) or {}
    comp = (view.get("components") or [{}])[0]
    inputs = _strip_forbidden(comp.get("blueprint_inputs") or sem_full.get("blueprint_inputs") or {})
    observed = _strip_forbidden(dict(sample.get("observed") or {}))
    runtime = _strip_forbidden(
        build_y_runtime_semantic_context(
            {
                "observed": observed,
                "entity_observations": sample.get("entity_observations") or [],
            },
            blueprint_id=bp,
            blueprint_inputs=inputs,
            canonical_semantics=sem_full,
        )
    )
    domain = view.get("candidate_action_domain") or {}
    c0_domain = domain.get("c0") or []
    if c0_domain and isinstance(c0_domain[0], dict):
        services = sorted({str(x.get("service")) for x in c0_domain if x.get("service")})
    else:
        services = sorted(set(c0_domain)) if isinstance(c0_domain, list) else []
    catalog = [
        {"entity_id": e.get("entity_id"), "role": e.get("role")}
        for e in (view.get("entity_catalog") or [])
        if e.get("entity_id")
    ]
    semantics = compact_canonical_semantics({**sem_full, "blueprint_id": bp, "blueprint_inputs": inputs})
    payload = {
        "schema_version": Y_V6_SCHEMA_VERSION,
        "prompt_version": Y_V6_PROMPT_VERSION,
        "scenario": view.get("scene_type") or sample.get("scene_type"),
        "observation": observed,
        "entity_observations": _compact_entity_observations(list(sample.get("entity_observations") or [])),
        "trigger": semantics.get("triggers") or [],
        "condition": semantics.get("conditions") or [],
        "blueprint": bp,
        "bindings": {
            "blueprint_inputs": inputs,
            "entity_catalog": catalog,
        },
        "automation_specification": {
            "control_flow": semantics.get("control_flow") or [],
            "services": semantics.get("services") or [],
            "input_selectors": semantics.get("input_selectors") or {},
        },
        "intended_action_definition": {
            "candidate_services": services,
            "runtime_context": runtime,
        },
    }
    return _json_safe(_strip_forbidden(payload))

def build_y_v6_user_prompt(view: dict[str, Any], sample: dict[str, Any] | None = None) -> str:
    if sample is None:
        sample = {
            "observed": (view.get("evidence") or {}).get("observed") or {},
            "entity_observations": (view.get("evidence") or {}).get("entity_observations") or [],
            "scene_type": view.get("scene_type"),
        }
    payload = build_y_v6_payload(sample, view)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def y_v6_prompt_hash() -> str:
    return hashlib.sha256(
        (FROZEN_INDEPENDENT_Y_V6_SS_SYSTEM_PROMPT + Y_V6_PROMPT_VERSION).encode("utf-8")
    ).hexdigest()[:16]

_SKIP_SUBSTRING_PATHS = {"schema_version", "prompt_version"}
_SCHEMA_NAME_ALLOW = {"formal_y_minimal_v1", Y_V6_PROMPT_VERSION, Y_V6_SCHEMA_VERSION}

def _walk_forbidden(obj: Any, *, path: str = "") -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k) in Y_V6_FORBIDDEN_KEYS:
                hits.append({"path": here, "kind": "forbidden_key", "key": str(k)})
            hits.extend(_walk_forbidden(v, path=here))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            hits.extend(_walk_forbidden(item, path=f"{path}[{i}]"))
    elif isinstance(obj, str):
        leaf = path.split(".")[-1] if path else ""
        if leaf in _SKIP_SUBSTRING_PATHS or obj in _SCHEMA_NAME_ALLOW:
            return hits
        lower = obj.lower()
        for tok in Y_V6_FORBIDDEN_SUBSTRINGS:
            if tok.lower() in lower:
                hits.append({"path": path, "kind": "forbidden_substring", "token": tok})
    return hits

def audit_y_v6_payload(payload: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    key_hits = _walk_forbidden(payload)
    prompt_hits = []
    lower = prompt_text.lower()
    scanned = lower.replace("formal_y_minimal_v1", "").replace("frozen_independent_y_v6", "")
    for tok in Y_V6_FORBIDDEN_SUBSTRINGS:
        if tok.lower() in scanned:
            prompt_hits.append(tok)
    extra_keys = sorted(set(payload) - set(Y_ALLOWED_FIELDS) - {"schema_version", "prompt_version"})
    return {
        "formal_b0_leaked": any(h.get("key") == "formal_b0" or h.get("token") == "formal_b0" for h in key_hits)
        or "formal_b0" in prompt_hits,
        "b0_output_leaked": any(h.get("key") == "b0_output" or h.get("token") == "b0_output" for h in key_hits)
        or "b0_output" in prompt_hits,
        "repair_result_leaked": any(
            h.get("key") in {"repair_result", "repaired_actions"} or h.get("token") in {"repair_result", "repaired_actions"}
            for h in key_hits
        )
        or "repair_result" in prompt_hits,
        "old_y_leaked": any(h.get("key") == "formal_y" or h.get("token") == "formal_y" for h in key_hits)
        or "formal_y" in prompt_hits,
        "forbidden_key_hits": key_hits[:50],
        "forbidden_prompt_tokens": prompt_hits,
        "extra_top_level_keys": extra_keys,
        "allowed_fields": list(Y_ALLOWED_FIELDS),
        "Y_LEAKAGE": int(bool(key_hits or prompt_hits or extra_keys)),
    }
