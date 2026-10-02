from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import Y_OUTPUT_SCHEMA_VERSION, Y_PROMPT_VERSION
from smarthome_mdf.multi_action_frozen_v1.y_runtime_semantic_context import build_y_runtime_semantic_context

FROZEN_INDEPENDENT_Y_V5_SYSTEM_PROMPT = """You label expected smart-home automation behavior from canonical Blueprint semantics and frozen runtime evidence.

Rules:
- Choose the single most likely expected behavior (forced argmax).
- Valid decisions: ACTIONS or NO_ACTION only.
- Missing evidence does NOT automatically mean NO_ACTION.
- If ACTIONS are more likely than NO_ACTION, output ACTIONS even when some uncertainty remains.
- Choose NO_ACTION only when NO_ACTION is more likely than every valid action outcome.
- Do NOT use NO_ACTION as a safe fallback. Do NOT abstain.
- Do NOT return UNKNOWN, UNCERTAIN, ABSTAIN, or any third-class decision label.
- candidate_action_domain is the legal search space only — not the answer.
- Base decisions on trigger → condition → control-flow → action semantics plus runtime evidence.

Return JSON only. No markdown. No explanation. No reasoning text outside JSON.

NO_ACTION schema:
{"decision":"NO_ACTION","actions":[]}

ACTIONS schema (single-scene):
{"decision":"ACTIONS","actions":[{"service":"...","target":"...","data":{}}]}

Include "data" only when parameters are required. Omit empty fields.
For multi-component samples include "component_origin" on each action."""

FROZEN_INDEPENDENT_Y_V5_SS_SYSTEM_PROMPT = FROZEN_INDEPENDENT_Y_V5_SYSTEM_PROMPT.replace(
    "multi-component samples include",
    "always use component_origin c0; multi-component samples include",
)

def _json_prompt_safe(obj: Any) -> Any:

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_prompt_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_prompt_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def _dump_prompt_payload(payload: dict[str, Any]) -> str:
    return json.dumps(_json_prompt_safe(payload), ensure_ascii=False, separators=(",", ":"))

def _compact_triggers(triggers: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in triggers or []:
        if not isinstance(t, dict):
            continue
        row: dict[str, Any] = {}
        for key in (
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
        ):
            if t.get(key) not in (None, "", [], {}):
                row[key] = t[key]
        if t.get("value_template"):
            row["value_template"] = t["value_template"]
        if row:
            out.append(row)
    return out

def compact_canonical_semantics(semantics: dict[str, Any]) -> dict[str, Any]:

    return {
        "blueprint_id": semantics.get("blueprint_id"),
        "inputs": semantics.get("blueprint_inputs") or {},
        "input_defaults": semantics.get("input_defaults") or {},
        "input_selectors": semantics.get("input_selectors") or {},
        "triggers": _compact_triggers(list(semantics.get("canonical_trigger_semantics") or [])),
        "conditions": semantics.get("canonical_condition_semantics") or [],
        "control_flow": semantics.get("canonical_action_control_flow") or [],
        "services": semantics.get("static_declared_services") or [],
    }

def _referenced_entity_keys(semantics: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    inputs = semantics.get("blueprint_inputs") or {}
    keys.update(str(k) for k in inputs)
    text = json.dumps(compact_canonical_semantics(semantics), ensure_ascii=False)
    for m in re.finditer(r"!input\s+(\w+)", text):
        keys.add(m.group(1))
    return keys

def _prune_runtime(runtime: dict[str, Any], semantics: dict[str, Any]) -> dict[str, Any]:

    bp = str(semantics.get("blueprint_id") or "")
    pruned = dict(runtime)
    obs = dict(pruned.get("observed_state") or {})

    keep_obs_keys = set(obs)
    if bp.startswith("presence_"):
        keep_obs_keys = {k for k in obs if "person" in k.lower() or "presence" in k.lower() or k in obs}
    elif bp.startswith("lighting_"):
        keep_obs_keys = {
            k
            for k in obs
            if any(x in k.lower() for x in ("motion", "lux", "illuminance", "light", "time", "schedule"))
        }
    elif bp == "security_osam_sensor_alert":
        keep_obs_keys = {k for k in obs if "motion" in k.lower() or "window" in k.lower() or "door" in k.lower()}
    pruned["observed_state"] = {k: obs[k] for k in sorted(keep_obs_keys) if k in obs}
    return pruned

def build_compact_ss_user_prompt(
    view: dict[str, Any],
    sample: dict[str, Any] | None = None,
) -> str:

    bp = str(view.get("blueprint_id") or "")
    sem_full = (view.get("canonical_blueprint_semantics") or {}).get(bp) or {}
    comp = (view.get("components") or [{}])[0]
    inputs = comp.get("blueprint_inputs") or sem_full.get("blueprint_inputs") or {}

    ss = sample or {
        "observed": (view.get("evidence") or {}).get("observed") or {},
        "entity_observations": (view.get("evidence") or {}).get("entity_observations") or [],
    }
    runtime = _prune_runtime(
        build_y_runtime_semantic_context(
            ss,
            blueprint_id=bp,
            blueprint_inputs=inputs,
            canonical_semantics=sem_full,
        ),
        sem_full,
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

    payload = {
        "schema_version": Y_OUTPUT_SCHEMA_VERSION,
        "prompt_version": Y_PROMPT_VERSION,
        "blueprint_id": bp,
        "scene": view.get("scene_type"),
        "canonical_semantics": compact_canonical_semantics({**sem_full, "blueprint_id": bp, "blueprint_inputs": inputs}),
        "runtime_context": runtime,
        "entity_catalog": catalog,
        "candidate_services": services,
    }
    return _dump_prompt_payload(payload)

def build_compact_ma_user_prompt(view: dict[str, Any], samples_by_comp: dict[str, dict] | None = None) -> str:

    per_bp: dict[str, Any] = {}
    runtime_by_comp: dict[str, Any] = {}
    for comp in view.get("components") or []:
        cid = str(comp.get("component_id") or "")
        bid = str(comp.get("blueprint_id") or "")
        sem = (view.get("canonical_blueprint_semantics") or {}).get(bid) or {}
        inputs = comp.get("blueprint_inputs") or {}
        per_bp[cid] = compact_canonical_semantics({**sem, "blueprint_id": bid, "blueprint_inputs": inputs})
        ss = (samples_by_comp or {}).get(cid)
        if ss:
            runtime_by_comp[cid] = _prune_runtime(
                build_y_runtime_semantic_context(ss, blueprint_id=bid, blueprint_inputs=inputs, canonical_semantics=sem),
                sem,
            )

    payload = {
        "schema_version": Y_OUTPUT_SCHEMA_VERSION,
        "prompt_version": Y_PROMPT_VERSION,
        "component_count": view.get("component_count"),
        "components": [
            {
                "component_id": c.get("component_id"),
                "blueprint_id": c.get("blueprint_id"),
                "scene": c.get("scene"),
                "inputs": c.get("blueprint_inputs") or {},
            }
            for c in (view.get("components") or [])
        ],
        "canonical_semantics": per_bp,
        "runtime_context": runtime_by_comp or None,
        "entity_catalog": view.get("entity_catalog") or [],
        "candidate_action_domain": view.get("candidate_action_domain") or {},
    }
    return _dump_prompt_payload(payload)

def estimate_tokens(text: str) -> int:

    return max(1, len(text) // 4)

def audit_prompt_fields(prompt_text: str) -> dict[str, Any]:

    forbidden = (
        "formal_b0",
        "semantic_b0",
        "execution_status",
        "behavior_target",
        "action_path_signature",
        "branch_id",
        "expected_service",
        "expected_action",
        "selected_runtime_path",
        "yaml_trigger_id",
        "conflict_label",
        "repair_actions",
    )
    lower = prompt_text.lower()
    hits = [f for f in forbidden if f in lower]
    return {
        "B0_FIELD_IN_LLM_INPUT": sum(1 for h in hits if "b0" in h or "execution_status" in h or "yaml_trigger" in h),
        "CONSTRUCTION_ANSWER_FIELD_IN_LLM_INPUT": sum(
            1 for h in hits if h in ("behavior_target", "action_path_signature", "branch_id", "expected_service", "expected_action")
        ),
        "forbidden_hits": hits,
    }
