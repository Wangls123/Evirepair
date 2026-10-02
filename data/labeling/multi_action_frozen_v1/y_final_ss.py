from __future__ import annotations

import hashlib
import json
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.action_type_classifier import (
    CLASSIFIER_VERSION,
    classify_payload,
)
from smarthome_mdf.multi_action_frozen_v1.y_v6_ss import (
    Y_ALLOWED_FIELDS,
    Y_V6_SCHEMA_VERSION,
    audit_y_v6_payload,
    build_y_v6_payload,
)

Y_FINAL_PROMPT_VERSION = "final_gold_y"
Y_FINAL_SCHEMA_VERSION = Y_V6_SCHEMA_VERSION

FINAL_STATE_SYSTEM_PROMPT = """You label the correct smart-home automation behavior from the current scene state and automation semantics only.

This sample is pre-classified as STATE_ACTION. Do not reclassify it as an event/log/notify problem.

Task:
Decide whether the automation should execute one device/helper state change now, or execute no action.

Focus on: current state, target state, state transition.

Rules:
- If the target device/helper already has the target state, choose NO_ACTION.
- If trigger, condition, observation, and automation semantics require a real state change now, choose ACTIONS.
- Do NOT treat a trigger alone as sufficient for ACTIONS.
- Do NOT treat a blueprint-declared action alone as sufficient for ACTIONS.
- ACTIONS: exactly one action. NO_ACTION: zero actions.
- Do not return UNKNOWN, UNCERTAIN, or multi-action.
- Ignore baseline execution, B0, repair, and previous labels. They are not in the input.

Return JSON only.

NO_ACTION:
{"decision":"NO_ACTION","actions":[]}

ACTIONS:
{"decision":"ACTIONS","actions":[{"service":"...","target":"...","data":{},"component_origin":"c0"}]}

Examples:
1. light.kitchen=off, declared light.turn_off → {"decision":"NO_ACTION","actions":[]}
2. motion on, night, light.kitchen=off, declared light.turn_on → {"decision":"ACTIONS","actions":[{"service":"light.turn_on","target":"light.kitchen","data":{},"component_origin":"c0"}]}
3. light.kitchen=on, declared light.turn_on → {"decision":"NO_ACTION","actions":[]}
4. window open, climate still cool, declared climate.turn_off → {"decision":"ACTIONS","actions":[{"service":"climate.turn_off","target":"climate.living_room","data":{},"component_origin":"c0"}]}
5. helper already off, declared input_boolean.turn_off → {"decision":"NO_ACTION","actions":[]}
"""

FINAL_EVENT_SYSTEM_PROMPT = """You label the correct smart-home automation behavior from the current scene state and automation semantics only.

This sample is pre-classified as EVENT_ACTION. Do not reclassify it as a device state-change problem.

Task:
Decide whether the automation should produce one event/effect now (notify, log, record, sheet, persistent notification), or execute no action.

Focus on: trigger occurrence, condition satisfaction, automation semantics, event effect.

The declared effect service is given after the input JSON as declared_effect_service.

Rules:
- The goal is to produce an event result, not to change a persistent light/climate/switch state.
- Do NOT choose NO_ACTION because no device state would change.
- Do NOT output light.*, climate.*, switch.*, cover.*, fan.*, or scene.* on this sample.
- If observation/entity_observations show the relevant event (motion, occupancy, door, window, power, camera/person, security sensor, scheduled tick) AND conditions/control-flow select this effect, choose ACTIONS with declared_effect_service (or another event service from candidate_services).
- Prefer declared_effect_service when it is in candidate_services.
- system_log.write, logbook.log, and many notify.* services are targetless — omit target.
- If the event did not occur, or conditions/control-flow do not select this effect now, choose NO_ACTION.
- Do NOT treat a trigger alone as sufficient for ACTIONS.
- Do NOT treat a blueprint-declared action alone as sufficient for ACTIONS.
- ACTIONS: exactly one action. NO_ACTION: zero actions.
- Do not return UNKNOWN, UNCERTAIN, or multi-action.
- Ignore baseline execution, B0, repair, and previous labels. They are not in the input.

Return JSON only.

NO_ACTION:
{"decision":"NO_ACTION","actions":[]}

ACTIONS:
{"decision":"ACTIONS","actions":[{"service":"...","data":{},"component_origin":"c0"}]}

Examples:
1. window opened, declared notify.mobile_app → {"decision":"ACTIONS","actions":[{"service":"notify.mobile_app","data":{},"component_origin":"c0"}]}
2. motion detected, declared system_log.write → {"decision":"ACTIONS","actions":[{"service":"system_log.write","data":{},"component_origin":"c0"}]}
3. scheduled report time now, declared google_sheets.append_sheet → {"decision":"ACTIONS","actions":[{"service":"google_sheets.append_sheet","data":{},"component_origin":"c0"}]}
4. door opened, declared persistent_notification.create → {"decision":"ACTIONS","actions":[{"service":"persistent_notification.create","data":{},"component_origin":"c0"}]}
5. no security event in observation, blueprint lists notify → {"decision":"NO_ACTION","actions":[]}
6. report time not reached, declared google_sheets.append_sheet → {"decision":"NO_ACTION","actions":[]}
"""

def build_y_final_payload(sample: dict[str, Any], view: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = build_y_v6_payload(sample, view)
    payload["prompt_version"] = Y_FINAL_PROMPT_VERSION
    payload["schema_version"] = Y_FINAL_SCHEMA_VERSION
    return payload

def build_y_final_user_prompt(view: dict[str, Any], sample: dict[str, Any] | None = None) -> str:
    if sample is None:
        sample = {
            "observed": (view.get("evidence") or {}).get("observed") or {},
            "entity_observations": (view.get("evidence") or {}).get("entity_observations") or [],
            "scene_type": view.get("scene_type"),
        }
    payload = build_y_final_payload(sample, view)
    classified = classify_payload(payload)
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        body
        + "\n\nCLASSIFIER\n"
        + f"action_type={classified['action_type']}\n"
        + f"declared_effect_service={classified.get('declared_effect_service') or ''}\n"
        + f"classifier_version={CLASSIFIER_VERSION}\n"
    )

def select_system_prompt(action_type: str) -> str:
    if action_type == "EVENT_ACTION":
        return FINAL_EVENT_SYSTEM_PROMPT
    return FINAL_STATE_SYSTEM_PROMPT

def y_final_prompt_hash() -> str:
    blob = (
        FINAL_STATE_SYSTEM_PROMPT
        + FINAL_EVENT_SYSTEM_PROMPT
        + Y_FINAL_PROMPT_VERSION
        + CLASSIFIER_VERSION
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

def audit_y_final_payload(payload: dict[str, Any], prompt_text: str) -> dict[str, Any]:
    scanned = (
        prompt_text.replace(Y_FINAL_PROMPT_VERSION, "")
        .replace(CLASSIFIER_VERSION, "")
        .replace("frozen_independent_y_v6", "")
    )
    report = audit_y_v6_payload(payload, scanned)
    report["allowed_fields"] = list(Y_ALLOWED_FIELDS)
    report["prompt_version"] = Y_FINAL_PROMPT_VERSION
    return report
