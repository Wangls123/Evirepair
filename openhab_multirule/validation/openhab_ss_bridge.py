from __future__ import annotations

import re
from typing import Any

from openhab_multirule.validation.action_adapter import openhab_command_to_token, token_to_openhab_command
from openhab_multirule.validation.types import OpenHABItem, OpenHABRule, OpenHABScenario

ITEM_TYPE_TO_DOMAIN = {
    "Switch": "switch",
    "Dimmer": "light",
    "Contact": "binary_sensor",
    "Number": "sensor",
    "String": "input_text",
    "Color": "light",
    "Player": "media_player",
    "Rollershutter": "cover",
    "Trigger": "event",

    "InputBoolean": "input_boolean",

    "HVAC": "climate",
}

DOMAIN_SERVICES = {
    "switch": {"on": "switch.turn_on", "off": "switch.turn_off"},
    "light": {"on": "light.turn_on", "off": "light.turn_off"},
    "input_boolean": {"on": "input_boolean.turn_on", "off": "input_boolean.turn_off"},
    "binary_sensor": {"open": "binary_sensor.update", "closed": "binary_sensor.update"},
    "cover": {"on": "cover.open", "off": "cover.close"},
    "sensor": {"on": "number.set_value", "off": "number.set_value"},
}

def _normalize_entity_id(item_name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(item_name)).lower()
    return s.strip("_") or "item"

def item_to_entity(item: OpenHABItem) -> str:
    explicit = str(getattr(item, "entity_id", "") or "")
    if explicit:
        return explicit
    domain = ITEM_TYPE_TO_DOMAIN.get(item.item_type, "switch")
    return f"{domain}.{_normalize_entity_id(item.name)}"

def openhab_token_to_ha_action(token: str, item_map: dict[str, OpenHABItem]) -> str:
    parsed = token_to_openhab_command(token)
    if not parsed:
        return token
    item_name, cmd = parsed
    item = item_map.get(item_name)
    if not item:
        return f"switch.turn_{cmd.lower()}" if cmd in ("ON", "OFF") else token
    domain = ITEM_TYPE_TO_DOMAIN.get(item.item_type, "switch")
    svc_map = DOMAIN_SERVICES.get(domain, DOMAIN_SERVICES["switch"])
    if item.item_type == "HVAC" and str(cmd).upper().startswith("MODE_"):
        return "climate.set_hvac_mode"
    if cmd == "ON":
        return svc_map.get("on", "switch.turn_on")
    if cmd == "OFF":
        return svc_map.get("off", "switch.turn_off")
    return f"{domain}.set"

def ha_action_to_openhab_token(action: str, item_map: dict[str, OpenHABItem]) -> str:

    a = str(action or "")
    if a.startswith("openhab.item."):
        return a
    entity = ""
    svc = a
    if "|" in a:
        svc, entity = a.split("|", 1)
    if svc == "climate.set_hvac_mode":

        return a
    elif a.count(".") >= 1 and a.split(".")[0] in ITEM_TYPE_TO_DOMAIN.values():
        entity = a
        svc = a

    ent_norm = entity.lower() if entity else ""
    for item in item_map.values():
        ie = item_to_entity(item).lower()

        if ent_norm and ie == ent_norm:
            cmd = "ON" if "turn_on" in svc or svc.endswith(".open") else "OFF"
            return openhab_command_to_token(item.name, cmd)

    dom = svc.split(".")[0] if "." in svc else ""
    cmd = "ON" if "turn_on" in svc else "OFF"
    for item in item_map.values():
        if ITEM_TYPE_TO_DOMAIN.get(item.item_type, "switch") == dom:
            return openhab_command_to_token(item.name, cmd)
    return a

def _entity_map(scenario: OpenHABScenario) -> dict[str, OpenHABItem]:
    return {i.name: i for i in scenario.items}

def project_sample_for_ss_v5(
    sample: dict[str, Any],
    scenario: OpenHABScenario,
) -> dict[str, Any]:

    imap = _entity_map(scenario)
    entities = [item_to_entity(i) for i in scenario.items]
    entity_obs = {}
    for item in scenario.items:
        ent = item_to_entity(item)
        init = item.initial_state
        entity_obs[ent] = str(init or "off").lower()

    projected = dict(sample)
    rt = dict(projected.get("runtime") or {})
    b0_oh = list(rt.get("b0_actions") or [])
    b0_ha = [openhab_token_to_ha_action(t, imap) for t in b0_oh]
    rt["b0_actions"] = b0_ha
    rt["openhab_b0_tokens"] = b0_oh
    projected["runtime"] = rt

    rules = [r for r in scenario.rules if r.enabled]
    scene_type = rules[0].scene_type if rules else "on_off_schedule"
    projected["scene_type"] = scene_type

    projected["blueprint_binding"] = {
        "entities": entities,
        "blueprint_inputs": {},
        "blueprint_id": f"openhab.{scene_type}",
    }
    projected["entity_observations"] = [
        {
            "entity_id": item_to_entity(item),
            "domain": ITEM_TYPE_TO_DOMAIN.get(item.item_type, "switch"),
            "state": str(item.initial_state or "off").lower(),
            "attributes": {},
        }
        for item in scenario.items
    ]
    projected["devices"] = [{"entity_id": e, "platform": "openhab"} for e in entities]

    traces = []
    for rule in rules:
        intent_tokens = []
        if rule.intent_actions:
            for act in rule.intent_actions:
                if act.get("type") == "sendCommand":
                    tok = openhab_command_to_token(str(act["item"]), str(act["command"]))
                    intent_tokens.append(openhab_token_to_ha_action(tok, imap))
        else:
            for act in rule.actions:
                if act.get("type") == "sendCommand":
                    tok = openhab_command_to_token(str(act["item"]), str(act["command"]))
                    intent_tokens.append(openhab_token_to_ha_action(tok, imap))
        traces.append(
            {
                "component_id": rule.component_id,
                "scene_type": rule.scene_type,
                "blueprint_id": f"openhab.{rule.scene_type}",
                "automation_instance_id": rule.uid,
                "local_expected_actions": intent_tokens,
                "local_generation_side_effect_actions": [],
                "executed_branch": "openhab_rule",
                "condition_results": True,
            }
        )
    projected["component_oracle_traces"] = traces

    if len(rules) == 1:
        r = rules[0]
        expected = []
        for act in (r.intent_actions or r.actions):
            if act.get("type") == "sendCommand":
                tok = openhab_command_to_token(str(act["item"]), str(act["command"]))
                expected.append(openhab_token_to_ha_action(tok, imap))
        projected["oracle_trace"] = {"expected_actions": expected, "side_effect_actions": []}

    meta = dict(projected.get("metadata") or {})
    meta["openhab_ss_bridge"] = True
    meta["platform"] = "openhab"
    projected["metadata"] = meta
    projected["platform"] = "openhab"
    return projected

def _tokens_from_executed_steps(repair_out: dict[str, Any], imap: dict[str, OpenHABItem]) -> list[str]:
    tokens: list[str] = []
    for step in repair_out.get("v4_executed_steps") or []:
        if not step.get("applied"):
            continue
        for key in ("new_action", "platform_action"):
            val = step.get(key)
            if val:
                oh = ha_action_to_openhab_token(str(val), imap)
                if oh.startswith("openhab.item."):
                    tokens.append(oh)
    return tokens

def demote_ss_v5_output(
    repair_out: dict[str, Any],
    scenario: OpenHABScenario,
) -> dict[str, Any]:

    imap = _entity_map(scenario)
    out = dict(repair_out)
    step_tokens = _tokens_from_executed_steps(out, imap)
    ha_tokens = list(out.get("repaired_actions") or [])
    oh_tokens = step_tokens or [ha_action_to_openhab_token(t, imap) for t in ha_tokens]
    out["repaired_actions"] = oh_tokens
    out["repaired_actions_ha_projection"] = ha_tokens
    steps = out.get("v4_executed_steps") or []
    applied = sum(1 for s in steps if s.get("applied"))
    abstain = bool(out.get("abstain")) or any(
        str(a.get("operator")) == "ABSTAIN" for a in (out.get("v4_atoms") or [])
    )
    coverage = out.get("obligation_coverage") or {}
    out["semantic_success"] = applied > 0 or bool(coverage.get("complete") or coverage.get("partial_success"))
    out["abstain"] = abstain and applied == 0
    return out
