from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from eval_ma_evidence_attribution import (
    apply_generic_gate,
    attribute_parent,
    compose_parent,
    ownership_for_component,
)
from eval_ma_evidence_sufficiency import apply_verifier
from eval_strict_v2_set_remove import apply_strict_v2, map_repaired, set_remove_v1
from openhab_multirule.validation.action_adapter import (
    openhab_command_to_token,
    token_to_openhab_command,
    tokens_from_native_actions,
)
from openhab_multirule.validation.adapter_execution import merge_b0_repair_commands
from openhab_multirule.validation.e2e.grounding import ground_repair_plan
from openhab_multirule.validation.final_benchmark_generator import (
    FinalBenchmarkGenerator,
)
from openhab_multirule.validation.openhab_simulator import OpenHABSimulator
from openhab_multirule.validation.openhab_ss_bridge import (
    ha_action_to_openhab_token,
    item_to_entity,
    openhab_token_to_ha_action,
)
from openhab_multirule.validation.repair_eval.metrics import (
    aggregate_track,
    classify_failure,
    false_repair_flags,
    obligation_satisfaction,
    partial_repair_bucket,
    repaired_conflict_actions,
    semantic_success,
)
from openhab_multirule.validation.repair_eval.semantic_reference import load_references
from openhab_multirule.validation.types import OpenHABRule, OpenHABScenario
from smarthome_mdf.multi_action_frozen_v1.action_type_classifier import classify_services
from smarthome_mdf.ss_trhr_repair.actions import compact_action, normalize_action
from smarthome_mdf.ss_trhr_repair.add import run_add
from smarthome_mdf.ss_trhr_repair.modify import run_modify
from smarthome_mdf.ss_trhr_repair.pipeline import _illegal
from smarthome_mdf.ss_trhr_repair.remove import run_remove

DATASET = ROOT / "data" / "openhab" / "benchmarks"
REFS = ROOT / "data" / "openhab" / "references" / "openhab_semantic_reference.jsonl"
OUT = ROOT / "results" / "openhab_multirule_evirepair.json"
METHOD = "EviRepair Frozen Repair (ss_trhr + Strict-v2)"

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def _norm_state(v: Any) -> str:
    return str(v or "").strip().lower()

def observation_from_states(sc: OpenHABScenario, states: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs: dict[str, Any] = {"platform": "openhab"}
    ents: list[dict[str, Any]] = []
    for item in sc.items:
        ent = item_to_entity(item)
        raw = states.get(item.name, item.initial_state)
        st = _norm_state(raw)
        domain = ent.split(".", 1)[0]
        ents.append({"entity_id": ent, "domain": domain, "state": st, "attributes": {}})
        name = item.name.lower()

        if "window" in name:
            obs["window_state"] = "open" if st in {"open", "on", "true", "1"} else "closed"
        if any(k in name for k in ("lux", "illum", "brightness")):
            try:
                obs["illuminance_lux"] = float(raw)
                obs["illuminance"] = obs["illuminance_lux"]
            except (TypeError, ValueError):
                pass
        if any(k in name for k in ("motion", "presence", "occup")):
            obs["motion_state"] = st
            obs["motion"] = st
        if "door" in name:
            obs["door_state"] = st
        if item.item_type == "Number" and "temp" in name:
            try:
                obs["temperature"] = float(raw)
            except (TypeError, ValueError):
                pass

        if "light_state" in name:
            obs["light_state"] = st
        if "hvac_mode" in name:
            obs["hvac_mode"] = st
        if "quiet_hours" in name:
            obs["quiet_hours"] = st
        if "silent_period" in name:
            obs["silent_period"] = st
        if "camera_presence" in name:
            obs["camera_presence"] = st
        if "person_detected" in name:
            obs["person_detected"] = st
        if "human_count" in name:
            try:
                obs["human_count"] = float(raw)
            except (TypeError, ValueError):
                obs["human_count"] = st
        if "current_power_w" in name:
            try:
                obs["current_power_w"] = float(raw)
            except (TypeError, ValueError):
                obs["current_power_w"] = st

    return obs, ents

def ha_acts_from_tokens(tokens: list[str], sc: OpenHABScenario) -> list[dict[str, Any]]:
    imap = {i.name: i for i in sc.items}
    out: list[dict[str, Any]] = []
    for tok in tokens:
        parsed = token_to_openhab_command(tok)
        if parsed:
            item_name, cmd = parsed
            item = imap.get(item_name)
            if item is not None and item.item_type == "HVAC" and cmd.upper().startswith("MODE_"):
                mode = cmd.split("_", 1)[1].lower()
                act = compact_action(
                    {
                        "service": "climate.set_hvac_mode",
                        "target_entity": item_to_entity(item),
                        "parameters": {"hvac_mode": mode},
                    }
                )
                if act:
                    out.append(act)
                continue
        ha = openhab_token_to_ha_action(tok, imap)
        act = normalize_action(
            {
                "service": ha.split("|")[0] if "|" in ha else ha,
                "target_entity": item_to_entity(imap[token_item(tok)]) if token_item(tok) in imap else None,
            }
        )
        if not act:

            svc = ha if "." in ha and not ha.startswith("openhab.") else ""
            if not svc:
                continue
            item = token_item(tok)
            act = compact_action(
                {
                    "service": svc if svc.startswith(("light.", "switch.", "cover.", "climate.", "notify.")) else _svc_from_item(sc, item, tok),
                    "target_entity": item_to_entity(imap[item]) if item in imap else None,
                    "parameters": {},
                }
            )
        if act:
            out.append(act)
    return out

def token_item(tok: str) -> str:
    parts = str(tok).split(".")
    if len(parts) >= 4 and parts[0] == "openhab" and parts[1] == "item":
        return ".".join(parts[2:-1]) if len(parts) > 4 else parts[2]
    return ""

def _svc_from_item(sc: OpenHABScenario, item_name: str, tok: str) -> str:
    imap = {i.name: i for i in sc.items}
    item = imap.get(item_name)
    cmd = str(tok).rsplit(".", 1)[-1].lower()
    if item is None:
        return f"switch.turn_{cmd}" if cmd in {"on", "off"} else "switch.turn_on"
    ent = item_to_entity(item)
    domain = ent.split(".", 1)[0]
    if cmd == "on":
        return f"{domain}.turn_on"
    if cmd == "off":
        return f"{domain}.turn_off"
    return f"{domain}.set"

def declared_services(rule: OpenHABRule, sc: OpenHABScenario) -> list[str]:

    imap = {i.name: i for i in sc.items}
    acts = rule.actions
    svcs: list[str] = []
    for act in acts or []:
        if act.get("type") != "sendCommand":
            continue
        item = str(act.get("item") or "")
        cmd = str(act.get("command") or "").lower()
        if item in imap:
            domain = item_to_entity(imap[item]).split(".", 1)[0]
            if cmd in {"on", "off"}:
                svcs.append(f"{domain}.turn_{cmd}")
            else:
                svcs.append(f"{domain}.set")
    seen: set[str] = set()
    out: list[str] = []
    for s in svcs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def make_ctx(
    sc: OpenHABScenario,
    rule: OpenHABRule,
    b0_act: dict[str, Any] | None,
    obs: dict[str, Any],
    ents: list[dict[str, Any]],
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:

    if contract and contract.get("blueprint_services"):

        blueprint_svcs = [str(s) for s in contract.get("blueprint_services") or [] if s]
        classified = classify_services(blueprint_svcs)
        svcs = [str(s) for s in (classified.get("services") or blueprint_svcs)]
        for s in contract.get("candidate_services") or []:
            if s and str(s) not in svcs:
                svcs.append(str(s))
        scene = str(contract.get("scene_type") or rule.scene_type)
        catalog = list(contract.get("entity_catalog") or [])
        declared = str(classified.get("declared_effect_service") or "")
    elif contract:
        svcs = [str(s) for s in (contract.get("candidate_services") or []) if s]
        scene = str(contract.get("scene_type") or rule.scene_type)
        catalog = list(contract.get("entity_catalog") or [])
        classified = classify_services(svcs)
        declared = str(classified.get("declared_effect_service") or (svcs[0] if svcs else ""))
    else:
        svcs = declared_services(rule, sc)
        if b0_act and b0_act.get("service") and str(b0_act["service"]) not in svcs:
            svcs.append(str(b0_act["service"]))
        scene = rule.scene_type
        catalog = [{"entity_id": e["entity_id"]} for e in ents]
        classified = classify_services(svcs)
        declared = str(classified.get("declared_effect_service") or (svcs[0] if svcs else ""))
    payload = {
        "scenario": scene,
        "observation": obs,
        "entity_observations": ents,
        "trigger": [{"item": rule.trigger_item, "state": rule.trigger_state}],
        "condition": list(getattr(rule, "conditions", None) or []),
        "blueprint": f"openhab.{scene}",
        "bindings": {"entity_catalog": catalog},
        "automation_specification": {"services": svcs},
        "intended_action_definition": {"candidate_services": svcs},
    }
    return {
        "sample_id": f"{sc.scenario_id}:{rule.component_id}",
        "scene_type": scene,
        "b0_decision": "ACTION" if b0_act else "NO_ACTION",
        "b0_action": compact_action(b0_act),
        "payload": payload,
        "action_type": classified.get("action_type") or "STATE_ACTION",
        "declared_effect_service": declared,
        "candidate_services": svcs,
        "classifier_reason": classified.get("reason"),
        "contract_source": "explicit_contract" if contract else "rule_actions",
    }

def frozen_repair_one(ctx: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    original = compact_action(ctx.get("b0_action"))
    repaired = original
    reason = "KEEP_B0"
    if original:
        r = run_remove(ctx, original, llm=None)
        if r.get("changed"):
            repaired = r.get("action_out")
            reason = str(r.get("reason") or "REMOVE")
        else:
            reason = str(r.get("reason") or reason)
            m = run_modify(ctx, original)
            if m.get("changed"):
                repaired = m.get("action_out")
                reason = str(m.get("reason") or "MODIFY")
            else:
                reason = str(m.get("reason") or reason)
    else:
        a = run_add(ctx, llm=None)
        if a.get("changed"):
            repaired = a.get("action_out")
            reason = str(a.get("reason") or "ADD")
        else:
            reason = str(a.get("reason") or reason)
    repaired = compact_action(repaired) if repaired else None
    if _illegal(repaired, ctx, original):
        repaired = original
        reason = "ROLLBACK"

    strict, kinds = apply_strict_v2(ctx, repaired)
    if "hvac_r1" in kinds:
        reason = "HVAC_R1"
    repaired = compact_action(strict) if strict else None
    return repaired, reason

def act_to_oh_token(act: dict[str, Any] | None, sc: OpenHABScenario) -> str | None:
    if not act:
        return None
    if str(act.get("service") or "") == "climate.set_hvac_mode":
        mode = str((act.get("parameters") or {}).get("hvac_mode") or "").strip().lower()
        target = str(act.get("target_entity") or "")
        if not mode or not target:
            return None
        for item in sc.items:
            if item.item_type == "HVAC" and item_to_entity(item) == target:
                return openhab_command_to_token(item.name, f"MODE_{mode}")
        return None
    imap = {i.name: i for i in sc.items}
    ha = f"{act.get('service')}|{act.get('target_entity') or ''}"
    tok = ha_action_to_openhab_token(ha, imap)
    if tok.startswith("openhab.item."):
        return tok
    return None

def assign_b0(rule: OpenHABRule, b0_acts: list[dict[str, Any]], sc: OpenHABScenario) -> list[dict[str, Any]]:

    targets = set()
    acts = rule.actions
    imap = {i.name: i for i in sc.items}
    for a in acts or []:
        if a.get("type") == "sendCommand" and a.get("item") in imap:
            targets.add(item_to_entity(imap[str(a["item"])]))
    matched = [a for a in b0_acts if str(a.get("target_entity") or "") in targets]
    return matched if matched else []

def _acts_to_tokens(acts: list[dict[str, Any]], sc: OpenHABScenario) -> list[str]:
    tokens: list[str] = []
    for act in acts:
        tok = act_to_oh_token(act, sc)
        if tok:
            tokens.append(tok)
    return tokens

def _openhab_component(
    rule: OpenHABRule,
    sc: OpenHABScenario,
    ctx: dict[str, Any],
    obs: dict[str, Any],
    ents: list[dict[str, Any]],
    pre: dict[str, str],
) -> dict[str, Any]:

    imap = {i.name: i for i in sc.items}
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    names = [rule.trigger_item]
    for act in rule.actions or []:
        if act.get("item"):
            names.append(str(act["item"]))
    for name in names:
        item = imap.get(name)
        if item is None:
            continue
        eid = item_to_entity(item)
        if eid in seen:
            continue
        seen.add(eid)
        records.append(
            {"original_entity": eid, "parent_entity": eid, "component_local_entity": eid}
        )
    declared = str(ctx.get("declared_effect_service") or "")
    triggered = str(pre.get(rule.trigger_item, "")).upper() == str(rule.trigger_state).upper()
    return {
        "component_id": rule.component_id,
        "scene": rule.scene_type,
        "component_action_schema": {"service": declared},
        "entity_binding": {
            "records": records,
            "component_entity_map": {eid: eid for eid in seen},
        },
        "runtime_state": {
            "observed": {k: v for k, v in obs.items() if k != "platform"},
            "entity_observations": ents,
            "system_state": {"behavior_target": declared},
        },
        "formal_b0_component": {
            "execution_trace": {
                "trigger_evaluation": {"trigger_matched": triggered},
                "condition_evaluation": {"conditions_passed": None},
                "event_type": None,
            }
        },
    }

def _parent_gate_verifier(
    rules: list[OpenHABRule],
    sc: OpenHABScenario,
    ctx_by_id: dict[str, dict[str, Any]],
    obs: dict[str, Any],
    ents: list[dict[str, Any]],
    pre: dict[str, str],
    acts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:

    comps = [
        _openhab_component(rule, sc, ctx_by_id[rule.component_id], obs, ents, pre)
        for rule in rules
        if rule.component_id in ctx_by_id
    ]
    ctxs = {c["component_id"]: ctx_by_id[c["component_id"]] for c in comps}
    _atoms, meta = attribute_parent(comps, ctxs)
    owns = [
        ownership_for_component(
            c,
            meta["per_comp_atoms"].get(c["component_id"]) or [],
            meta["contracts"].get(c["component_id"]) or [],
        )
        for c in comps
    ]
    owns_by_id = {o["component_id"]: o for o in owns}
    composition = compose_parent(comps, owns)
    gated, gate_trace = apply_generic_gate(acts, owns_by_id, composition)
    comps_by_id = {c["component_id"]: c for c in comps}
    verified, verifier_trace = apply_verifier(gated, comps_by_id, owns_by_id, composition)
    info = {
        "gate": gate_trace,
        "verifier": verifier_trace,
        "excluded_components": composition.get("excluded_components"),
        "parent_intended_capabilities": composition.get("parent_intended_capabilities"),
        "origins_after_gate": [str(a.get("component_origin") or "") for a in gated],
        "origins_after_verifier": [str(a.get("component_origin") or "") for a in verified],
    }
    return verified, info

def observation_for_rule(
    sc: OpenHABScenario,
    pre: dict[str, str],
    rule: OpenHABRule,
    fallback_obs: dict[str, Any],
    fallback_ents: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    spec = ((sc.metadata or {}).get("component_evidence") or {}).get(rule.component_id)
    if not spec:
        return fallback_obs, fallback_ents
    names = set(spec.get("item_names") or [])
    names.add(rule.trigger_item)
    for act in rule.actions or []:
        if act.get("item"):
            names.add(str(act["item"]))
    subset = replace(sc, items=[item for item in sc.items if item.name in names])
    return observation_from_states(subset, pre)

def pre_action_states(sc: OpenHABScenario) -> dict[str, str]:

    sim = OpenHABSimulator(sc)
    sim.reset()
    sim.apply_triggers()
    return dict(sim.item_states)

def _send_indexes(actions: list[dict[str, Any]]) -> list[int]:
    return [i for i, act in enumerate(actions) if act.get("type") == "sendCommand"]

def _copy_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(act) for act in actions]

def _delete_index(actions: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    return [dict(act) for i, act in enumerate(actions) if i != index]

def _command_at_slot(pred: dict[str, Any] | None, sc: OpenHABScenario) -> tuple[str, str] | None:
    if not pred:
        return None
    tok = act_to_oh_token(pred, sc)
    if not tok:
        return None
    return token_to_openhab_command(tok)

def _replace_index(actions: list[dict[str, Any]], index: int, item: str, command: str) -> list[dict[str, Any]]:
    out = _copy_actions(actions)
    slot = dict(out[index])
    slot["type"] = "sendCommand"
    slot["item"] = item
    slot["command"] = command
    slot.pop("state", None)
    out[index] = slot
    return out

def scenario_with_actions(sc: OpenHABScenario, updates: dict[str, list[dict[str, Any]]]) -> OpenHABScenario:

    rules = []
    for rule in sc.rules:
        if rule.component_id in updates:
            rules.append(replace(rule, actions=updates[rule.component_id]))
        else:
            rules.append(rule)
    return replace(sc, rules=rules)

def repair_scenario(
    sc: OpenHABScenario,
    before_sim: Any,
    stage_out: dict[str, Any] | None = None,
    contracts: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], Counter]:

    executed = list(before_sim.executed_actions)
    b0_tokens = tokens_from_native_actions(executed)
    pre = pre_action_states(sc)
    obs, ents = observation_from_states(sc, pre)
    rules = [r for r in sc.rules if r.enabled]
    reasons: Counter = Counter()
    repaired_acts: list[dict[str, Any]] = []
    ctx_by_id: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []

    if not rules:
        return [], b0_tokens, reasons

    for rule in rules:
        actions = _copy_actions(list(rule.actions or []))
        send_ix = _send_indexes(actions)
        own = [a for a in executed if a.get("component_id") == rule.component_id]
        record: dict[str, Any] = {
            "component_id": rule.component_id,
            "slot_index": send_ix[0] if send_ix else None,
            "original_actions": actions,
            "status": "unchanged",
            "reason": "",
        }
        rule_obs, rule_ents = observation_for_rule(sc, pre, rule, obs, ents)
        if not own:
            ctx = make_ctx(sc, rule, None, rule_obs, rule_ents, (contracts or {}).get(rule.component_id))
            ctx_by_id[rule.component_id] = ctx
            pred, reason = frozen_repair_one(ctx)
            reasons[reason] += 1
            record["reason"] = reason
            record["pred"] = compact_action(pred) if pred else None
            if pred is None:
                record["status"] = "no_execution_kept"
                record["written_actions"] = actions
            elif not actions:
                parsed = _command_at_slot(pred, sc)
                if parsed is None:
                    record["status"] = "add_unmapped_unsupported"
                    record["written_actions"] = actions
                else:
                    item, command = parsed
                    record["status"] = "add_on_empty_action_list"
                    record["written_actions"] = [{"type": "sendCommand", "item": item, "command": command}]
                    pred = dict(pred)
                    pred["component_origin"] = rule.component_id
                    repaired_acts.append(pred)
            else:

                record["status"] = "add_unsupported_existing_actions"
                record["written_actions"] = actions
            pending.append(record)
            continue

        slot = send_ix[0] if send_ix else None
        head = actions[slot] if slot is not None else None
        aligned = (
            head is not None
            and str(own[0].get("item") or "") == str(head.get("item") or "")
            and str(own[0].get("command") or "").upper() == str(head.get("command") or "").upper()
        )
        if not aligned:
            record["status"] = "unsupported_slot_mismatch"
            record["reason"] = "FIRST_EXECUTED_COMMAND_NOT_AT_SLOT_INDEX"
            record["written_actions"] = actions
            reasons[record["reason"]] += 1
            pending.append(record)
            continue

        first = ha_acts_from_tokens(tokens_from_native_actions(own[:1]), sc)
        b0_act = first[0] if first else None
        ctx = make_ctx(sc, rule, b0_act, rule_obs, rule_ents, (contracts or {}).get(rule.component_id))
        ctx_by_id[rule.component_id] = ctx
        pred, reason = frozen_repair_one(ctx)
        reasons[reason] += 1
        record["reason"] = reason
        record["slot_index"] = slot
        if pred is None:
            record["status"] = "remove_slot"
            record["written_actions"] = _delete_index(actions, slot)
        else:
            pred = dict(pred)
            pred["component_origin"] = rule.component_id
            record["pred"] = pred
            repaired_acts.append(pred)
            record["status"] = "slot_pending_parent"
            record["written_actions"] = actions
        pending.append(record)

    parent_info = None
    origins_component = [str(a.get("component_origin") or "") for a in repaired_acts]
    pre_gate_tokens = _acts_to_tokens(repaired_acts, sc)
    origins_set_remove = list(origins_component)
    stored_bindings = dict((sc.metadata or {}).get("component_bindings") or {})
    if stored_bindings:

        mapped_acts = []
        for act in repaired_acts:
            origin = str(act.get("component_origin") or "")
            mapped = map_repaired(dict(act), stored_bindings.get(origin) or {})
            if mapped:
                mapped["component_origin"] = origin
                mapped_acts.append(mapped)
        repaired_acts = mapped_acts
    if len(rules) > 1 and repaired_acts:
        if stored_bindings:
            bind_all = {r.component_id: dict(stored_bindings.get(r.component_id) or {}) for r in rules}
        else:
            bind_all = {r.component_id: {} for r in rules}
        repaired_acts, remove_trace = set_remove_v1(repaired_acts, bind_all)
        origins_set_remove = [str(a.get("component_origin") or "") for a in repaired_acts]
        pre_gate_tokens = _acts_to_tokens(repaired_acts, sc)
        repaired_acts, parent_info = _parent_gate_verifier(
            rules, sc, ctx_by_id, obs, ents, pre, repaired_acts
        )
        parent_info["set_remove_before_gate"] = remove_trace
        parent_info["origins_after_set_remove"] = origins_set_remove
    survivors = {str(a.get("component_origin") or "") for a in repaired_acts}
    written: dict[str, list[dict[str, Any]]] = {}
    for record in pending:
        origin = record["component_id"]
        actions = record["original_actions"]
        slot = record["slot_index"]
        pred = record.get("pred")
        if record["status"] == "add_on_empty_action_list" and origin not in survivors:
            record["status"] = "parent_removed_add"
            record["written_actions"] = []
        if record["status"] == "slot_pending_parent":
            if origin not in survivors:
                record["status"] = "parent_removed_slot"
                record["written_actions"] = _delete_index(actions, slot)
            else:
                parsed = _command_at_slot(pred, sc)
                if parsed is None:
                    record["status"] = "modify_unmapped_unsupported"
                    record["written_actions"] = actions
                else:
                    item, command = parsed
                    record["written_actions"] = _replace_index(actions, slot, item, command)
                    same = (
                        str(actions[slot].get("item")) == item
                        and str(actions[slot].get("command") or "").upper() == command.upper()
                    )
                    record["status"] = "slot_kept" if same else "slot_replaced"
        written[origin] = record["written_actions"]

    patched = scenario_with_actions(sc, written)
    patched_run = OpenHABSimulator(patched).run_baseline()
    tokens = [str(a["repair_token"]) for a in patched_run.executed_actions if a.get("repair_token")]
    if stage_out is not None:
        stage_out["pre_gate_tokens"] = pre_gate_tokens
        stage_out["parent"] = parent_info
        stage_out["operator_attempts"] = sum(reasons.values())
        stage_out["writeback"] = [
            {
                "component_id": r["component_id"],
                "slot_index": r["slot_index"],
                "status": r["status"],
                "reason": r["reason"],
                "pred": r.get("pred"),
                "original_actions": r["original_actions"],
                "written_actions": r["written_actions"],
            }
            for r in pending
        ]
        stage_out["unsupported"] = [
            r["component_id"] for r in pending if "unsupported" in str(r["status"])
        ]
        stage_out["patched_states"] = dict(patched_run.item_states)
        stage_out["equipment_effects"] = dict(getattr(patched_run, "equipment_effects", {}) or {})
        stage_out["before_states"] = dict(getattr(before_sim, "item_states", {}) or {})
        stage_out["before_equipment_effects"] = dict(getattr(before_sim, "equipment_effects", {}) or {})
        stage_out["verification"] = "in_process_simulator"
        stage_out["patched_component_ids"] = [str(a.get("component_id") or "") for a in patched_run.executed_actions]
        stage_out["patched_actions"] = [
            {
                "component_id": str(a.get("component_id") or ""),
                "item": a.get("item"),
                "command": a.get("command"),
                "repair_token": a.get("repair_token"),
            }
            for a in patched_run.executed_actions
        ]
        stage_out["evidence_time"] = "after_triggers_before_any_command"
        stage_out["evidence_snapshot"] = pre
        stage_out["origins_after_component"] = origins_component
        stage_out["origins_after_set_remove"] = origins_set_remove
        stage_out["origins_after_gate"] = list(origins_set_remove if not parent_info else parent_info.get("origins_after_gate") or [])
        stage_out["origins_after_verifier"] = list(origins_set_remove if not parent_info else parent_info.get("origins_after_verifier") or [])
    return tokens, b0_tokens, reasons

def score_row(sc: OpenHABScenario, ref: dict[str, Any], before_sim: Any, tokens: list[str], b0: list[str]) -> dict[str, Any]:
    grounded = ground_repair_plan(tokens)
    native = grounded.get("native_plan") or []
    if tokens:
        native = merge_b0_repair_commands(b0, tokens) or native
    after_sim = OpenHABSimulator(sc).run_with_commands([(str(a), str(b)) for a, b in native])
    before_states = dict(before_sim.item_states)
    after_states = dict(after_sim.item_states)
    before_sem = semantic_success(before_states, ref) if ref else False
    after_sem = semantic_success(after_states, ref) if ref else False
    carr_info = repaired_conflict_actions(before_states, after_states, ref) if ref else {
        "total_conflict_actions": 0,
        "repaired_conflict_actions": 0,
        "carr": 0.0,
        "by_type": {},
    }
    fr = false_repair_flags(
        before_states=before_states,
        after_states=after_states,
        reference=ref,
        repaired_tokens=tokens,
        b0_tokens=b0,
    ) if ref else {"false_repair": False, "flags": []}
    complete = bool(
        ref.get("repair_required")
        and not ref.get("ambiguity")
        and after_sem
        and carr_info["carr"] >= 1.0
        and not fr["false_repair"]
    )
    n_rules = len([r for r in sc.rules if r.enabled])
    track = "ss" if n_rules <= 1 else "ma"
    row = {
        "scenario_id": sc.scenario_id,
        "execution_label": "OPENHAB_SIMULATOR_SEMANTICS",
        "scenario_type": (sc.metadata or {}).get("scenario_type"),
        "scenario_family": (sc.metadata or {}).get("scenario_family"),
        "complexity": (sc.metadata or {}).get("complexity"),
        "conflict_type": sc.conflict_type,
        "repair_mode": track,
        "repair_method": METHOD,
        "n_rules": n_rules,
        "runtime_success": True,
        "grounding_success": bool(native) or not tokens,
        "failure_class": None,
        "repair_required": bool(ref.get("repair_required")),
        "ambiguity": bool(ref.get("ambiguity")),
        "before_semantic_success": float(before_sem),
        "after_semantic_success": float(after_sem),
        "semantic_repair_gain": float(after_sem) - float(before_sem),
        "carr": carr_info["carr"],
        "total_conflict_actions": carr_info["total_conflict_actions"],
        "repaired_conflict_actions": carr_info["repaired_conflict_actions"],
        "by_type": carr_info["by_type"],
        "partial_bucket": partial_repair_bucket(carr_info["carr"]),
        "complete_repair": complete,
        "conflict_resolution_rate": round(
            carr_info["repaired_conflict_actions"] / max(carr_info["total_conflict_actions"], 1), 4
        ),
        "false_repair": fr["false_repair"],
        "false_repair_flags": fr["flags"],
        "new_conflict": False,
        "b0_tokens": b0,
        "repaired_tokens": tokens,
        "native_plan": native,
        "before_obligation": obligation_satisfaction(before_states, ref) if ref else {},
        "after_obligation": obligation_satisfaction(after_states, ref) if ref else {},
    }
    row["failure_taxonomy"] = classify_failure(row)
    return row

def main() -> None:
    print(f"[{utc_now()}] load OpenHAB corpus + live_split + refs", flush=True)
    ids = json.loads((DATASET / "splits" / "live_split.json").read_text(encoding="utf-8"))["scenario_ids"]
    refs = load_references(REFS)
    corpus = {s.scenario_id: s for s in FinalBenchmarkGenerator.load_corpus(DATASET)}
    print(f"[{utc_now()}] corpus={len(corpus)} live_split={len(ids)} refs={len(refs)}", flush=True)

    rows: list[dict[str, Any]] = []
    reasons = Counter()
    for i, sid in enumerate(ids, 1):
        sc = corpus[sid]
        if len([r for r in sc.rules if r.enabled]) <= 1:
            continue
        sc.hidden_ground_truth = {}
        before = OpenHABSimulator(sc).run_baseline()
        tokens, b0, rs = repair_scenario(sc, before)
        reasons.update(rs)
        rows.append(score_row(sc, refs.get(sid) or {}, before, tokens, b0))
        if i % 80 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}", flush=True)

    ma = [r for r in rows if r.get("repair_mode") == "ma"]
    block = aggregate_track(ma)
    summary = {
        "what_this_file_is": "OpenHAB multi-rule EviRepair migration result. Single-rule scenarios are excluded.",
        "track": "multi_rule",
        "method": METHOD,
        "execution_label": "OPENHAB_SIMULATOR_SEMANTICS",
        "generated": utc_now(),
        "split": "live_split scenarios with more than one enabled rule",
        "benchmark": "data/openhab/benchmarks",
        "reference": "data/openhab/references/openhab_semantic_reference.jsonl",
        "n": len(ma),
        "metrics": block,
        "operator_reasons": dict(reasons),
        "note": "Frozen Repair operators + Strict-v2 HVAC R1/payload drop. MA parent uses set_remove_v1 only (HA evidence-ownership gate/verifier not claimed on OpenHAB).",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    dump(OUT, summary)
    print(json.dumps({"multi_rule": block, "n": len(ma)}, indent=2), flush=True)
    print("wrote", OUT, flush=True)

if __name__ == "__main__":
    main()
