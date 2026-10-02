from __future__ import annotations

from typing import Any

from baselines.autotap.property_compiler import compile_ss_property

NEUTRAL_DELAY = 0

def _rule_from_component(comp_ir: dict[str, Any]) -> dict[str, Any]:
    trig = comp_ir.get("trigger") or []
    acts = comp_ir.get("actions") or []
    obs = comp_ir.get("observations") or {}
    return {
        "component_id": comp_ir.get("component_id") or "c0",
        "sample_id": comp_ir.get("id"),
        "scene_type": comp_ir.get("scene_type"),
        "blueprint_id": comp_ir.get("blueprint_id"),
        "triggers": trig,
        "conditions": comp_ir.get("conditions") or [],
        "actions": acts,
        "observations": obs,
        "entities": comp_ir.get("entities") or [],
        "delay_seconds": NEUTRAL_DELAY,
        "physical_transition": None,
        "note": "delay_seconds=0 is a documented neutral default; no invented latency",
    }

def compile_ma_property(parent_ir: dict[str, Any]) -> dict[str, Any]:

    if parent_ir.get("unsupported_input"):
        return {
            "applicable": False,
            "status": "UNSUPPORTED_CAPABILITY",
            "properties": [],
            "reason": "visual_or_llmvision_component",
        }
    props = []
    rules = []
    for comp in parent_ir.get("components") or []:
        rules.append(_rule_from_component(comp))
        sub = compile_ss_property(comp)
        for p in sub.get("properties") or []:
            p = dict(p)
            p["component_id"] = comp.get("component_id")
            props.append(p)
    if len(rules) < 2:
        return {
            "applicable": False,
            "status": "UNSUPPORTED_INPUT",
            "properties": [],
            "reason": "tapfixer_targets_interacting_rules_need_ge_2_components",
            "rules": rules,
        }
    if not props:
        return {
            "applicable": False,
            "status": "PROPERTY_UNAVAILABLE",
            "properties": [],
            "reason": "no_gold_free_correctness_property_for_parent",
            "rules": rules,
        }
    return {
        "applicable": True,
        "status": "APPLICABLE",
        "properties": props,
        "rules": rules,
        "reason": "compiled_from_component_observations_and_capabilities",
        "uses_invented_physical_params": False,
        "delay_policy": "zero_delay_neutral",
    }

def compile_ss_for_tapfixer(ir: dict[str, Any]) -> dict[str, Any]:

    from baselines.tapfixer.ss_property_compiler import compile_ss_property_tapfixer

    return compile_ss_property_tapfixer(ir)
