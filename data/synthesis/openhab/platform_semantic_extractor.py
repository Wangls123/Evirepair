from __future__ import annotations

from pathlib import Path
from typing import Any

from smarthome_mdf.openhab_validation.openhab_dsl_parser import (
    infer_item_semantics,
    parse_openhab_directory,
)
from smarthome_mdf.openhab_validation.platform_semantic_model import (
    PlatformRuleSemantic,
    PlatformSemanticModel,
)
from smarthome_mdf.openhab_validation.types import OpenHABScenario

def extract_from_scenario(scenario: OpenHABScenario) -> PlatformSemanticModel:

    items = {}
    for item in scenario.items:
        items[item.name] = infer_item_semantics(
            item.name,
            item.item_type,
            label=item.label,
            group=item.group,
        )

    rules: dict[str, PlatformRuleSemantic] = {}
    for rule in scenario.rules:
        trigger = {
            "type": "item_state",
            "item": rule.trigger_item,
            "state": str(rule.trigger_state).upper(),
        }
        actions = []
        dependencies: list[str] = [rule.trigger_item]
        for act in rule.actions:
            atype = act.get("type", "sendCommand")
            if atype == "sendCommand":
                actions.append(
                    {
                        "type": "sendCommand",
                        "item": act["item"],
                        "command": str(act["command"]).upper(),
                    }
                )
                dependencies.append(str(act["item"]))
            elif atype == "postUpdate":
                actions.append(
                    {
                        "type": "postUpdate",
                        "item": act["item"],
                        "state": str(act.get("state") or act.get("command", "")).upper(),
                    }
                )
                dependencies.append(str(act["item"]))
        rules[rule.uid] = PlatformRuleSemantic(
            uid=rule.uid,
            name=rule.name,
            trigger=trigger,
            conditions=[],
            actions=actions,
            dependencies=sorted(set(dependencies)),
            enabled=rule.enabled,
            source="scenario_json",
        )

    return PlatformSemanticModel(
        platform="openhab",
        items=items,
        rules=rules,
        thing_bindings={},
        config_bindings={},
        extraction_sources=[f"scenario:{scenario.scenario_id}"],
    )

def extract_from_openhab_conf(conf_dir: Path) -> PlatformSemanticModel:

    parsed = parse_openhab_directory(conf_dir)
    return PlatformSemanticModel(
        platform="openhab",
        items=parsed["items"],
        rules=parsed["rules"],
        thing_bindings=parsed["thing_bindings"],
        config_bindings=parsed["config_bindings"],
        extraction_sources=parsed["extraction_sources"],
    )

def extract_platform_semantics(
    *,
    scenario: OpenHABScenario | None = None,
    conf_dir: Path | None = None,
) -> PlatformSemanticModel:

    if scenario is None and conf_dir is None:
        raise ValueError("extract_platform_semantics requires scenario and/or conf_dir")

    base: PlatformSemanticModel | None = None
    if conf_dir is not None and conf_dir.is_dir():
        base = extract_from_openhab_conf(conf_dir)

    if scenario is not None:
        sc_model = extract_from_scenario(scenario)
        if base is None:
            return sc_model
        merged_items = dict(base.items)
        merged_items.update(sc_model.items)
        merged_rules = dict(base.rules)
        merged_rules.update(sc_model.rules)
        return PlatformSemanticModel(
            platform="openhab",
            items=merged_items,
            rules=merged_rules,
            thing_bindings=base.thing_bindings,
            config_bindings=base.config_bindings,
            extraction_sources=base.extraction_sources + sc_model.extraction_sources,
        )

    assert base is not None
    return base

def scenario_to_openhab_files(scenario: OpenHABScenario, out_dir: Path) -> Path:

    out_dir.mkdir(parents=True, exist_ok=True)
    items_dir = out_dir / "items"
    rules_dir = out_dir / "rules"
    items_dir.mkdir(exist_ok=True)
    rules_dir.mkdir(exist_ok=True)

    lines = []
    for item in scenario.items:
        label = f' "{item.label}"' if item.label else ""
        group = f" ({item.group})" if item.group else ""
        lines.append(f"{item.item_type} {item.name}{label}{group}")
    (items_dir / f"{scenario.scenario_id}.items").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rule_blocks = []
    for rule in scenario.rules:
        when = f"Item {rule.trigger_item} changed to {rule.trigger_state}"
        then_lines = []
        for act in rule.actions:
            if act.get("type") == "sendCommand":
                then_lines.append(f"    {act['item']}.sendCommand({act['command']})")
            elif act.get("type") == "postUpdate":
                then_lines.append(f"    {act['item']}.postUpdate({act.get('state', act.get('command'))})")
        block = f'rule "{rule.name}"\nwhen\n    {when}\nthen\n' + "\n".join(then_lines) + "\nend\n"
        rule_blocks.append(block)
    (rules_dir / f"{scenario.scenario_id}.rules").write_text("\n".join(rule_blocks), encoding="utf-8")

    return out_dir
