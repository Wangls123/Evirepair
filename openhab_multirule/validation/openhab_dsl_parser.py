from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openhab_multirule.validation.capability_extractor import (
    extract_capability_for_item_type,
    supported_commands_for_item_type,
)
from openhab_multirule.validation.platform_semantic_model import (
    PlatformItemSemantic,
    PlatformRuleSemantic,
)

ITEM_TYPE_CAPABILITIES = {
    k: extract_capability_for_item_type(k) for k in (
        "Switch", "Dimmer", "Contact", "Number", "Rollershutter", "String", "Trigger"
    )
}

ITEM_LINE_RE = re.compile(
    r"^(?P<type>\w+)\s+(?P<name>[\w]+)\s*(?:\"(?P<label>[^\"]*)\")?\s*(?:\((?P<group>[^)]+)\))?",
    re.MULTILINE,
)

RULE_WHEN_RE = re.compile(
    r"Item\s+(?P<item>[\w]+)\s+(?:changed|received command|updated)\s+(?:to\s+)?(?P<state>[\w]+)",
    re.IGNORECASE,
)
SEND_COMMAND_RE = re.compile(
    r"sendCommand\s*\(\s*(?P<item>[\w]+)\s*,\s*(?P<cmd>[\w]+)\s*\)",
    re.IGNORECASE,
)
POST_UPDATE_RE = re.compile(
    r"postUpdate\s*\(\s*(?P<item>[\w]+)\s*,\s*(?P<state>[\w]+)\s*\)",
    re.IGNORECASE,
)
THING_UID_RE = re.compile(r"UID\s*=\s*(?P<uid>[\w:-]+)", re.IGNORECASE)
CHANNEL_ITEM_RE = re.compile(
    r"(?P<channel>[\w:-]+)\s*->\s*(?P<item>[\w]+)",
    re.IGNORECASE,
)

def infer_item_semantics(
    name: str,
    item_type: str,
    *,
    label: str = "",
    group: str = "",
    channel_uid: str | None = None,
    thing_uid: str | None = None,
) -> PlatformItemSemantic:

    cap = extract_capability_for_item_type(item_type)
    commands = supported_commands_for_item_type(item_type)
    return PlatformItemSemantic(
        entity_name=name,
        item_type=item_type,
        label=label,
        group=group,
        capabilities=[cap],
        supported_commands=commands,
        state_model={
            "type": item_type,
            "allowed_states": list(cap.state_values) if cap.state_values else None,
            "value_type": cap.value_type,
        },
        channel_uid=channel_uid,
        thing_uid=thing_uid,
    )

def _commands_for_capability(cap: Any) -> list[str]:
    return supported_commands_for_item_type(
        next((k for k, v in ITEM_TYPE_CAPABILITIES.items() if v.name == cap.name), "Switch")
    )

def parse_items_file(text: str, *, source: str = "") -> dict[str, PlatformItemSemantic]:
    items: dict[str, PlatformItemSemantic] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = ITEM_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        item_type = m.group("type")
        items[name] = infer_item_semantics(
            name,
            item_type,
            label=m.group("label") or "",
            group=m.group("group") or "",
        )
    return items

def parse_rules_file(text: str, *, source: str = "") -> dict[str, PlatformRuleSemantic]:
    rules: dict[str, PlatformRuleSemantic] = {}
    blocks = re.split(r"\n(?=rule\s+)", text, flags=re.IGNORECASE)
    for block in blocks:
        block = block.strip()
        if not block.lower().startswith("rule"):
            continue
        name_m = re.match(r"rule\s+\"(?P<name>[^\"]+)\"", block, re.IGNORECASE)
        if not name_m:
            continue
        name = name_m.group("name")
        uid = _slug_uid(name, source)
        when_m = re.search(r"when\s+(?P<when>.+?)(?:\nthen|\nend)", block, re.IGNORECASE | re.DOTALL)
        when_text = when_m.group("when").strip() if when_m else ""
        trigger = _parse_when_clause(when_text)
        conditions: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        dependencies: list[str] = []
        for sc in SEND_COMMAND_RE.finditer(block):
            item = sc.group("item")
            cmd = sc.group("cmd").upper()
            actions.append({"type": "sendCommand", "item": item, "command": cmd})
            dependencies.append(item)
        for pu in POST_UPDATE_RE.finditer(block):
            item = pu.group("item")
            state = pu.group("state").upper()
            actions.append({"type": "postUpdate", "item": item, "state": state})
            dependencies.append(item)
        if trigger.get("item"):
            dependencies.append(str(trigger["item"]))
        rules[uid] = PlatformRuleSemantic(
            uid=uid,
            name=name,
            trigger=trigger,
            conditions=conditions,
            actions=actions,
            dependencies=sorted(set(dependencies)),
            enabled=True,
            source=source or "rules_file",
        )
    return rules

def parse_things_file(text: str, *, source: str = "") -> tuple[dict[str, dict[str, Any]], dict[str, str]]:

    thing_bindings: dict[str, dict[str, Any]] = {}
    item_channels: dict[str, str] = {}
    current_thing: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("thing"):
            uid_m = THING_UID_RE.search(line)
            if uid_m:
                current_thing = uid_m.group("uid")
                thing_bindings[current_thing] = {"uid": current_thing, "source": source}
        ch_m = CHANNEL_ITEM_RE.search(line)
        if ch_m:
            item_channels[ch_m.group("item")] = ch_m.group("channel")
            if current_thing:
                thing_bindings.setdefault(current_thing, {})["channels"] = thing_bindings.get(current_thing, {}).get(
                    "channels", []
                )
                if isinstance(thing_bindings[current_thing].get("channels"), list):
                    thing_bindings[current_thing]["channels"].append(
                        {"channel": ch_m.group("channel"), "item": ch_m.group("item")}
                    )
    return thing_bindings, item_channels

def parse_config_dir(config_dir: Path) -> dict[str, Any]:

    bindings: dict[str, Any] = {}
    if not config_dir.is_dir():
        return bindings
    for cfg in config_dir.glob("*.cfg"):
        section = cfg.stem
        entries: dict[str, str] = {}
        for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                entries[k.strip()] = v.strip()
        if entries:
            bindings[section] = entries
    return bindings

def _parse_when_clause(when_text: str) -> dict[str, Any]:
    m = RULE_WHEN_RE.search(when_text)
    if m:
        return {
            "type": "item_state",
            "item": m.group("item"),
            "state": m.group("state").upper(),
            "raw": when_text,
        }
    return {"type": "unknown", "raw": when_text}

def _slug_uid(name: str, source: str) -> str:
    base = re.sub(r"[^\w]+", "_", name.lower()).strip("_")
    if source:
        src = Path(source).stem
        return f"{src}_{base}"
    return base

def parse_openhab_directory(conf_dir: Path) -> dict[str, Any]:

    items: dict[str, PlatformItemSemantic] = {}
    rules: dict[str, PlatformRuleSemantic] = {}
    thing_bindings: dict[str, dict[str, Any]] = {}
    item_channel_map: dict[str, str] = {}
    sources: list[str] = []

    items_dir = conf_dir / "items"
    if items_dir.is_dir():
        for path in sorted(items_dir.glob("*.items")):
            text = path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_items_file(text, source=str(path))
            items.update(parsed)
            sources.append(str(path))

    rules_dir = conf_dir / "rules"
    if rules_dir.is_dir():
        for path in sorted(rules_dir.glob("*.rules")):
            text = path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_rules_file(text, source=str(path))
            rules.update(parsed)
            sources.append(str(path))

    things_dir = conf_dir / "things"
    if things_dir.is_dir():
        for path in sorted(things_dir.glob("*.things")):
            text = path.read_text(encoding="utf-8", errors="replace")
            tb, ic = parse_things_file(text, source=str(path))
            thing_bindings.update(tb)
            item_channel_map.update(ic)
            sources.append(str(path))

    config_bindings = parse_config_dir(conf_dir / "services")
    for name, ch in item_channel_map.items():
        if name in items:
            items[name].channel_uid = ch

    return {
        "items": items,
        "rules": rules,
        "thing_bindings": thing_bindings,
        "config_bindings": config_bindings,
        "extraction_sources": sources,
    }
