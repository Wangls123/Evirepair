from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ItemCapability:

    name: str
    operations: tuple[str, ...]
    state_values: tuple[str, ...]
    value_type: str = "string"

@dataclass
class PlatformItemSemantic:
    entity_name: str
    item_type: str
    label: str = ""
    group: str = ""
    capabilities: list[ItemCapability] = field(default_factory=list)
    supported_commands: list[str] = field(default_factory=list)
    state_model: dict[str, Any] = field(default_factory=dict)
    channel_uid: str | None = None
    thing_uid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "item_type": self.item_type,
            "label": self.label,
            "group": self.group,
            "capabilities": [
                {"name": c.name, "operations": list(c.operations), "state_values": list(c.state_values)}
                for c in self.capabilities
            ],
            "supported_commands": self.supported_commands,
            "state_model": self.state_model,
            "channel_uid": self.channel_uid,
            "thing_uid": self.thing_uid,
        }

@dataclass
class PlatformRuleSemantic:
    uid: str
    name: str
    trigger: dict[str, Any]
    conditions: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    dependencies: list[str] = field(default_factory=list)
    enabled: bool = True
    source: str = "extracted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "trigger": self.trigger,
            "conditions": self.conditions,
            "actions": self.actions,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "source": self.source,
        }

@dataclass
class PlatformSemanticModel:

    platform: str
    items: dict[str, PlatformItemSemantic]
    rules: dict[str, PlatformRuleSemantic]
    thing_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    config_bindings: dict[str, Any] = field(default_factory=dict)
    extraction_sources: list[str] = field(default_factory=list)

    def get_item(self, name: str) -> PlatformItemSemantic | None:
        return self.items.get(name)

    def item_supports_command(self, entity: str, command: str) -> bool:
        item = self.items.get(entity)
        if not item:
            return False
        return str(command).upper() in {c.upper() for c in item.supported_commands}

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "rules": {k: v.to_dict() for k, v in self.rules.items()},
            "thing_bindings": self.thing_bindings,
            "config_bindings": self.config_bindings,
            "extraction_sources": self.extraction_sources,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformSemanticModel:
        items = {}
        for k, v in (data.get("items") or {}).items():
            caps = [
                ItemCapability(
                    name=c["name"],
                    operations=tuple(c.get("operations") or ()),
                    state_values=tuple(c.get("state_values") or ()),
                )
                for c in v.get("capabilities") or []
            ]
            items[k] = PlatformItemSemantic(
                entity_name=v["entity_name"],
                item_type=v["item_type"],
                label=v.get("label") or "",
                group=v.get("group") or "",
                capabilities=caps,
                supported_commands=list(v.get("supported_commands") or []),
                state_model=dict(v.get("state_model") or {}),
                channel_uid=v.get("channel_uid"),
                thing_uid=v.get("thing_uid"),
            )
        rules = {}
        for k, v in (data.get("rules") or {}).items():
            rules[k] = PlatformRuleSemantic(
                uid=v["uid"],
                name=v.get("name") or k,
                trigger=dict(v.get("trigger") or {}),
                conditions=list(v.get("conditions") or []),
                actions=list(v.get("actions") or []),
                dependencies=list(v.get("dependencies") or []),
                enabled=bool(v.get("enabled", True)),
                source=str(v.get("source") or "extracted"),
            )
        return cls(
            platform=str(data.get("platform") or "openhab"),
            items=items,
            rules=rules,
            thing_bindings=dict(data.get("thing_bindings") or {}),
            config_bindings=dict(data.get("config_bindings") or {}),
            extraction_sources=list(data.get("extraction_sources") or []),
        )
