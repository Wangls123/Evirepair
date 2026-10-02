from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class OpenHABItem:
    name: str
    item_type: str
    label: str = ""
    group: str = ""
    initial_state: str | float | None = None

    entity_id: str = ""

@dataclass
class OpenHABRule:
    uid: str
    name: str
    component_id: str
    scene_type: str
    trigger_item: str
    trigger_state: str
    actions: list[dict[str, Any]]
    enabled: bool = True
    description: str = ""
    intent_actions: list[dict[str, Any]] | None = None
    conditions: list[Any] = field(default_factory=list)

@dataclass
class OpenHABScenario:
    scenario_id: str
    seed: int
    conflict_type: str
    items: list[OpenHABItem]
    rules: list[OpenHABRule]
    trigger_sequence: list[dict[str, Any]]
    hidden_ground_truth: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "conflict_type": self.conflict_type,
            "items": [item.__dict__ for item in self.items],
            "rules": [rule.__dict__ for rule in self.rules],
            "trigger_sequence": self.trigger_sequence,
            "hidden_ground_truth": self.hidden_ground_truth,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenHABScenario:
        return cls(
            scenario_id=str(data["scenario_id"]),
            seed=int(data.get("seed") or 0),
            conflict_type=str(data.get("conflict_type") or "opposing_light"),
            items=[OpenHABItem(**i) for i in data.get("items") or []],
            rules=[OpenHABRule(**r) for r in data.get("rules") or []],
            trigger_sequence=list(data.get("trigger_sequence") or []),
            hidden_ground_truth=dict(data.get("hidden_ground_truth") or {}),
            metadata=dict(data.get("metadata") or {}),
        )
