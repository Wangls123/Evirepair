from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class WalkContext:
    inputs: dict[str, Any]
    trigger_context: dict[str, Any]
    observed: dict[str, Any]
    control_flow_decisions: list[dict[str, Any]] = field(default_factory=list)
    parallel_group_id: str | None = None
    repeat_iteration: int | None = None
    repeat_item: Any = None
    blocked_future: bool = False
    abstain_reason: str | None = None
    abstain_missing: list[str] = field(default_factory=list)
    time_condition_required: bool = False
    output_contract: str = "v3"
    branch_resolutions: list[dict[str, Any]] = field(default_factory=list)
    raw_yaml: str | None = None
    fired_trigger_ids: set[str] = field(default_factory=set)
