from __future__ import annotations

from typing import Any

from smarthome_mdf.synthesis_v3.blueprint_parser import BlueprintSpec

def success(
    trigger_id: str,
    branch_id: str,
    actions: list[dict],
    spec: BlueprintSpec,
    decisions: list[dict],
) -> dict[str, Any]:
    return {
        "trigger_matched": True,
        "conditions_passed": True,
        "executed_branch": branch_id,
        "selected_runtime_path": branch_id,
        "selected_runtime_path_source": "runtime_execution",
        "failed_conditions": [],
        "missing_inputs": [],
        "unsupported_constructs": spec.unsupported_constructs,
        "actions": actions,
        "event_type": branch_id,
        "label_status": "grounded",
        "runtime_memory_after": None,
        "side_effect_actions": [],
        "trigger_id": trigger_id,
        "control_flow_decisions": decisions,
        "formal_b0_status": "SUCCESS_ACTION",
    }

def no_action(
    trigger_id: str,
    failed: list[str],
    spec: BlueprintSpec,
    decisions: list[dict],
) -> dict[str, Any]:
    return {
        "trigger_matched": True,
        "conditions_passed": False,
        "executed_branch": None,
        "selected_runtime_path": None,
        "selected_runtime_path_source": "runtime_execution",
        "failed_conditions": failed,
        "missing_inputs": [],
        "unsupported_constructs": spec.unsupported_constructs,
        "actions": [],
        "event_type": None,
        "label_status": "grounded",
        "runtime_memory_after": None,
        "side_effect_actions": [],
        "trigger_id": trigger_id,
        "control_flow_decisions": decisions,
        "formal_b0_status": "SUCCESS_NO_ACTION",
    }

def abstain(
    reason: str,
    missing: list[str],
    spec: BlueprintSpec,
    decisions: list[dict],
    *,
    trigger_matched: bool | None = None,
) -> dict[str, Any]:
    status = "insufficient_event_evidence" if "insufficient" in reason or "missing" in reason.lower() else "underdetermined"
    if trigger_matched is None:
        trigger_matched = "insufficient" not in reason and "missing" not in reason.lower()
    return {
        "trigger_matched": trigger_matched,
        "conditions_passed": False,
        "executed_branch": None,
        "selected_runtime_path": None,
        "selected_runtime_path_source": "runtime_execution",
        "failed_conditions": [reason] if reason else [],
        "missing_inputs": missing,
        "unsupported_constructs": spec.unsupported_constructs,
        "actions": [],
        "event_type": None,
        "label_status": status,
        "runtime_memory_after": None,
        "side_effect_actions": [],
        "trigger_id": "",
        "control_flow_decisions": decisions,
        "formal_b0_status": "ABSTAIN",
    }
