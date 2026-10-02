from __future__ import annotations

from typing import Any

SECURITY_TRIGGER_BLUEPRINTS = frozenset(
    {
        "security_osam_sensor_alert",
        "security_ikea_myggbett",
        "security_window_dynamic_wait",
    }
)

EXECUTED_ACTION = "EXECUTED_ACTION"
EXECUTED_NO_ACTION = "EXECUTED_NO_ACTION"
CONSERVATIVE_NO_ACTION = "CONSERVATIVE_NO_ACTION"

TRUE_GAP_MARKERS = (
    "EXECUTOR_IMPLEMENTATION_GAP",
    "unsupported_blueprint",
    "not_implemented",
)

def _diag_text(oracle: dict) -> str:
    parts = [
        str(oracle.get("label_status") or ""),
        str(oracle.get("failure_reason") or ""),
        " ".join(str(x) for x in (oracle.get("failed_conditions") or [])),
        " ".join(str(x) for x in (oracle.get("missing_inputs") or [])),
        " ".join(str(x) for x in (oracle.get("unsupported_constructs") or [])),
    ]
    return " ".join(parts).lower()

def classify_diagnostic_reason(oracle: dict, *, blueprint_id: str = "") -> str | None:

    bp = str(blueprint_id or oracle.get("blueprint_id") or "")
    text = _diag_text(oracle)
    label = str(oracle.get("label_status") or "")

    if oracle.get("executor_implementation_gap"):
        return "EXECUTOR_IMPLEMENTATION_GAP"

    if any(g in text for g in TRUE_GAP_MARKERS):

        unsupported = oracle.get("unsupported_constructs") or []
        if unsupported and "unsupported_blueprint" in text:
            return "UNSUPPORTED_RUNTIME_SEMANTIC"
        return "EXECUTOR_IMPLEMENTATION_GAP"

    if oracle.get("blocked_future") or "wait_or_delay" in text or "blocked_future" in text:
        return "SNAPSHOT_FUTURE_UNRESOLVED"

    if "wait_for_trigger" in text or "dead_zone_not_elapsed" in text:
        return "SNAPSHOT_FUTURE_UNRESOLVED"

    if "template" in text and ("fail" in text or "unresolved" in text):
        return "TEMPLATE_EVALUATION_FAILURE"

    if any(k in text for k in ("entity_binding", "bound_entity", "missing_blueprint_inputs", "light_target")):
        return "ENTITY_BINDING_MISSING"

    if bp == "presence_automation_state":
        return "MISSING_RUNTIME_EVIDENCE"
    if bp in SECURITY_TRIGGER_BLUEPRINTS:
        return "MISSING_TRIGGER_CONTEXT"
    if bp == "climate_window_restore":
        return "MISSING_RUNTIME_EVIDENCE"

    if any(k in text for k in ("yaml_trigger_id", "trigger_not_matched", "trigger_id", "trigger_context")):
        if "security" in bp or "notification" in bp:
            return "MISSING_TRIGGER_CONTEXT"

    if any(
        k in text
        for k in (
            "person_state",
            "unoccupied_duration",
            "state_duration",
            "current_power",
            "above_threshold_duration",
            "illuminance",
            "local_time",
            "insufficient_event_evidence",
        )
    ):
        return "MISSING_RUNTIME_EVIDENCE"

    if label in ("insufficient_event_evidence", "underdetermined") or oracle.get("missing_inputs"):
        if bp.startswith("security") or "notification" in bp:
            return "MISSING_TRIGGER_CONTEXT"
        return "MISSING_RUNTIME_EVIDENCE"

    if "unsupported" in text:

        if any(k in text for k in ("wait", "delay", "repeat", "snapshot", "future")):
            return "SNAPSHOT_FUTURE_UNRESOLVED"
        return "UNSUPPORTED_RUNTIME_SEMANTIC"

    return "OTHER"

def _was_v3_abstain(oracle: dict) -> bool:
    internal = oracle.get("internal_formal_b0_status") or oracle.get("formal_b0_status")
    if internal == "ABSTAIN":
        return True
    label = str(oracle.get("label_status") or "")
    missing = oracle.get("missing_inputs") or []
    if label in ("underdetermined", "insufficient_event_evidence") or missing:
        return True
    unsupported = oracle.get("unsupported_constructs") or []
    if unsupported and not oracle.get("trigger_matched"):
        return True
    return False

def _is_deterministic_no_action(oracle: dict) -> bool:
    if _was_v3_abstain(oracle):
        return False
    actions = oracle.get("actions") or []
    if actions:
        return False
    if not oracle.get("trigger_matched"):
        return False
    failed = oracle.get("failed_conditions") or []
    if failed:
        return True
    if oracle.get("conditions_passed") is False:
        return True
    label = str(oracle.get("label_status") or "")
    if label == "grounded" and oracle.get("trigger_matched"):
        return True
    return bool(oracle.get("trigger_matched"))

def detect_executor_implementation_gap(oracle: dict, *, blueprint_id: str = "") -> bool:
    reason = classify_diagnostic_reason(oracle, blueprint_id=blueprint_id)
    if reason == "EXECUTOR_IMPLEMENTATION_GAP":
        return True
    if oracle.get("execution_status") in ("EXECUTION_ERROR", "ADAPTER_ERROR"):
        return True
    unsupported = oracle.get("unsupported_constructs") or []
    for u in unsupported:
        us = str(u).lower()
        if "not_implemented" in us or "executor_gap" in us:
            return True
    text = _diag_text(oracle)
    if "unsupported_blueprint" in text and not unsupported:
        return True
    return any(g in text for g in TRUE_GAP_MARKERS if g != "unsupported_blueprint")

def apply_v4_output_contract(oracle: dict, *, blueprint_id: str = "") -> dict[str, Any]:

    out = dict(oracle)
    bp = str(blueprint_id or out.get("blueprint_id") or "")
    actions = list(out.get("actions") or [])

    internal_status = out.get("formal_b0_status")
    out["internal_formal_b0_status"] = internal_status
    out["diagnostic_details"] = {
        "missing_inputs": list(out.get("missing_inputs") or []),
        "failed_conditions": list(out.get("failed_conditions") or []),
        "label_status": out.get("label_status"),
        "unsupported_constructs": list(out.get("unsupported_constructs") or []),
        "control_flow_decisions": list(out.get("control_flow_decisions") or []),
    }

    gap = detect_executor_implementation_gap(out, blueprint_id=bp)
    out["executor_implementation_gap"] = gap

    if actions:
        out["formal_b0_status"] = "SUCCESS_ACTION"
        out["execution_resolution"] = EXECUTED_ACTION
        out["diagnostic_reason"] = None
        out["actions"] = actions
        return out

    if _is_deterministic_no_action(out):
        out["formal_b0_status"] = "SUCCESS_NO_ACTION"
        out["execution_resolution"] = EXECUTED_NO_ACTION
        out["diagnostic_reason"] = None
        out["actions"] = []
        return out

    reason = classify_diagnostic_reason(out, blueprint_id=bp)
    if reason == "EXECUTOR_IMPLEMENTATION_GAP":
        out["formal_b0_status"] = "SUCCESS_NO_ACTION"
        out["execution_resolution"] = CONSERVATIVE_NO_ACTION
        out["diagnostic_reason"] = reason
    elif _was_v3_abstain(out) or out.get("blocked_future"):
        out["formal_b0_status"] = "SUCCESS_NO_ACTION"
        out["execution_resolution"] = CONSERVATIVE_NO_ACTION
        out["diagnostic_reason"] = reason or "OTHER"
    else:
        out["formal_b0_status"] = "SUCCESS_NO_ACTION"
        out["execution_resolution"] = CONSERVATIVE_NO_ACTION
        out["diagnostic_reason"] = reason or "OTHER"

    out["actions"] = []
    return out
