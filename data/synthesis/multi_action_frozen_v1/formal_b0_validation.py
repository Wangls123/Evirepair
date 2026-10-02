from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import audit_b0_information_flow
from smarthome_mdf.multi_action_frozen_v1.b0_blueprint_inventory_audit import (
    audit_blueprint_inventory,
    audit_construction_path_leakage,
    audit_runtime_projection_neutrality,
)
from smarthome_mdf.multi_action_frozen_v1.config import EXPECTED_MULTI_ACTION_FROZEN_SHA256
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import audit_instance_reconstruction
from smarthome_mdf.synthesis_v3.frozen_blueprint_executor_registry import ExecutionMode, classify_blueprint

FORMAL_BLUEPRINT_IDS = [
    "appliance_notifications_actions",
    "appliance_power_google_sheets",
    "appliance_power_state_detect",
    "camera_frigate_intelligent",
    "camera_frigate_vision_llm",
    "climate_hvac_auto_adjust",
    "climate_production_grade",
    "climate_smarter_thermostat",
    "climate_window_restore",
    "lighting_motion_advanced_v22",
    "lighting_motion_extended",
    "lighting_motion_maestro_48",
    "lighting_motion_sensor_advanced",
    "lighting_sensor_comprehensive",
    "presence_automation_state",
    "presence_better_thermostat",
    "presence_holiday_away_lighting",
    "security_contact_left_open",
    "security_ikea_myggbett",
    "security_osam_sensor_alert",
    "security_window_dynamic_wait",
]

REQUIRED_ACTION_FIELDS = ("service", "component_origin")
TARGET_REQUIRED_DOMAINS = frozenset({"light", "climate", "switch", "cover", "fan", "media_player", "lock"})

def load_b0_outputs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _service_requires_target(service: str) -> bool:
    if not service or "." not in service:
        return False
    domain = service.split(".", 1)[0]
    return domain in TARGET_REQUIRED_DOMAINS

def validate_canonical_action(action: dict[str, Any], component_ids: set[str]) -> list[str]:
    issues: list[str] = []
    if not action.get("service"):
        issues.append("missing_service")
    svc = str(action.get("service") or "")
    if _service_requires_target(svc) and not action.get("target_entity"):
        issues.append("missing_target_entity")
    co = action.get("component_origin")
    if co and str(co) not in component_ids:
        issues.append("invalid_component_origin")
    if action.get("execution_order") is not None:
        try:
            if int(action.get("execution_order")) < 1:
                issues.append("invalid_execution_order")
        except (TypeError, ValueError):
            issues.append("invalid_execution_order")
    if not isinstance(action.get("parameters"), dict):
        issues.append("parameters_not_dict")
    return issues

def validate_formal_b0_outputs(
    records: list[dict[str, Any]],
    input_corpus: list[dict[str, Any]],
    *,
    expected_count: int = 2400,
) -> dict[str, Any]:
    input_ids = [str(m.get("multi_action_id")) for m in input_corpus]
    input_id_set = set(input_ids)
    output_ids = [str(r.get("multi_action_id")) for r in records]
    output_id_set = set(output_ids)

    status_counts = Counter(r.get("execution_status") for r in records)
    error_statuses = {
        "EXECUTION_ERROR",
        "ADAPTER_ERROR",
        "UNSUPPORTED_BLUEPRINT",
        "YAML_RESOLUTION_ERROR",
        "INSTANCE_RESOLUTION_ERROR",
        "RUNTIME_PROJECTION_ERROR",
    }

    canonical_invalid = 0
    canonical_loss = 0
    component_total = 0
    component_action = 0
    component_no_action = 0
    component_error = 0
    per_blueprint: dict[str, dict[str, int]] = defaultdict(
        lambda: {"executions": 0, "action": 0, "no_action": 0, "error": 0}
    )
    trigger_true = 0
    trigger_false = 0
    condition_true = 0
    condition_false = 0
    parent_action_dist = Counter()
    comp_count_dist = Counter()

    instance_invalid = 0
    yaml_mismatch = 0

    for ma, rec in zip(
        sorted(input_corpus, key=lambda m: str(m.get("multi_action_id"))),
        sorted(records, key=lambda r: str(r.get("multi_action_id"))),
    ):
        if ma.get("multi_action_id") != rec.get("multi_action_id"):
            pass

    id_to_input = {str(m["multi_action_id"]): m for m in input_corpus}
    for rec in records:
        mid = str(rec.get("multi_action_id"))
        ma = id_to_input.get(mid, {})
        comp_ids = {str(c.get("component_id")) for c in ma.get("components") or []}
        comp_count_dist[len(ma.get("components") or [])] += 1

        st = rec.get("execution_status")
        semantic = (rec.get("canonical_b0") or {}).get("semantic_actions") or []
        if st in error_statuses:
            component_error += len(ma.get("components") or [])
        elif st == "SUCCESS_ACTIONS":
            parent_action_dist[len(semantic)] += 1
        elif st == "SUCCESS_NO_ACTION":
            parent_action_dist[0] += 1

        for act in semantic:
            issues = validate_canonical_action(act, comp_ids)
            if issues:
                canonical_invalid += 1

        per = rec.get("per_component_b0") or []
        direct = sum(len(c.get("semantic_actions") or []) for c in per)
        if direct > 0 and len(semantic) == 0 and st not in error_statuses:
            canonical_loss += 1

        debug_sv = (rec.get("compatibility_debug") or {}).get("service_level_view")
        if debug_sv and not semantic and st == "SUCCESS_ACTIONS":
            canonical_loss += 1

        for comp in ma.get("components") or []:
            component_total += 1
            bid = str(comp.get("blueprint_id") or "")
            if bid in per_blueprint or bid in FORMAL_BLUEPRINT_IDS:
                per_blueprint[bid]["executions"] += 1
            iid = str(comp.get("grounded_instance") or "")
            ia = audit_instance_reconstruction(
                iid, blueprint_id=bid, scene=str(comp.get("scene") or "")
            )
            if ia.get("invalid"):
                instance_invalid += 1

        for pc in per:
            bid = next(
                (
                    str(c.get("blueprint_id"))
                    for c in ma.get("components") or []
                    if str(c.get("component_id")) == str(pc.get("component_id"))
                ),
                "",
            )
            n = len(pc.get("semantic_actions") or [])
            if bid:
                if n:
                    per_blueprint[bid]["action"] += 1
                    component_action += 1
                else:
                    per_blueprint[bid]["no_action"] += 1
                    component_no_action += 1

        for tr in rec.get("execution_traces") or []:
            te = tr.get("trigger_evaluation") or {}
            if te.get("trigger_matched") is True:
                trigger_true += 1
            elif te.get("trigger_matched") is False:
                trigger_false += 1
            ce = tr.get("condition_evaluation") or {}
            if ce.get("conditions_passed") is True:
                condition_true += 1
            elif ce.get("conditions_passed") is False:
                condition_false += 1

    inventory = audit_blueprint_inventory()
    formal_bp_unsupported = sum(
        1 for bp in FORMAL_BLUEPRINT_IDS if classify_blueprint(bp) == ExecutionMode.UNSUPPORTED
    )

    flow = audit_b0_information_flow()
    constr = audit_construction_path_leakage()
    runtime = audit_runtime_projection_neutrality()

    missing_ids = sorted(input_id_set - output_id_set)
    extra_ids = sorted(output_id_set - input_id_set)
    duplicate_ids = [i for i, c in Counter(output_ids).items() if c > 1]

    gates = {
        "input_sample_count": len(input_corpus),
        "formal_b0_record_count": len(records),
        "missing_input_ids": len(missing_ids),
        "duplicate_b0_ids": len(duplicate_ids),
        "SCENE_FALLBACK_ONLY": inventory["SCENE_FALLBACK_ONLY"],
        "UNSUPPORTED_BLUEPRINT": formal_bp_unsupported,
        "EXECUTOR_NOT_IMPLEMENTED": formal_bp_unsupported,
        "B0_BLUEPRINT_YAML_MISMATCH": yaml_mismatch,
        "B0_INSTANCE_RECONSTRUCTION_INVALID": instance_invalid,
        "B0_RUNTIME_PROJECTION_DECISION_LEAKAGE": runtime["B0_RUNTIME_PROJECTION_DECISION_LEAKAGE"],
        "B0_CONSTRUCTION_PATH_LEAKAGE": constr["B0_CONSTRUCTION_PATH_LEAKAGE"],
        "CANONICAL_ACTION_SCHEMA_INVALID": canonical_invalid,
        "CANONICAL_ACTION_LOSS": canonical_loss,
        "B0_INFORMATION_FLOW_VIOLATION": flow["B0_INFORMATION_FLOW_VIOLATION"],
    }

    blockers = []
    if gates["input_sample_count"] != expected_count:
        blockers.append("INPUT_COUNT_MISMATCH")
    if gates["formal_b0_record_count"] != expected_count:
        blockers.append("OUTPUT_COUNT_MISMATCH")
    if missing_ids:
        blockers.append("MISSING_INPUT_IDS")
    if duplicate_ids:
        blockers.append("DUPLICATE_B0_IDS")
    if status_counts.get("EXECUTION_ERROR", 0) or status_counts.get("ADAPTER_ERROR", 0):
        blockers.append("EXECUTION_ERRORS")
    for key in (
        "SCENE_FALLBACK_ONLY",
        "UNSUPPORTED_BLUEPRINT",
        "EXECUTOR_NOT_IMPLEMENTED",
        "B0_INSTANCE_RECONSTRUCTION_INVALID",
        "B0_RUNTIME_PROJECTION_DECISION_LEAKAGE",
        "B0_CONSTRUCTION_PATH_LEAKAGE",
        "CANONICAL_ACTION_SCHEMA_INVALID",
        "CANONICAL_ACTION_LOSS",
        "B0_INFORMATION_FLOW_VIOLATION",
    ):
        if gates[key] > 0:
            blockers.append(key)

    verdict = "FORMAL_B0_FROZEN_V1_READY" if not blockers else "BLOCKED_BEFORE_FORMAL_B0_FREEZE"

    return {
        "verdict": verdict,
        "blockers": blockers,
        "gates": gates,
        "execution_status_counts": dict(status_counts),
        "parent_statistics": {
            "SUCCESS_ACTIONS": status_counts.get("SUCCESS_ACTIONS", 0),
            "SUCCESS_NO_ACTION": status_counts.get("SUCCESS_NO_ACTION", 0),
            "parent_action_count_distribution": dict(parent_action_dist),
            "component_count_distribution": dict(comp_count_dist),
        },
        "component_audit": {
            "total_component_executions": component_total,
            "component_action_producing": component_action,
            "component_legitimate_no_action": component_no_action,
            "component_execution_error": component_error,
            "trigger_true": trigger_true,
            "trigger_false": trigger_false,
            "condition_true": condition_true,
            "condition_false": condition_false,
        },
        "per_blueprint": {
            bp: {
                **stats,
                "executor_mode": (
                    "GENERIC_YAML"
                    if classify_blueprint(bp).value == "GENERIC_YAML_EXECUTOR"
                    else "BLUEPRINT_SPECIFIC"
                ),
            }
            for bp, stats in sorted(per_blueprint.items())
        },
        "missing_ids": missing_ids[:20],
        "duplicate_ids": duplicate_ids[:20],
        "multi_action_frozen_corpus_sha256": EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    }
