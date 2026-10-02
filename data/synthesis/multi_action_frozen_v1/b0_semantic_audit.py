from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import (
    blueprint_yaml_bridge,
    execute_frozen_parent_b0,
    frozen_to_b0_parent,
)
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import audit_instance_reconstruction
from smarthome_mdf.multi_action_vnext.parent_executor import _execute_component_blueprint
from smarthome_mdf.multi_action_vnext.action_schema import from_raw_action

def _classify_component(
    *,
    oracle: dict,
    semantic_count: int,
    service_level: list[str],
    action_path_sig: str,
) -> str:
    if action_path_sig.startswith("valid."):
        return "LEGITIMATE_NO_ACTION"
    if semantic_count > 0:
        return "CANONICAL_ACTION_PRESENT"
    if service_level and semantic_count == 0:
        return "LEGACY_DEBUG_ONLY_ACTION"
    failed = oracle.get("failed_conditions") or []
    if oracle.get("conditions_passed") is False and not oracle.get("actions"):
        return "LEGITIMATE_NO_ACTION"
    if oracle.get("label_status") == "underdetermined" and "unsupported_blueprint" in str(failed):
        return "EXECUTOR_SEMANTIC_MISMATCH"
    if not oracle.get("actions") and not service_level:
        return "LEGITIMATE_NO_ACTION"
    return "ADAPTER_RUNTIME_MISMATCH"

def audit_parent_b0(
    ma: dict[str, Any],
    frozen_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent = frozen_to_b0_parent(ma, frozen_index)
    components_audit: list[dict[str, Any]] = []
    instance_invalid = 0

    with blueprint_yaml_bridge():
        parent_out = execute_frozen_parent_b0(ma, frozen_index)
        for ctx in parent.get("component_runtime_contexts") or []:
            comp = next(
                (c for c in parent.get("automation_components") or [] if c.get("component_id") == ctx.get("component_id")),
                {},
            )
            iid = str(comp.get("automation_instance_id") or "")
            bp_id = str(comp.get("blueprint_id") or "")
            scene = str(ctx.get("scene_type") or "")
            inst_audit = audit_instance_reconstruction(iid, blueprint_id=bp_id, scene=scene)
            if inst_audit.get("invalid"):
                instance_invalid += 1

            raw, semantic, trace = _execute_component_blueprint(parent, ctx)
            ss = frozen_index.get(
                next(
                    (c.get("single_scene_sample_id") for c in ma.get("components") or [] if c.get("component_id") == ctx.get("component_id")),
                    "",
                ),
                {},
            )
            action_sig = str((ss.get("synthesis_metadata") or {}).get("action_path_signature") or "")
            iso = {
                "scene_type": scene,
                "observed": ctx.get("observed"),
                "blueprint_binding": {"blueprint_id": bp_id},
            }
            from smarthome_mdf.multi_action_vnext.parent_executor import build_component_runtime_sample, _rule_engine_service_view

            iso_full = build_component_runtime_sample(parent, ctx)
            rule_sv = _rule_engine_service_view(iso_full)
            oracle = trace if isinstance(trace, dict) else {}
            if not oracle.get("executed_branch"):
                oracle = {"actions": raw, "failed_conditions": trace.get("failed_conditions") if isinstance(trace, dict) else []}

            cls = _classify_component(
                oracle={"actions": raw, "conditions_passed": bool(raw), "failed_conditions": []},
                semantic_count=len(semantic),
                service_level=rule_sv,
                action_path_sig=action_sig,
            )
            components_audit.append(
                {
                    "component_id": ctx.get("component_id"),
                    "blueprint_id": bp_id,
                    "scene": scene,
                    "classification": cls,
                    "direct_semantic_count": len(semantic),
                    "raw_action_count": len(raw),
                    "rule_engine_service_level": rule_sv,
                    "action_path_signature": action_sig,
                    "instance_audit": inst_audit,
                }
            )

    canonical = parent_out.get("canonical_b0", {}).get("semantic_actions") or []
    direct_total = sum(c["direct_semantic_count"] for c in components_audit)
    parent_empty = len(canonical) == 0
    direct_nonempty = direct_total > 0

    canonical_loss = 0
    if direct_nonempty and parent_empty:

        canonical_loss = 1

    for c in components_audit:
        if c["direct_semantic_count"] > 0 and c["classification"] == "LEGACY_DEBUG_ONLY_ACTION":
            c["classification"] = "CANONICAL_ACTION_PRESENT"

    return {
        "multi_action_id": ma.get("multi_action_id"),
        "components": components_audit,
        "parent_canonical_count": len(canonical),
        "direct_component_semantic_total": direct_total,
        "canonical_action_loss": canonical_loss,
        "B0_INSTANCE_RECONSTRUCTION_INVALID": instance_invalid,
        "service_level_view_source": "get_v3_b0_actions via _rule_engine_service_view (debug only, not canonical)",
    }

def run_b0_readiness_audit(
    corpus: list[dict[str, Any]],
    frozen_index: dict[str, Any],
    *,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    selected = corpus
    if sample_ids:
        ids = set(sample_ids)
        selected = [ma for ma in corpus if ma.get("multi_action_id") in ids]

    per_parent: list[dict] = []
    classifications = Counter()
    blueprint_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"action": 0, "no_action": 0})
    totals = {
        "total_parents": 0,
        "total_components": 0,
        "component_executor_no_action": 0,
        "component_executor_action": 0,
        "parent_canonical_empty": 0,
        "parent_canonical_non_empty": 0,
        "canonical_action_count": 0,
        "CANONICAL_ACTION_LOSS": 0,
        "B0_INSTANCE_RECONSTRUCTION_INVALID": 0,
    }

    for ma in selected:
        audit = audit_parent_b0(ma, frozen_index)
        per_parent.append(audit)
        totals["total_parents"] += 1
        totals["CANONICAL_ACTION_LOSS"] += audit.get("canonical_action_loss", 0)
        totals["B0_INSTANCE_RECONSTRUCTION_INVALID"] += audit.get("B0_INSTANCE_RECONSTRUCTION_INVALID", 0)
        pc = audit.get("parent_canonical_count", 0)
        totals["canonical_action_count"] += pc
        if pc:
            totals["parent_canonical_non_empty"] += 1
        else:
            totals["parent_canonical_empty"] += 1
        for comp in audit.get("components") or []:
            totals["total_components"] += 1
            classifications[comp.get("classification", "OTHER")] += 1
            bid = str(comp.get("blueprint_id") or "")
            if comp.get("direct_semantic_count", 0) > 0:
                totals["component_executor_action"] += 1
                blueprint_stats[bid]["action"] += 1
            else:
                totals["component_executor_no_action"] += 1
                blueprint_stats[bid]["no_action"] += 1

    return {
        "totals": totals,
        "classifications": dict(classifications),
        "per_blueprint": dict(blueprint_stats),
        "per_parent_sample": per_parent[:10],
        "intermediate_audit_note": (
            "Root cause of prior 30/30 empty canonical: execute_blueprint returned unsupported_blueprint "
            "for frozen blueprint IDs (non community.* prefix) while service_level_view came from "
            "independent get_v3_b0_actions rule-engine path."
        ),
    }

def write_audit_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "b0_semantic_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
