from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import blueprint_yaml_bridge, verify_yaml_resolution
from smarthome_mdf.multi_action_frozen_v1.config import (
    B0_ADAPTER_VERSION,
    FORMAL_B0_V2_PIPELINE_VERSION,
    FORMAL_B0_V4_PIPELINE_VERSION,
)
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import resolve_frozen_instance
from smarthome_mdf.multi_action_frozen_v1.frozen_runtime_projector import project_component_runtime
from smarthome_mdf.multi_action_vnext.blueprint_execution_context import extract_synthesis_audit_metadata
from smarthome_mdf.multi_action_vnext.parent_executor import execute_parent_b0

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _synthetic_parent_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    bb = sample.get("blueprint_binding") or {}
    snap = bb.get("grounded_instance") or {}
    bp_id = str(bb.get("blueprint_id") or "")
    scene = str(sample.get("scene_type") or snap.get("scene") or "")
    iid = str(bb.get("automation_instance_id") or snap.get("automation_instance_id") or "")
    inst = resolve_frozen_instance(iid, blueprint_id=bp_id, scene=scene)
    if snap.get("blueprint_inputs"):
        inst["blueprint_inputs"] = dict(snap["blueprint_inputs"])
        inst["bound_entities"] = list(snap.get("bound_entities") or bb.get("entities") or [])

    projected = project_component_runtime(
        scene=scene,
        single_scene_sample=sample,
        component_runtime={
            "observed": sample.get("observed") or {},
            "entity_observations": sample.get("entity_observations") or [],
            "system_state": sample.get("system_state") or {},
        },
        blueprint_inputs=dict(inst.get("blueprint_inputs") or {}),
    )
    synthesis_audit = extract_synthesis_audit_metadata(sample)
    cid = "c0"
    return {
        "sample_id": sample.get("sample_id"),
        "schema_version": "single_scene_b0_v1",
        "synthesis_audit_metadata": synthesis_audit,
        "automation_components": [
            {
                "component_id": cid,
                "blueprint_id": bp_id,
                "automation_instance_id": iid or inst.get("automation_instance_id"),
                "blueprint_inputs": dict(inst.get("blueprint_inputs") or {}),
                "bound_entities": list(inst.get("bound_entities") or bb.get("entities") or []),
                "scene_type": scene,
            }
        ],
        "component_runtime_contexts": [
            {
                "component_id": cid,
                "scene_type": scene,
                "observed": projected["observed"],
                "derived_observation": projected["derived_observation"],
                "system_state": dict(sample.get("system_state") or {}),
                "entity_observations": projected["entity_observations"],
                "trigger_context": projected["trigger_context"],
                "entity_states": projected["entity_states"],
                "runtime_memory": projected["runtime_memory"],
                "synthetic_runtime_state": projected["synthetic_runtime_state"],
            }
        ],
        "ordering_semantics": {"component_execution_order": [cid]},
    }

def _classify_status(record: dict[str, Any]) -> str:
    semantic = (record.get("canonical_b0") or {}).get("semantic_actions") or []
    if semantic:
        return "SUCCESS_ACTIONS"
    traces = record.get("execution_traces") or []
    if traces:
        return "SUCCESS_NO_ACTION"
    return "SUCCESS_NO_ACTION"

def run_single_scene_b0(
    sample: dict[str, Any],
    *,
    upstream_ss_sha256: str,
    pipeline_version: str = FORMAL_B0_V2_PIPELINE_VERSION,
    output_contract: str | None = None,
) -> dict[str, Any]:
    if output_contract is None:
        output_contract = "v4" if pipeline_version == FORMAL_B0_V4_PIPELINE_VERSION else "v3"
    sid = str(sample.get("sample_id") or "")
    bb = sample.get("blueprint_binding") or {}
    bp_id = str(bb.get("blueprint_id") or "")
    try:
        parent = _synthetic_parent_from_sample(sample)
        parent["output_contract"] = output_contract
    except Exception as exc:
        return {
            "sample_id": sid,
            "blueprint_id": bp_id,
            "formal_b0_v2": True,
            "single_scene_b0": True,
            "execution_status": "ADAPTER_ERROR",
            "execution_error": str(exc)[:500],
            "pipeline_version": pipeline_version,
            "upstream_ss_sha256": upstream_ss_sha256,
            "generated_at": _utc_now(),
            "canonical_b0": {"semantic_actions": []},
        }

    _, yaml_failures = verify_yaml_resolution(parent)
    if yaml_failures:
        return {
            "sample_id": sid,
            "blueprint_id": bp_id,
            "formal_b0_v2": True,
            "single_scene_b0": True,
            "execution_status": "YAML_RESOLUTION_ERROR",
            "execution_error": "B0_YAML_RESOLUTION_FAIL",
            "execution_detail": {"failures": yaml_failures},
            "pipeline_version": pipeline_version,
            "upstream_ss_sha256": upstream_ss_sha256,
            "generated_at": _utc_now(),
            "canonical_b0": {"semantic_actions": []},
        }

    try:
        with blueprint_yaml_bridge():
            raw = execute_parent_b0(parent)
    except Exception as exc:
        return {
            "sample_id": sid,
            "blueprint_id": bp_id,
            "formal_b0_v2": True,
            "single_scene_b0": True,
            "execution_status": "EXECUTION_ERROR",
            "execution_error": str(exc)[:500],
            "pipeline_version": pipeline_version,
            "upstream_ss_sha256": upstream_ss_sha256,
            "generated_at": _utc_now(),
            "canonical_b0": {"semantic_actions": []},
        }

    semantic = raw.get("semantic_actions")
    if not isinstance(semantic, list):
        raise RuntimeError("B0 execution missing semantic_actions list")

    record = {
        "sample_id": sid,
        "blueprint_id": bp_id,
        "scene_type": sample.get("scene_type"),
        "formal_b0_v2": True,
        "single_scene_b0": True,
        "execution_status": _classify_status({"canonical_b0": {"semantic_actions": semantic}, "execution_traces": raw.get("execution_traces")}),
        "adapter_version": B0_ADAPTER_VERSION,
        "pipeline_version": pipeline_version,
        "output_contract": output_contract,
        "upstream_ss_sha256": upstream_ss_sha256,
        "generated_at": _utc_now(),
        "canonical_b0": {
            "semantic_actions": semantic,
            "provenance": raw.get("provenance"),
            "dedupe_log": raw.get("dedupe_log"),
        },
        "per_component_b0": raw.get("per_component_b0"),
        "execution_traces": raw.get("execution_traces"),
    }
    trace0 = (record.get("execution_traces") or [{}])[0]
    if output_contract == "v4":
        record["formal_b0_status"] = trace0.get("formal_b0_status")
        record["execution_resolution"] = trace0.get("execution_resolution")
        record["diagnostic_reason"] = trace0.get("diagnostic_reason")
        record["internal_formal_b0_status"] = trace0.get("internal_formal_b0_status")
    return record
