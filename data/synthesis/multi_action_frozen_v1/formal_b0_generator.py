from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import (
    execute_frozen_parent_b0,
    frozen_to_b0_parent,
    verify_yaml_resolution,
)
from smarthome_mdf.multi_action_frozen_v1.config import (
    ACTION_SCHEMA_VERSION,
    B0_ADAPTER_VERSION,
    EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    FORMAL_B0_GENERIC_EXECUTOR_VERSION,
    FORMAL_B0_INSTANCE_RESOLVER_VERSION,
    FORMAL_B0_PIPELINE_VERSION,
    FORMAL_B0_REGISTRY_VERSION,
    FORMAL_B0_RUNTIME_PROJECTOR_VERSION,
    FORMAL_B0_V2_PIPELINE_VERSION,
    SCHEMA_VERSION,
)

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _error_record(
    ma: dict[str, Any],
    *,
    execution_status: str,
    error: str,
    detail: dict[str, Any] | None = None,
    pipeline_version: str = FORMAL_B0_PIPELINE_VERSION,
    formal_key: str = "formal_b0_v1",
) -> dict[str, Any]:
    return {
        "multi_action_id": ma.get("multi_action_id"),
        formal_key: True,
        "development_pilot_only": False,
        "execution_status": execution_status,
        "execution_error": error,
        "execution_detail": detail or {},
        "multi_action_frozen_corpus_sha256": EXPECTED_MULTI_ACTION_FROZEN_SHA256,
        "schema_version": SCHEMA_VERSION,
        "adapter_version": B0_ADAPTER_VERSION,
        "pipeline_version": pipeline_version,
        "generated_at": _utc_now(),
        "canonical_b0": {"semantic_actions": []},
        "compatibility_debug": {"service_level_view_is_canonical": False},
    }

def _classify_parent_status(record: dict[str, Any]) -> str:
    semantic = (record.get("canonical_b0") or {}).get("semantic_actions") or []
    if semantic:
        return "SUCCESS_ACTIONS"
    traces = record.get("execution_traces") or []
    if traces:
        outcomes = {t.get("execution_outcome") for t in traces}
        allowed = {"VALID_NO_ACTION", "VALID_EMPTY_BRANCH", "RUNTIME_EVIDENCE_INSUFFICIENT"}
        if outcomes and outcomes <= allowed:
            return "SUCCESS_NO_ACTION"
    return "SUCCESS_NO_ACTION"

def run_formal_b0_for_parent(
    ma: dict[str, Any],
    frozen_index: dict[str, dict[str, Any]],
    *,
    pipeline_version: str = FORMAL_B0_PIPELINE_VERSION,
) -> dict[str, Any]:

    is_v2 = pipeline_version == FORMAL_B0_V2_PIPELINE_VERSION
    formal_key = "formal_b0_v2" if is_v2 else "formal_b0_v1"
    err_kw = {"pipeline_version": pipeline_version, "formal_key": formal_key}
    try:
        parent = frozen_to_b0_parent(ma, frozen_index)
    except Exception as exc:
        return _error_record(ma, execution_status="ADAPTER_ERROR", error=str(exc)[:500], **err_kw)

    _, yaml_failures = verify_yaml_resolution(parent)
    if yaml_failures:
        return _error_record(
            ma,
            execution_status="YAML_RESOLUTION_ERROR",
            error="B0_YAML_RESOLUTION_FAIL",
            detail={"failures": yaml_failures},
            **err_kw,
        )

    try:
        record = execute_frozen_parent_b0(ma, frozen_index)
    except RuntimeError as exc:
        msg = str(exc)
        if "YAML" in msg:
            status = "YAML_RESOLUTION_ERROR"
        elif "INSTANCE" in msg:
            status = "INSTANCE_RESOLUTION_ERROR"
        else:
            status = "EXECUTION_ERROR"
        return _error_record(ma, execution_status=status, error=msg[:500], **err_kw)
    except Exception as exc:
        return _error_record(ma, execution_status="EXECUTION_ERROR", error=str(exc)[:500], **err_kw)

    record[formal_key] = True
    if is_v2:
        record.pop("formal_b0_v1", None)
    record["development_pilot_only"] = False
    record["schema_version"] = SCHEMA_VERSION
    record["pipeline_version"] = pipeline_version
    record["generated_at"] = _utc_now()
    record["execution_status"] = _classify_parent_status(record)
    record.pop("execution_error", None)
    return record

def pipeline_lineage(*, pipeline_version: str = FORMAL_B0_PIPELINE_VERSION) -> dict[str, str]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": pipeline_version,
        "adapter_version": B0_ADAPTER_VERSION,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "blueprint_executor_registry_version": FORMAL_B0_REGISTRY_VERSION,
        "generic_executor_version": FORMAL_B0_GENERIC_EXECUTOR_VERSION,
        "runtime_projector_version": FORMAL_B0_RUNTIME_PROJECTOR_VERSION,
        "instance_resolver_version": FORMAL_B0_INSTANCE_RESOLVER_VERSION,
        "multi_action_frozen_corpus_sha256": EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    }
