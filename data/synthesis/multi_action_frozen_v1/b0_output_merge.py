from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import (
    FORMAL_B0_SS_REPAIR_DIR,
    MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES,
    SS_REPAIR_SAMPLES,
)
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

SS_FORMAL_B0_SCHEMA = "single_scene_formal_b0_v5"
MA_FORMAL_B0_SCHEMA = "multi_action_formal_b0_v3"

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def semantic_action_to_expected_string(action: dict[str, Any]) -> str:

    service = str(action.get("service") or "").strip()
    entity = action.get("target_entity")
    params = dict(action.get("parameters") or {})
    if entity:
        base = f"{service}({entity})"
    else:
        base = service
    extras: list[str] = []
    for key in ("message", "title", "brightness", "brightness_pct"):
        if params.get(key) is not None:
            extras.append(f"{key}={params[key]}")
    if extras:
        return f"{base} {' '.join(extras)}"
    return base

def _matched_rules_from_traces(traces: list[dict[str, Any]]) -> list[str]:
    rules: list[str] = []
    for trace in traces or []:
        branch = trace.get("selected_branch")
        if branch:
            rules.append(str(branch))
        for bid in trace.get("selected_branch_ids") or []:
            rules.append(str(bid))
    return sorted(set(rules))

def _expected_ha_from_semantic(semantic: list[dict[str, Any]]) -> list[str]:
    return [s for s in (semantic_action_to_expected_string(a) for a in semantic) if s]

def _index_traces_by_component(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for trace in traces or []:
        cid = str(trace.get("component_id") or "")
        if cid:
            index[cid] = trace
    return index

def build_formal_b0_single_scene_block(
    b0_record: dict[str, Any],
    *,
    sidecar_sha256: str | None = None,
) -> dict[str, Any]:

    canonical = b0_record.get("canonical_b0") or {}
    semantic = list(canonical.get("semantic_actions") or [])
    return {
        "schema_version": str(b0_record.get("formal_b0_version") or SS_FORMAL_B0_SCHEMA),
        "b0_scope": "single_scene",
        "semantic_actions": semantic,
        "expected_action_ha": _expected_ha_from_semantic(semantic),
        "execution_status": b0_record.get("execution_status"),
        "formal_b0_status": b0_record.get("formal_b0_status"),
        "execution_traces": list(b0_record.get("execution_traces") or []),
        "provenance": canonical.get("provenance"),
        "dedupe_log": canonical.get("dedupe_log"),
        "pipeline_version": b0_record.get("pipeline_version"),
        "adapter_version": b0_record.get("adapter_version"),
        "executor_id": b0_record.get("executor_id"),
        "b0_version": b0_record.get("b0_version"),
        "sidecar_sha256": sidecar_sha256,
        "generated_at": b0_record.get("generated_at"),
    }

def build_formal_b0_multi_action_block(
    b0_record: dict[str, Any],
    *,
    sidecar_sha256: str | None = None,
) -> dict[str, Any]:

    canonical = b0_record.get("canonical_b0") or {}
    parent_semantic = list(canonical.get("semantic_actions") or [])
    traces_by_component = _index_traces_by_component(list(b0_record.get("execution_traces") or []))

    components_block: dict[str, dict[str, Any]] = {}
    for comp in list(b0_record.get("per_component_b0") or []):
        cid = str(comp.get("component_id") or "")
        if not cid:
            continue
        comp_semantic = list(comp.get("semantic_actions") or [])
        components_block[cid] = {
            "b0_scope": "component",
            "component_id": cid,
            "scene_type": comp.get("scene_type"),
            "semantic_actions": comp_semantic,
            "expected_action_ha": _expected_ha_from_semantic(comp_semantic),
            "service_level_view": list(comp.get("service_level_view") or []),
            "uses_common_rule_b0": comp.get("uses_common_rule_b0"),
            "evidence_summary": comp.get("evidence_summary"),
            "execution_trace": traces_by_component.get(cid),
        }

    return {
        "schema_version": str(b0_record.get("schema_version") or MA_FORMAL_B0_SCHEMA),
        "b0_scope": "multi_action_parent",
        "parent": {
            "b0_scope": "parent",
            "semantic_actions": parent_semantic,
            "expected_action_ha": _expected_ha_from_semantic(parent_semantic),
            "execution_status": b0_record.get("execution_status"),
            "provenance": canonical.get("provenance"),
            "merge_order": canonical.get("merge_order"),
            "dedupe_log": canonical.get("dedupe_log"),
            "side_effects_audit": canonical.get("side_effects_audit"),
        },
        "components": components_block,
        "pipeline_version": b0_record.get("pipeline_version"),
        "adapter_version": b0_record.get("adapter_version"),
        "sidecar_sha256": sidecar_sha256,
        "generated_at": b0_record.get("generated_at"),
    }

def formal_b0_record_to_b0_output(
    b0_record: dict[str, Any],
    *,
    scope: str = "legacy_mixed",
    include_component_payload: bool = True,
) -> dict[str, Any]:

    canonical = b0_record.get("canonical_b0") or {}
    semantic = list(canonical.get("semantic_actions") or [])
    expected = _expected_ha_from_semantic(semantic)
    traces = list(b0_record.get("execution_traces") or [])
    per_component = list(b0_record.get("per_component_b0") or [])

    service_level: list[str] = []
    for comp in per_component:
        service_level.extend(comp.get("service_level_view") or [])
    if not service_level:
        service_level = [str(a.get("service") or "") for a in semantic if a.get("service")]

    payload: dict[str, Any] = {
        "scope": scope,
        "expected_action": expected,
        "blocked_action": [],
        "conflict_type": "none",
        "matched_rules": _matched_rules_from_traces(traces),
        "candidate_actions_abstract": sorted(set(service_level)),
        "candidate_actions_ha": list(expected),
        "action_support": {},
        "rule_evidence_map": {},
        "source": "formal_b0_v2",
        "formal_b0_v2": bool(b0_record.get("formal_b0_v2", True)),
        "execution_status": b0_record.get("execution_status"),
        "pipeline_version": b0_record.get("pipeline_version"),
        "adapter_version": b0_record.get("adapter_version"),
        "provenance": canonical.get("provenance"),
        "dedupe_log": canonical.get("dedupe_log"),
    }
    if include_component_payload:
        payload["semantic_actions"] = semantic
        payload["per_component_b0"] = per_component
        payload["execution_traces"] = traces
    return payload

def merge_formal_b0_into_sample(
    sample: dict[str, Any],
    b0_record: dict[str, Any],
    *,
    id_key: str,
    sidecar_sha256: str | None = None,
) -> dict[str, Any]:

    out = dict(sample)
    if id_key == "sample_id":
        out["formal_b0"] = build_formal_b0_single_scene_block(
            b0_record,
            sidecar_sha256=sidecar_sha256,
        )
        out["b0_output"] = formal_b0_record_to_b0_output(
            b0_record,
            scope="single_scene_legacy_compat",
            include_component_payload=False,
        )
        return out

    if id_key != "multi_action_id":
        raise ValueError(f"Unsupported id_key for formal B0 merge: {id_key!r}")

    formal = build_formal_b0_multi_action_block(b0_record, sidecar_sha256=sidecar_sha256)
    out["formal_b0"] = formal
    out["b0_output"] = formal_b0_record_to_b0_output(
        b0_record,
        scope="parent_legacy_compat",
        include_component_payload=False,
    )

    comp_blocks = formal.get("components") or {}
    merged_components: list[dict[str, Any]] = []
    for comp in list(out.get("components") or []):
        merged = dict(comp)
        cid = str(merged.get("component_id") or "")
        if cid in comp_blocks:
            merged["formal_b0_component"] = comp_blocks[cid]
        merged_components.append(merged)
    out["components"] = merged_components
    return out

def _index_b0_records(b0_rows: list[dict[str, Any]], id_key: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in b0_rows:
        rid = str(row.get(id_key) or "")
        if not rid:
            continue
        if rid in index:
            raise RuntimeError(f"Duplicate B0 id {rid!r}")
        index[rid] = row
    return index

def merge_formal_b0_into_samples(
    *,
    samples_jsonl: Path,
    b0_jsonl: Path,
    id_key: str,
    out_jsonl: Path | None = None,
    backup: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:

    if not samples_jsonl.is_file():
        raise FileNotFoundError(f"Missing samples: {samples_jsonl}")
    if not b0_jsonl.is_file():
        raise FileNotFoundError(f"Missing B0 sidecar: {b0_jsonl}")

    out_jsonl = out_jsonl or samples_jsonl
    samples = read_jsonl(samples_jsonl)
    b0_index = _index_b0_records(read_jsonl(b0_jsonl), id_key)
    sidecar_sha256 = _sha256(b0_jsonl)

    missing: list[str] = []
    status_counts: Counter[str] = Counter()
    parent_nonempty = 0
    component_nonempty = 0

    merged_samples: list[dict[str, Any]] = []
    for sample in samples:
        sid = str(sample.get(id_key) or "")
        b0_record = b0_index.get(sid)
        if b0_record is None:
            missing.append(sid)
            continue
        merged = merge_formal_b0_into_sample(
            sample,
            b0_record,
            id_key=id_key,
            sidecar_sha256=sidecar_sha256,
        )
        merged_samples.append(merged)

        formal = merged.get("formal_b0") or {}
        if id_key == "sample_id":
            status = str(formal.get("execution_status") or "UNKNOWN")
            if formal.get("expected_action_ha"):
                parent_nonempty += 1
        else:
            parent = formal.get("parent") or {}
            status = str(parent.get("execution_status") or "UNKNOWN")
            if parent.get("expected_action_ha"):
                parent_nonempty += 1
            for comp in (formal.get("components") or {}).values():
                if comp.get("expected_action_ha"):
                    component_nonempty += 1
        status_counts[status] += 1

    if missing:
        raise RuntimeError(f"Missing B0 for {len(missing)} samples (first: {missing[:5]})")

    if backup and out_jsonl == samples_jsonl and samples_jsonl.exists():
        backup_path = samples_jsonl.with_suffix(samples_jsonl.suffix + ".pre_formal_b0_merge.bak")
        shutil.copy2(samples_jsonl, backup_path)
        if verbose:
            print(f"Backup: {backup_path}", flush=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in merged_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    report = {
        "verdict": "FORMAL_B0_MERGE_COMPLETE",
        "merged_at": _utc_now(),
        "samples_jsonl": str(samples_jsonl),
        "b0_jsonl": str(b0_jsonl),
        "out_jsonl": str(out_jsonl),
        "id_key": id_key,
        "total_samples": len(merged_samples),
        "labeled": len(merged_samples),
        "parent_nonempty_expected_action": parent_nonempty,
        "component_nonempty_expected_action": component_nonempty if id_key == "multi_action_id" else None,
        "execution_status_counts": dict(status_counts),
        "input_samples_sha256": _sha256(samples_jsonl) if samples_jsonl != out_jsonl else None,
        "output_sha256": _sha256(out_jsonl),
        "b0_sha256": sidecar_sha256,
        "merge_mode": "scope_separated_formal_b0",
    }
    return report

def merge_b0_into_samples(
    *,
    samples_jsonl: Path,
    b0_jsonl: Path,
    id_key: str,
    out_jsonl: Path | None = None,
    backup: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    if not samples_jsonl.is_file():
        raise FileNotFoundError(f"Missing samples: {samples_jsonl}")
    if not b0_jsonl.is_file():
        raise FileNotFoundError(f"Missing B0 sidecar: {b0_jsonl}")

    out_jsonl = out_jsonl or samples_jsonl
    samples = read_jsonl(samples_jsonl)
    b0_index = _index_b0_records(read_jsonl(b0_jsonl), id_key)

    missing: list[str] = []
    status_counts: Counter[str] = Counter()
    nonempty = 0

    for sample in samples:
        sid = str(sample.get(id_key) or "")
        b0_record = b0_index.get(sid)
        if b0_record is None:
            missing.append(sid)
            continue
        merged = merge_formal_b0_into_sample(sample, b0_record, id_key=id_key)
        sample.clear()
        sample.update(merged)
        payload = sample.get("b0_output") or {}
        status = str(payload.get("execution_status") or "UNKNOWN")
        status_counts[status] += 1
        if payload.get("expected_action"):
            nonempty += 1

    if missing:
        raise RuntimeError(f"Missing B0 for {len(missing)} samples (first: {missing[:5]})")

    if backup and out_jsonl == samples_jsonl and samples_jsonl.exists():
        backup_path = samples_jsonl.with_suffix(samples_jsonl.suffix + ".pre_b0_merge.bak")
        shutil.copy2(samples_jsonl, backup_path)
        if verbose:
            print(f"Backup: {backup_path}", flush=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    report = {
        "verdict": "B0_MERGE_COMPLETE",
        "merged_at": _utc_now(),
        "samples_jsonl": str(samples_jsonl),
        "b0_jsonl": str(b0_jsonl),
        "out_jsonl": str(out_jsonl),
        "id_key": id_key,
        "total_samples": len(samples),
        "labeled": len(samples),
        "nonempty_expected_action": nonempty,
        "execution_status_counts": dict(status_counts),
        "input_samples_sha256": _sha256(samples_jsonl) if samples_jsonl != out_jsonl else None,
        "output_sha256": _sha256(out_jsonl),
        "b0_sha256": _sha256(b0_jsonl),
    }
    return report

def run_ss_repair_b0_merge(
    *,
    ss_samples: Path | None = None,
    ss_b0: Path | None = None,
    ma_samples: Path | None = None,
    ma_b0: Path | None = None,
    out_ss: Path | None = None,
    out_ma: Path | None = None,
    backup: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    ss_samples = ss_samples or SS_REPAIR_SAMPLES
    ss_b0 = ss_b0 or (FORMAL_B0_SS_REPAIR_DIR / "single_scene_b0_outputs.jsonl")
    ma_samples = ma_samples or MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES
    ma_b0 = ma_b0 or (FORMAL_B0_SS_REPAIR_DIR / "multi_action_b0_outputs.jsonl")

    ss_report = merge_b0_into_samples(
        samples_jsonl=ss_samples,
        b0_jsonl=ss_b0,
        id_key="sample_id",
        out_jsonl=out_ss or ss_samples,
        backup=backup,
        verbose=verbose,
    )
    ma_report = merge_b0_into_samples(
        samples_jsonl=ma_samples,
        b0_jsonl=ma_b0,
        id_key="multi_action_id",
        out_jsonl=out_ma or ma_samples,
        backup=backup,
        verbose=verbose,
    )

    val_dir = FORMAL_B0_SS_REPAIR_DIR / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": "B0_OUTPUT_MERGE_SS_REPAIR_COMPLETE",
        "merged_at": _utc_now(),
        "single_scene": ss_report,
        "multi_action": ma_report,
    }
    (val_dir / "b0_output_merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
