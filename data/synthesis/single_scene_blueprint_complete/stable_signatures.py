from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_DYNAMIC_SUFFIX_RE = re.compile(r"::a\d+$")
_RAW_BRANCH_EXPR = re.compile(r"^(\{|\}|and|or|template|\{\{)")

def strip_dynamic_behavior_suffix(behavior_target: str | None) -> str:
    if not behavior_target:
        return ""
    return _DYNAMIC_SUFFIX_RE.sub("", str(behavior_target))

def stable_action_node_id(
    *,
    blueprint_id: str,
    branch_id: str,
    action_path_signature: str,
    ir_path: str | None = None,
) -> str:

    if ir_path:
        return f"{blueprint_id}:{ir_path}"
    safe_branch = branch_id if not _RAW_BRANCH_EXPR.match(str(branch_id)) else f"expr_{hashlib.sha256(str(branch_id).encode()).hexdigest()[:8]}"
    return f"{blueprint_id}:{safe_branch}:{action_path_signature}"

def stable_branch_id(*, blueprint_id: str, branch_id: str, ir_path: str | None = None) -> str:
    if ir_path:
        return f"{blueprint_id}:{ir_path}"
    bid = str(branch_id or "default")
    if _RAW_BRANCH_EXPR.match(bid) or "{{" in bid:
        return f"{blueprint_id}:expr_{hashlib.sha256(bid.encode()).hexdigest()[:8]}"
    return f"{blueprint_id}:{bid}"

def stable_action_path_id(
    *,
    blueprint_id: str,
    branch_id: str,
    action_path_signature: str,
    ir_path: str | None = None,
) -> str:
    sb = stable_branch_id(blueprint_id=blueprint_id, branch_id=branch_id, ir_path=ir_path)
    return f"{sb}→{action_path_signature}"

def _norm_temporal(temporal: Any) -> str:
    if temporal is None:
        return "point"
    if isinstance(temporal, dict):
        keys = sorted(temporal.keys())
        return "|".join(f"{k}={temporal.get(k)}" for k in keys if k not in ("timestamp",))
    return str(temporal)

def canonical_behavior_signature(sample: dict) -> str:

    bb = sample.get("blueprint_binding") or {}
    meta = sample.get("synthesis_metadata") or {}
    bt = strip_dynamic_behavior_suffix(meta.get("behavior_target") or (sample.get("system_state") or {}).get("behavior_target"))
    parts = [
        bb.get("blueprint_id"),
        bb.get("automation_instance_id"),
        stable_branch_id(
            blueprint_id=str(bb.get("blueprint_id") or ""),
            branch_id=str(meta.get("branch_id") or meta.get("stable_branch_id") or ""),
            ir_path=meta.get("ir_branch_path"),
        ),
        stable_action_path_id(
            blueprint_id=str(bb.get("blueprint_id") or ""),
            branch_id=str(meta.get("branch_id") or ""),
            action_path_signature=str(meta.get("action_path_signature") or ""),
            ir_path=meta.get("ir_action_path"),
        ),
        meta.get("stable_action_node_id") or stable_action_node_id(
            blueprint_id=str(bb.get("blueprint_id") or ""),
            branch_id=str(meta.get("branch_id") or ""),
            action_path_signature=str(meta.get("action_path_signature") or ""),
            ir_path=meta.get("ir_action_path"),
        ),
        meta.get("action_path_signature"),
        meta.get("runtime_situation"),
        meta.get("entity_binding_signature"),
        _norm_temporal(meta.get("temporal_pattern")),
    ]
    if bt and not _DYNAMIC_SUFFIX_RE.search(bt):
        parts.append(bt.split("::")[0] if "::" in bt else bt)
    return "|".join(str(p) for p in parts if p)

def canonical_behavior_hash(sample: dict) -> str:
    return hashlib.sha256(canonical_behavior_signature(sample).encode()).hexdigest()[:16]

def structural_signature_stable(sample: dict) -> str:

    return canonical_behavior_signature(sample)

def is_dynamic_behavior_target(behavior_target: str | None) -> bool:
    return bool(behavior_target and _DYNAMIC_SUFFIX_RE.search(str(behavior_target)))

def numeric_only_near_duplicate_stable(a: dict, b: dict) -> bool:
    if canonical_behavior_signature(a) != canonical_behavior_signature(b):
        return False
    bb_a = a.get("blueprint_binding") or {}
    bb_b = b.get("blueprint_binding") or {}
    if bb_a.get("blueprint_id") != bb_b.get("blueprint_id"):
        return False
    if bb_a.get("automation_instance_id") != bb_b.get("automation_instance_id"):
        return False
    return True

def temporal_shift_near_duplicate_stable(a: dict, b: dict) -> bool:
    if not numeric_only_near_duplicate_stable(a, b):
        return False
    prov_a = a.get("provenance") or {}
    prov_b = b.get("provenance") or {}
    if prov_a.get("source_record_id") != prov_b.get("source_record_id"):
        meta_a = a.get("synthesis_metadata") or {}
        meta_b = b.get("synthesis_metadata") or {}
        return _norm_temporal(meta_a.get("temporal_pattern")) == _norm_temporal(meta_b.get("temporal_pattern"))
    return False

def temporal_semantic_signature(sample: dict) -> str:

    meta = sample.get("synthesis_metadata") or {}
    tp = meta.get("temporal_pattern")
    if isinstance(tp, dict):
        parts = []
        for k in sorted(tp.keys()):
            if k in ("timestamp", "synthesis_timestamp", "original_timestamp"):
                continue
            parts.append(f"{k}={tp[k]}")
        return "|".join(parts) or "point"
    return str(tp or "point")

def source_observation_signature(sample: dict) -> str:
    prov = sample.get("provenance") or {}
    parts = [prov.get("source_record_id")]
    vs = prov.get("visual_source") or {}
    if vs.get("frame_id"):
        parts.append(str(vs.get("frame_id")))
    return "|".join(str(p) for p in parts if p)

def runtime_semantic_signature(sample: dict) -> str:
    eo = (sample.get("entity_observations") or [{}])[0]
    attrs = dict(eo.get("attributes") or {})
    for k in ("timestamp",):
        attrs.pop(k, None)
    return json.dumps(attrs, sort_keys=True, default=str)

def canonical_sample_signature(sample: dict) -> str:
    bb = sample.get("blueprint_binding") or {}
    meta = sample.get("synthesis_metadata") or {}
    parts = [
        bb.get("blueprint_id"),
        bb.get("automation_instance_id"),
        meta.get("stable_branch_id") or stable_branch_id(blueprint_id=str(bb.get("blueprint_id")), branch_id=str(meta.get("branch_id") or ""), ir_path=meta.get("ir_branch_path")),
        meta.get("stable_action_path_id") or stable_action_path_id(blueprint_id=str(bb.get("blueprint_id")), branch_id=str(meta.get("branch_id") or ""), action_path_signature=str(meta.get("action_path_signature") or ""), ir_path=meta.get("ir_action_path")),
        meta.get("action_path_signature"),
        meta.get("entity_binding_signature"),
        runtime_semantic_signature(sample),
        temporal_semantic_signature(sample),
        source_observation_signature(sample),
    ]
    return "|".join(str(p) for p in parts if p)

def behavior_temporal_key_from_target(
    *,
    blueprint_id: str,
    automation_instance_id: str,
    branch_id: str,
    action_path_signature: str,
    entity_binding_signature: str = "",
    temporal_pattern: str = "point_timestamp",
) -> str:

    parts = [
        blueprint_id,
        automation_instance_id,
        stable_branch_id(blueprint_id=blueprint_id, branch_id=branch_id),
        stable_action_path_id(
            blueprint_id=blueprint_id,
            branch_id=branch_id,
            action_path_signature=action_path_signature,
        ),
        action_path_signature,
        entity_binding_signature,
        temporal_pattern,
    ]
    return "|".join(str(p) for p in parts if p)

def behavior_source_pair_key(
    *,
    behavior_temporal_key: str,
    source_record_id: str,
    visual_frame_id: str = "",
) -> str:
    return f"{behavior_temporal_key}|{source_record_id}|{visual_frame_id}"

def behavior_temporal_dedup_signature(sample: dict) -> str:

    bb = sample.get("blueprint_binding") or {}
    meta = sample.get("synthesis_metadata") or {}
    parts = [
        bb.get("blueprint_id"),
        bb.get("automation_instance_id"),
        meta.get("stable_branch_id") or stable_branch_id(blueprint_id=str(bb.get("blueprint_id")), branch_id=str(meta.get("branch_id") or ""), ir_path=meta.get("ir_branch_path")),
        meta.get("stable_action_path_id") or stable_action_path_id(blueprint_id=str(bb.get("blueprint_id")), branch_id=str(meta.get("branch_id") or ""), action_path_signature=str(meta.get("action_path_signature") or ""), ir_path=meta.get("ir_action_path")),
        meta.get("action_path_signature"),
        meta.get("entity_binding_signature"),
        runtime_semantic_signature(sample),
        temporal_semantic_signature(sample),
    ]
    return "|".join(str(p) for p in parts if p)

def behavior_source_dedup_key(sample: dict) -> str:
    prov = sample.get("provenance") or {}
    vs = prov.get("visual_source") or {}
    visual_id = vs.get("frame_id") or (sample.get("observed") or {}).get("frame_id") or ""
    return f"{behavior_temporal_dedup_signature(sample)}|{prov.get('source_record_id') or ''}|{visual_id}"

def _absolute_timestamp(sample: dict) -> str:
    prov = sample.get("provenance") or {}
    return str((sample.get("observed") or {}).get("timestamp") or prov.get("synthesis_timestamp") or "")

def is_timestamp_only_duplicate(a: dict, b: dict) -> bool:

    if behavior_temporal_dedup_signature(a) != behavior_temporal_dedup_signature(b):
        return False
    prov_a = a.get("provenance") or {}
    prov_b = b.get("provenance") or {}
    if prov_a.get("source_record_id") != prov_b.get("source_record_id"):
        return False
    ts_a = _absolute_timestamp(a)
    ts_b = _absolute_timestamp(b)
    return bool(ts_a and ts_b and ts_a != ts_b)

def same_source_observation_duplicate(a: dict, b: dict) -> bool:

    if canonical_behavior_hash(a) != canonical_behavior_hash(b):
        return False
    prov_a = a.get("provenance") or {}
    prov_b = b.get("provenance") or {}
    return bool(prov_a.get("source_record_id")) and prov_a.get("source_record_id") == prov_b.get("source_record_id")
