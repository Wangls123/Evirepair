from __future__ import annotations

import json
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import CONSTRUCTION_VALIDATOR_VERSION, FORBIDDEN_FIELDS
from smarthome_mdf.multi_action_frozen_v1.entity_policy import validate_entity_binding

def _has_forbidden(obj: Any, path: str = "") -> list[str]:
    violations: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in FORBIDDEN_FIELDS or any(k.lower().endswith(f"_{f}") for f in FORBIDDEN_FIELDS):
                violations.append(f"FORBIDDEN_FIELD:{path}.{k}" if path else f"FORBIDDEN_FIELD:{k}")
            violations.extend(_has_forbidden(v, f"{path}.{k}" if path else k))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(_has_forbidden(item, f"{path}[{i}]"))
    return violations

def validate_construction(
    record: dict[str, Any],
    *,
    frozen_index: dict[str, dict],
    seen_signatures: set[str] | None = None,
    seen_component_tuples: set[frozenset[str]] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []

    forbidden = _has_forbidden(record)
    if forbidden:
        reasons.extend(forbidden[:5])

    comps = record.get("components") or []
    if len(comps) not in (2, 3):
        reasons.append("INVALID_COMPONENT_COUNT")

    for comp in comps:
        sid = comp.get("single_scene_sample_id")
        if sid not in frozen_index:
            reasons.append(f"MISSING_FROZEN_SAMPLE:{sid}")
            continue
        frozen = frozen_index[sid]
        if comp.get("blueprint_id") != frozen.get("blueprint_binding", {}).get("blueprint_id"):
            reasons.append(f"BLUEPRINT_MISMATCH:{sid}")
        stored_hash = frozen.get("diversity_signature")
        prov_hash = comp.get("source_provenance", {}).get("single_scene_sample_hash")
        if stored_hash and prov_hash and stored_hash != prov_hash:
            reasons.append(f"HASH_MISMATCH:{sid}")

        bind = comp.get("entity_binding") or {}
        records = bind.get("records") or []
        from smarthome_mdf.multi_action_frozen_v1.entity_policy import EntityBindingRecord

        rec_objs = [EntityBindingRecord(**{k: r[k] for k in EntityBindingRecord.__dataclass_fields__}) for r in records]
        ok, msg = validate_entity_binding(bind, rec_objs)
        if not ok:
            reasons.append(f"ENTITY_BINDING_INVALID:{msg}")

    timeline = record.get("composition_timeline") or {}
    if not timeline.get("original_timestamps_preserved"):
        reasons.append("TEMPORAL_ALIGNMENT_INVALID:ORIGINALS_NOT_PRESERVED")
    for comp in timeline.get("components") or []:
        if not comp.get("source_timestamp_original") and comp.get("alignment_class") != "CROSS_DATASET_SYNTHETIC":
            reasons.append("TEMPORAL_ALIGNMENT_INVALID:MISSING_SOURCE_TIMESTAMP")

    parent_prov = record.get("parent_provenance") or {}
    if not parent_prov.get("component_provenance"):
        reasons.append("PROVENANCE_INVALID:EMPTY")
    if len(parent_prov.get("component_provenance") or []) != len(comps):
        reasons.append("PROVENANCE_INVALID:COMPONENT_COUNT")

    sig = record.get("canonical_parent_signature")
    if seen_signatures is not None and sig:
        if sig in seen_signatures:
            reasons.append("DUPLICATE_PARENT_SIGNATURE")
        else:
            seen_signatures.add(sig)

    comp_ids = frozenset(c.get("single_scene_sample_id") for c in comps if c.get("single_scene_sample_id"))
    if seen_component_tuples is not None and comp_ids:
        if comp_ids in seen_component_tuples:
            reasons.append("COMPONENT_SET_DUPLICATE")
        else:
            seen_component_tuples.add(comp_ids)

    runtime = record.get("parent_runtime_state") or {}
    if not runtime.get("component_runtime_states"):
        reasons.append("RUNTIME_INVALID:MISSING_COMPONENT_STATES")

    verdict = "CONSTRUCTION_ACCEPT" if not reasons else "CONSTRUCTION_REJECT"
    return {
        "verdict": verdict,
        "reasons": reasons,
        "validator_version": CONSTRUCTION_VALIDATOR_VERSION,
    }
