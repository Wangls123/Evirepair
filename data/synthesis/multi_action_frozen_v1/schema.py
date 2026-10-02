from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import SCHEMA_VERSION
from smarthome_mdf.multi_action_frozen_v1.entity_policy import EntityBindingRecord
from smarthome_mdf.multi_action_frozen_v1.parent_signature import parent_signature
from smarthome_mdf.multi_action_frozen_v1.provenance_policy import component_provenance, parent_provenance

def _action_schema(sample: dict) -> dict[str, Any]:
    meta = sample.get("synthesis_metadata") or {}
    return {
        "service": meta.get("action_path_signature"),
        "target_entity_roles": sample.get("blueprint_binding", {}).get("entities", []),
        "branch_id": meta.get("branch_id"),
        "stable_action_path_id": meta.get("stable_action_path_id"),
        "parameters": {},
    }

def _component_runtime(sample: dict, component_id: str) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "system_state": sample.get("system_state") or {},
        "entity_observations": sample.get("entity_observations") or [],
        "observed": sample.get("observed") or {},
    }

def build_multi_action_record(
    *,
    multi_action_id: str,
    component_samples: list[tuple[str, dict]],
    entity_bindings: list[tuple[str, list[EntityBindingRecord], dict[str, Any]]],
    temporal: dict[str, Any],
    shared_runtime: dict[str, Any] | None,
    construction_trace: list[str],
    composition_type: str,
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    prov_components: list[dict[str, Any]] = []
    runtime_states: list[dict[str, Any]] = []
    blueprint_ids: list[str] = []
    behavior_sigs: list[str] = []
    source_ids: list[str] = []

    shared_entity = False
    shared_map: dict[str, str] = {}

    for (cid, sample), (_, records, bind_meta) in zip(component_samples, entity_bindings):
        bb = sample.get("blueprint_binding") or {}
        prov = component_provenance(sample, component_id=cid)
        prov_components.append(prov)
        blueprint_ids.append(bb.get("blueprint_id", ""))
        behavior_sigs.append(sample.get("canonical_behavior_signature") or "")
        source_ids.append(f"{prov.get('source_dataset')}:{prov.get('source_observation_id')}")
        if bind_meta.get("shared_entity"):
            shared_entity = True
            shared_map.update(bind_meta.get("shared_entity_mapping") or {})

        comp_temp = next(
            (t for t in temporal.get("composition_timeline", {}).get("components", []) if t["component_id"] == cid),
            {},
        )
        components.append(
            {
                "component_id": cid,
                "blueprint_id": bb.get("blueprint_id"),
                "scene": sample.get("scene_type"),
                "single_scene_sample_id": sample.get("sample_id"),
                "grounded_instance": bb.get("automation_instance_id"),
                "behavior_id": (sample.get("synthesis_metadata") or {}).get("stable_action_path_id"),
                "runtime_state": _component_runtime(sample, cid),
                "source_provenance": prov,
                "entity_binding": {
                    "records": [r.to_dict() for r in records],
                    **bind_meta,
                },
                "temporal_alignment": comp_temp,
                "component_action_schema": _action_schema(sample),
            }
        )
        runtime_states.append(_component_runtime(sample, cid))

    alignment_mode = temporal.get("composition_timeline", {}).get("alignment_mode", "UNKNOWN")
    sig = parent_signature(
        blueprint_ids=blueprint_ids,
        behavior_signatures=behavior_sigs,
        source_identities=source_ids,
        shared_entity=shared_entity,
        shared_entity_mapping=shared_map,
        temporal_alignment_class=alignment_mode,
        component_count=len(component_samples),
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "multi_action_id": multi_action_id,
        "component_count": len(component_samples),
        "components": components,
        "parent_runtime_state": {
            "component_runtime_states": runtime_states,
            "shared_runtime_projection": shared_runtime or {},
        },
        "shared_entities": {
            "shared_entity": shared_entity,
            "shared_entity_mapping": shared_map,
        },
        "composition_timeline": temporal.get("composition_timeline"),
        "composition_type": composition_type,
        "source_mode": temporal.get("source_mode"),
        "parent_provenance": parent_provenance(prov_components),
        "construction_trace": construction_trace,
        "canonical_parent_signature": sig,
        "synthesis_metadata": {
            "construction_only": True,
            "synthesized_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    return record

def multi_action_id_from_signature(sig: str, *, seed: int, index: int) -> str:
    h = hashlib.sha256(f"{sig}:{seed}:{index}".encode()).hexdigest()[:12]
    return f"ma_frozen_{h}"
