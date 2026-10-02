from __future__ import annotations

from typing import Any

def component_provenance(sample: dict, *, component_id: str) -> dict[str, Any]:
    prov = sample.get("provenance") or {}
    meta = sample.get("synthesis_metadata") or {}
    bb = sample.get("blueprint_binding") or {}
    return {
        "component_id": component_id,
        "source_dataset": prov.get("source_dataset"),
        "source_observation_id": prov.get("source_record_id"),
        "source_timestamp": prov.get("synthesis_timestamp"),
        "source_provenance": {
            "lineage": prov.get("lineage"),
            "source_record_id": prov.get("source_record_id"),
            "source_dataset": prov.get("source_dataset"),
        },
        "blueprint_id": bb.get("blueprint_id"),
        "grounded_instance": bb.get("automation_instance_id"),
        "behavior_signature": sample.get("canonical_behavior_signature"),
        "behavior_id": meta.get("stable_action_path_id") or meta.get("behavior_target"),
        "single_scene_sample_id": sample.get("sample_id"),
        "single_scene_sample_hash": sample.get("diversity_signature"),
    }

def parent_provenance(components: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "component_provenance": components,
        "source_datasets": sorted({c.get("source_dataset") for c in components if c.get("source_dataset")}),
        "multi_source": len({c.get("source_dataset") for c in components}) > 1,
        "preservation_policy": "NO_COLLAPSE_MULTI_SOURCE",
    }
