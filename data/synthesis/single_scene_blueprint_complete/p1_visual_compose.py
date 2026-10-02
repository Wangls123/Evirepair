from __future__ import annotations

from typing import Any

from smarthome_mdf.multi_action_vnext.source_adapters import SourceObservation

def pick_youhome_visual_pool(pool, scene: str = "visual_fusion") -> list[dict]:

    rows = list(pool._pools.get(scene) or [])
    out = []
    for row in rows:
        obs = row.get("observed") or {}
        if obs.get("frame_id"):
            out.append(row)
    return out

def compose_visual_fusion_observed(
    runtime_obs: SourceObservation,
    visual_sample: dict,
) -> tuple[dict, dict]:

    visual_obs = dict(visual_sample.get("observed") or {})
    runtime_ds = str(runtime_obs.dataset)
    visual_prov = visual_sample.get("provenance") or {}
    visual_ds = str(visual_prov.get("source_dataset") or "YouHome")

    provenance_extra = {
        "runtime_source": {
            "dataset": runtime_ds,
            "observation_id": runtime_obs.record_id,
            "original_timestamp": runtime_obs.original_timestamp,
            "measurement": runtime_obs.measurement,
        },
        "visual_source": {
            "dataset": visual_ds,
            "participant_id": visual_obs.get("participant_id"),
            "sequence_id": visual_obs.get("sequence_id"),
            "frame_id": visual_obs.get("frame_id"),
            "original_timestamp": visual_obs.get("timestamp") or visual_sample.get("timestamp"),
        },
        "composition_timeline": "independent_runtime_and_visual_timestamps",
        "alignment_method": "blueprint_runtime_match_plus_visual_frame_selection",
    }
    if runtime_ds != visual_ds:
        provenance_extra["composition_type"] = "CROSS_DATASET_SYNTHETIC"
        provenance_extra["cross_dataset_alignment_method"] = "independent_runtime_and_visual_selection"
    else:
        provenance_extra["composition_type"] = "SAME_SOURCE"

    return visual_obs, provenance_extra
