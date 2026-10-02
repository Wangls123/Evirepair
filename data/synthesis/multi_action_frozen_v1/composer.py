from __future__ import annotations

import random
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.blueprint_profile import BlueprintProfile, _entity_role
from smarthome_mdf.multi_action_frozen_v1.composability import ComposabilityClass, classify_pair
from smarthome_mdf.multi_action_frozen_v1.entity_policy import (
    bind_entities_independent,
    bind_entities_shared,
    propose_shared_mapping,
)
from smarthome_mdf.multi_action_frozen_v1.schema import build_multi_action_record, multi_action_id_from_signature
from smarthome_mdf.multi_action_frozen_v1.temporal_policy import align_temporal_components

def _resolve_shared_runtime(
    component_samples: list[tuple[str, dict]],
    shared_map: dict[str, str],
) -> tuple[dict[str, Any], str]:
    if not shared_map:
        return {}, "NO_SHARED_ENTITIES"
    parent_states: dict[str, list[dict]] = {}
    for cid, sample in component_samples:
        for obs in sample.get("entity_observations") or []:
            eid = obs.get("entity_id", "")
            parent = shared_map.get(eid, eid)
            if parent.startswith("shared."):
                parent_states.setdefault(parent, []).append({"component_id": cid, "state": obs.get("state"), "ts": obs.get("timestamp")})
    projection: dict[str, Any] = {}
    for parent, states in parent_states.items():
        uniq = {(s["state"], s["ts"]) for s in states}
        if len({s["state"] for s in states}) == 1:
            outcome = "CONSISTENT_SHARED_STATE"
        elif len(uniq) <= len(states):
            outcome = "TEMPORALLY_ALIGNABLE_SHARED_STATE"
        else:
            outcome = "SHARED_STATE_CONFLICT"
        projection[parent] = {"outcome": outcome, "component_states": states}
        if outcome == "SHARED_STATE_CONFLICT":
            return projection, outcome
    return projection, "CONSISTENT_SHARED_STATE"

def compose_multi_action(
    component_samples: list[dict],
    profiles: dict[str, BlueprintProfile],
    *,
    rng: random.Random,
    use_shared_entity: bool | None = None,
    force_synthetic_temporal: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    if len(component_samples) not in (2, 3):
        return None, ["INVALID_COMPONENT_COUNT"]

    trace: list[str] = ["COMPOSE_START"]
    cids = [f"c{i}" for i in range(len(component_samples))]
    zipped = list(zip(cids, component_samples))

    bp_ids = [s["blueprint_binding"]["blueprint_id"] for s in component_samples]
    for i in range(len(bp_ids)):
        for j in range(i + 1, len(bp_ids)):
            pc = classify_pair(profiles[bp_ids[i]], profiles[bp_ids[j]])
            if pc.classification == ComposabilityClass.INCOMPATIBLE.value:
                return None, [f"INCOMPATIBLE_PAIR:{bp_ids[i]}:{bp_ids[j]}"]

    if use_shared_entity is None:
        use_shared_entity = rng.random() < 0.25

    entity_bindings: list[tuple[str, list, dict]] = []
    shared_map_all: dict[str, str] = {}

    if use_shared_entity and len(component_samples) >= 2:
        a, b = component_samples[0], component_samples[1]
        ents_a = a["blueprint_binding"].get("entities") or []
        ents_b = b["blueprint_binding"].get("entities") or []
        roles_a = [_entity_role(e) for e in ents_a]
        roles_b = [_entity_role(e) for e in ents_b]
        proposed = propose_shared_mapping(ents_a, ents_b, role_a=roles_a, role_b=roles_b)
        if proposed:
            shared_map_all = proposed
            trace.append("SHARED_ENTITY_SELECTED")
        else:
            use_shared_entity = False
            trace.append("SHARED_ENTITY_FALLBACK_INDEPENDENT")

    for cid, sample in zipped:
        ents = sample["blueprint_binding"].get("entities") or []
        comp_shared = {e: shared_map_all[e] for e in ents if e in shared_map_all} if use_shared_entity else {}
        if comp_shared:
            recs, meta = bind_entities_shared(cid, ents, comp_shared, reason="SEMANTICALLY_COMPATIBLE_SHARED")
        else:
            recs, meta = bind_entities_independent(cid, ents)
        entity_bindings.append((cid, recs, meta))

    temporal = align_temporal_components(zipped, force_synthetic=force_synthetic_temporal)
    trace.append(f"TEMPORAL:{temporal.get('source_mode')}")

    shared_runtime, runtime_outcome = _resolve_shared_runtime(zipped, shared_map_all)
    if runtime_outcome == "SHARED_STATE_CONFLICT":
        return None, ["SHARED_STATE_CONFLICT"]

    scenes = {s["scene_type"] for s in component_samples}
    if len(scenes) == 1:
        comp_type = "SAME_SCENE"
    else:
        comp_type = "CROSS_SCENE"

    record = build_multi_action_record(
        multi_action_id="pending",
        component_samples=zipped,
        entity_bindings=entity_bindings,
        temporal=temporal,
        shared_runtime=shared_runtime,
        construction_trace=trace,
        composition_type=comp_type,
    )
    sig = record["canonical_parent_signature"]
    record["multi_action_id"] = multi_action_id_from_signature(sig, seed=rng.randint(0, 2**31), index=0)
    record["construction_validator_result"] = {"pending": True}
    return record, trace
