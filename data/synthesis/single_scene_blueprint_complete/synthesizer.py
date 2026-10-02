from __future__ import annotations

import copy
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle
from smarthome_mdf.multi_action_vnext.source_adapters import SourceObservation, SourcePool
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import match_requirement_bundle
from smarthome_mdf.single_scene_blueprint_complete.config import FORBIDDEN_SYNTHESIS_FIELDS, SYNTHESIS_METHOD
from smarthome_mdf.single_scene_blueprint_complete.blueprint_evidence_enrichment import enrich_sample_evidence, repair_sample_evidence_valid
from smarthome_mdf.single_scene_blueprint_complete.diversity_source_selector import DiversitySourceSelector
from smarthome_mdf.single_scene_blueprint_complete.generation_path_request import GenerationPathRequest
from smarthome_mdf.single_scene_blueprint_complete.grounded_instance_snapshot import grounded_instance_snapshot
from smarthome_mdf.single_scene_blueprint_complete.p1_visual_compose import compose_visual_fusion_observed, pick_youhome_visual_pool
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import sanitize_attributes
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    canonical_behavior_hash,
    stable_action_node_id,
    stable_action_path_id,
    stable_branch_id,
    strip_dynamic_behavior_suffix,
)
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path

_IR_CACHE: dict[str, Any] = {}
_VISUAL_POOL_CACHE: dict[str, list] = {}

def _get_ir(blueprint_id: str, instance: dict, automation_instance_id: str):
    key = f"{blueprint_id}::{automation_instance_id}"
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    ypath = resolve_yaml_path(blueprint_id)
    ir = None
    if ypath and ypath.is_file():
        ir = parse_blueprint_ir(
            blueprint_id,
            Path(ypath),
            dict(instance.get("blueprint_inputs") or {}),
            automation_instance_id=automation_instance_id,
        )
    _IR_CACHE[key] = ir
    return ir

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _forbidden_keys(obj: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    forbidden = {f.lower() for f in FORBIDDEN_SYNTHESIS_FIELDS}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}" if path else k
                if k.lower() in forbidden:
                    found.append(p)
                walk(v, p)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(obj, prefix)
    return found

def _diversity_signature(sample: dict) -> str:

    return canonical_behavior_hash(sample)

def _build_entity_observations(obs: SourceObservation, src_sample: dict) -> list[dict]:
    entity_obs = copy.deepcopy(src_sample.get("entity_observations") or [])
    if not entity_obs:
        raw_attrs = dict(obs.attributes or {})
        dc = raw_attrs.get("device_class")
        clean_attrs, _ = sanitize_attributes(raw_attrs, device_class=str(dc) if dc else None)
        entity_obs = [
            {
                "entity_id": f"{obs.entity_domain or 'sensor'}.grounded",
                "domain": obs.entity_domain or "sensor",
                "state": obs.state,
                "source_dataset": obs.dataset,
                "source_entity": obs.record_id,
                "timestamp": obs.original_timestamp or _utc_now(),
                "fusion_role": "terminal_reading",
                "attributes": clean_attrs,
            }
        ]
    else:
        for eo in entity_obs:
            attrs = dict(eo.get("attributes") or {})
            dc = attrs.get("device_class") or eo.get("domain")
            clean, _ = sanitize_attributes(attrs, device_class=str(dc) if dc else None)
            eo["attributes"] = clean
    return entity_obs

def _observed_from_runtime(obs: SourceObservation) -> dict:
    attrs = dict(obs.attributes or {})
    out: dict[str, Any] = {}
    if obs.measurement == "power" or attrs.get("current_power_w") is not None:
        out["current_power_w"] = attrs.get("current_power_w")
    if attrs.get("motion_state") is not None:
        out["motion_state"] = attrs.get("motion_state")
    if attrs.get("window_state") is not None:
        out["window_state"] = attrs.get("window_state")
    if attrs.get("temperature") is not None:
        out["temperature"] = attrs.get("temperature")
    if attrs.get("illuminance_lux") is not None:
        out["illuminance_lux"] = attrs.get("illuminance_lux")
    if obs.original_timestamp:
        out["timestamp"] = obs.original_timestamp
    return out

def synthesize_single_scene(
    *,
    scene: str,
    blueprint_id: str,
    automation_instance_id: str,
    behavior_target: str,
    branch_id: str,
    action_path_signature: str,
    instance: dict,
    normalized_blueprint: Any,
    behavior_spec: dict,
    source_pool: SourcePool,
    reuse_counts: dict[str, int],
    attempt_index: int = 0,
    visual_frame_index: int | None = None,
    diversity_selector: DiversitySourceSelector | None = None,
    generation_path_request: GenerationPathRequest | None = None,
    target_template_key: str | None = None,
) -> tuple[str, dict | None, str]:

    ir = _get_ir(blueprint_id, instance, automation_instance_id)
    nb = normalized_blueprint
    if ir is not None:
        prov = dict(nb.provenance or {})
        prov["automation_instance_id"] = automation_instance_id
        prov["blueprint_inputs"] = dict(instance.get("blueprint_inputs") or {})
        nb.provenance = prov
    if nb.parse_status != "PARSE_OK":
        return "IMPLEMENTATION_GAP", None, nb.rejection_reason or "parser_not_ready"
    ypath = resolve_yaml_path(blueprint_id)
    if not ypath or not ypath.is_file():
        return "YAML_MISSING", None, "canonical_yaml_missing"

    bundle = extract_requirement_bundle(nb, ir=ir)
    if generation_path_request is not None:
        branch_id = generation_path_request.target_branch
        action_path_signature = generation_path_request.target_signature
        canonical_bt = strip_dynamic_behavior_suffix(
            f"{blueprint_id}::{generation_path_request.target_branch}::{generation_path_request.target_signature}"
        )
    else:
        canonical_bt = strip_dynamic_behavior_suffix(behavior_target)
    entity_sig = "|".join(sorted(instance.get("bound_entities") or []))

    if diversity_selector is not None:
        obs, match_status, traces = diversity_selector.select_source(
            source_pool,
            bundle,
            blueprint_id=blueprint_id,
            automation_instance_id=automation_instance_id,
            branch_id=branch_id,
            action_path_signature=action_path_signature,
            entity_binding_signature=entity_sig,
        )
    else:
        obs, match_status, traces = match_requirement_bundle(
            source_pool,
            bundle,
            seed_offset=(hash(canonical_bt) + attempt_index) % 10000,
        )
    if obs is None or match_status != "MATCH_OK":
        unsat = []
        if traces:
            unsat = traces[0].get("unsatisfied_requirements") or []
        return match_status or "NO_RUNTIME_MATCH", None, ",".join(unsat or ["no_match"])

    src_sample = _find_source_sample(source_pool, scene, obs.record_id)
    sample_id = f"ssc_{blueprint_id[:12]}_{uuid.uuid4().hex[:8]}"

    entity_obs = _build_entity_observations(obs, src_sample)

    provenance_extra: dict[str, Any] = {}
    if scene == "visual_fusion":
        cache_key = scene
        if cache_key not in _VISUAL_POOL_CACHE:
            _VISUAL_POOL_CACHE[cache_key] = pick_youhome_visual_pool(source_pool, scene)
        visual_pool = _VISUAL_POOL_CACHE[cache_key]
        if not visual_pool:
            return "NO_VISUAL_EVIDENCE", None, "no_youhome_frames"
        if diversity_selector is not None:
            vidx = diversity_selector.generation_step % len(visual_pool)
        else:
            vidx = (visual_frame_index if visual_frame_index is not None else attempt_index) % len(visual_pool)
        visual_sample = visual_pool[vidx]
        observed, provenance_extra = compose_visual_fusion_observed(obs, visual_sample)
        visual_observed = observed
    else:
        observed = copy.deepcopy(src_sample.get("observed") or _observed_from_runtime(obs))
        visual_observed = None

    stable_node = stable_action_node_id(
        blueprint_id=blueprint_id,
        branch_id=branch_id,
        action_path_signature=action_path_signature,
    )
    stable_branch = stable_branch_id(blueprint_id=blueprint_id, branch_id=branch_id)
    stable_path = stable_action_path_id(
        blueprint_id=blueprint_id,
        branch_id=branch_id,
        action_path_signature=action_path_signature,
    )

    trace = {
        "blueprint_id": blueprint_id,
        "automation_instance_id": automation_instance_id,
        "behavior_target": canonical_bt,
        "branch_id": branch_id,
        "action_path_signature": action_path_signature,
        "stable_action_node_id": stable_node,
        "trigger_matched": True,
        "match_traces": traces,
        "trace_role": "CONSTRUCTION_VALIDATOR_ONLY",
        "note": "Not B0/Y/GT — construction validation only",
    }

    sample = {
        "sample_id": sample_id,
        "scene_type": scene,
        "sample_type": "blueprint_grounded",
        "fusion_task": "single_scene_blueprint_driven",
        "entity_observations": entity_obs,
        "observed": observed,
        "system_state": {
            **copy.deepcopy(src_sample.get("system_state") or {}),
            "behavior_target": canonical_bt,
            "branch_id": branch_id,
        },
        "blueprint_binding": {
            "blueprint_id": blueprint_id,
            "automation_instance_id": automation_instance_id,
            "entities": list(instance.get("bound_entities") or []),
            "fusion_type": "blueprint_driven_single_scene",
            "blueprint_selected_before_synthesis": True,
            "grounded_instance": grounded_instance_snapshot(instance),
        },
        "provenance": {
            "source_dataset": obs.dataset,
            "source_record_id": obs.record_id,
            "lineage": "Public Raw → Adapter → Grounded SourceObservation → Blueprint Runtime Match → Single-scene Sample",
            "synthesis_timestamp": _utc_now(),
            **provenance_extra,
        },
        "metadata": {
            "schema_version": "single_scene_blueprint_complete_v2",
            "synthesis_method": SYNTHESIS_METHOD,
            "label_status": "construction_only",
        },
        "synthesis_metadata": {
            "behavior_target": canonical_bt,
            "canonical_behavior_target": canonical_bt,
            "branch_id": branch_id,
            "action_path_signature": action_path_signature,
            "stable_action_node_id": stable_node,
            "stable_branch_id": stable_branch,
            "stable_action_path_id": stable_path,
            "blueprint_selected_before_synthesis": True,
            "runtime_situation": match_status,
            "temporal_pattern": (obs.attributes or {}).get("trigger_context") or "point_timestamp",
            "entity_binding_signature": "|".join(sorted(instance.get("bound_entities") or [])),
        },
        "synthesis_construction_trace": trace,
    }
    sample["diversity_signature"] = _diversity_signature(sample)
    sample["canonical_behavior_signature"] = canonical_behavior_hash(sample)

    sample = enrich_sample_evidence(
        sample,
        blueprint_id=blueprint_id,
        instance=instance,
        attempt_index=attempt_index,
        visual_observed=visual_observed if scene == "visual_fusion" else None,
    )
    sample["diversity_signature"] = _diversity_signature(sample)
    sample["canonical_behavior_signature"] = canonical_behavior_hash(sample)

    if generation_path_request is not None:
        from smarthome_mdf.single_scene_blueprint_complete.template_path_resolver import (
            apply_template_path_metadata,
            validate_generated_matches_target,
        )

        ok_path, path_reason = validate_generated_matches_target(sample, generation_path_request, instance)
        if not ok_path:
            return "PATH_MISMATCH", None, path_reason
        sample = apply_template_path_metadata(sample, generation_path_request, instance)
        sample["diversity_signature"] = _diversity_signature(sample)
        sample["canonical_behavior_signature"] = canonical_behavior_hash(sample)
    elif target_template_key:
        meta = dict(sample.get("synthesis_metadata") or {})
        meta["target_template_key"] = target_template_key
        sample["synthesis_metadata"] = meta

    forbidden = _forbidden_keys(sample)
    if forbidden:
        return "FORBIDDEN_FIELD", None, forbidden[0]

    rid = obs.record_id
    reuse_counts[rid] = reuse_counts.get(rid, 0) + 1
    return "ACCEPT", sample, "ok"

def _find_source_sample(pool: SourcePool, scene: str, record_id: str) -> dict:

    for sample in pool._pools.get(scene, []):
        if str(sample.get("sample_id") or "") == record_id:
            return sample
    for obs in pool.candidates_for_scene(scene):
        if str(obs.record_id) == record_id:
            return {}
    return {}
