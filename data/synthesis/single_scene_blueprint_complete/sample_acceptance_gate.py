from __future__ import annotations

from typing import Any

from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle, minimal_grounding_requirements
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import satisfies_requirement
from smarthome_mdf.single_scene_blueprint_complete.blueprint_evidence_enrichment import repair_sample_evidence_valid
from smarthome_mdf.single_scene_blueprint_complete.grounded_instance_snapshot import build_ir_for_sample
from smarthome_mdf.single_scene_blueprint_complete.pre_freeze_verification import sample_to_source_observation
from smarthome_mdf.single_scene_blueprint_complete.semantic_sanitize import detect_semantic_contamination
from smarthome_mdf.single_scene_blueprint_complete.semantic_capabilities import attach_capabilities
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    behavior_source_dedup_key,
    is_dynamic_behavior_target,
    is_timestamp_only_duplicate,
)
from smarthome_mdf.single_scene_blueprint_complete.config import FORBIDDEN_SYNTHESIS_FIELDS

def validate_sample_accept(
    sample: dict,
    *,
    pool: Any,
    normalized_blueprint: Any,
    bundle: Any | None = None,
    prior_by_canonical: dict[str, dict] | None = None,
) -> tuple[bool, str]:

    forbidden = {f.lower() for f in FORBIDDEN_SYNTHESIS_FIELDS}

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.lower() in forbidden:
                    return f"forbidden_field:{path}.{k}"
                r = walk(v, f"{path}.{k}" if path else k)
                if r:
                    return r
        elif isinstance(node, list):
            for i, item in enumerate(node):
                r = walk(item, f"{path}[{i}]")
                if r:
                    return r
        return None

    hit = walk(sample)
    if hit:
        return False, hit

    meta = sample.get("synthesis_metadata") or {}
    if is_dynamic_behavior_target(meta.get("behavior_target")):
        return False, "DYNAMIC_BEHAVIOR_TARGET_SUFFIX"

    if detect_semantic_contamination(sample):
        return False, "SEMANTIC_CONTAMINATION"

    bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
    ok_ev, ev_reason = repair_sample_evidence_valid(sample, bp)
    if not ok_ev:
        return False, f"REPAIR_EVIDENCE_FAIL:{ev_reason}"

    scene = sample.get("scene_type", "")
    if scene == "visual_fusion":
        prov = sample.get("provenance") or {}
        if not prov.get("runtime_source") or not prov.get("visual_source"):
            return False, "VISUAL_PROVENANCE_INCOMPLETE"
        if prov.get("runtime_source", {}).get("dataset") != prov.get("visual_source", {}).get("dataset"):
            if prov.get("composition_type") != "CROSS_DATASET_SYNTHETIC":
                return False, "CROSS_DATASET_PROVENANCE_IMPLICIT"

    obs = sample_to_source_observation(sample, pool)
    if obs is None:
        return False, "NO_OBSERVATION"

    obs = attach_capabilities(obs)
    if bundle is None:
        ir_built, _inst, nb_built = build_ir_for_sample(sample)
        nb_use = normalized_blueprint if getattr(normalized_blueprint, "parse_status", None) == "PARSE_OK" else nb_built
        bundle = extract_requirement_bundle(nb_use, ir=ir_built)
    for req in minimal_grounding_requirements(bundle):
        ok, unsat, _ = satisfies_requirement(req, obs)
        if not ok:
            return False, f"RUNTIME_MATCH_FAIL:{','.join(unsat[:3])}"

    if prior_by_canonical is not None:
        dedup_key = behavior_source_dedup_key(sample)
        prior = prior_by_canonical.get(dedup_key)
        if prior is not None:
            if is_timestamp_only_duplicate(sample, prior):
                return False, "TIMESTAMP_ONLY_DUPLICATE"
            from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import canonical_sample_signature

            if canonical_sample_signature(sample) == canonical_sample_signature(prior):
                return False, "SAME_SOURCE_DUPLICATE_PADDING"

    return True, "ok"
