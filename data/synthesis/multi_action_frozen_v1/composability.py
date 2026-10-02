from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from itertools import combinations
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.blueprint_profile import BlueprintProfile

class ComposabilityClass(str, Enum):
    COMPOSABLE = "COMPOSABLE"
    CONDITIONALLY_COMPOSABLE = "CONDITIONALLY_COMPOSABLE"
    INCOMPATIBLE = "INCOMPATIBLE"

class IncompatibilityReason(str, Enum):
    ENTITY_ROLE_CONFLICT = "ENTITY_ROLE_CONFLICT"
    ENTITY_DOMAIN_CONFLICT = "ENTITY_DOMAIN_CONFLICT"
    TEMPORAL_INCOMPATIBILITY = "TEMPORAL_INCOMPATIBILITY"
    SOURCE_INCOMPATIBILITY = "SOURCE_INCOMPATIBILITY"
    UNRESOLVABLE_SHARED_ENTITY = "UNRESOLVABLE_SHARED_ENTITY"
    ACTION_PARAMETER_CONFLICT = "ACTION_PARAMETER_CONFLICT"
    INSUFFICIENT_GROUNDED_ALIGNMENT = "INSUFFICIENT_GROUNDED_ALIGNMENT"
    OTHER = "OTHER"

LIGHTING_SCENES = frozenset({"advanced_lighting", "on_off_schedule", "scene_schedule_override"})
CLIMATE_SCENES = frozenset({"climate_window"})
SECURITY_SCENES = frozenset({"notification_security"})

@dataclass
class PairComposability:
    blueprint_a: str
    blueprint_b: str
    scene_a: str
    scene_b: str
    classification: str
    reasons: list[str]
    shared_entity_opportunity: bool
    independent_entity_default: bool
    source_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _domains_overlap(a: BlueprintProfile, b: BlueprintProfile) -> set[str]:
    da = {r for r in a.entity_roles if r in {"light", "climate", "motion", "power", "contact", "window", "presence"}}
    db = {r for r in b.entity_roles if r in {"light", "climate", "motion", "power", "contact", "window", "presence"}}
    return da & db

def _shared_entity_viable(a: BlueprintProfile, b: BlueprintProfile) -> tuple[bool, list[str]]:
    overlap = _domains_overlap(a, b)
    if not overlap:
        return False, []
    reasons: list[str] = []
    if "power" in overlap and a.scene != b.scene:
        if a.scene in CLIMATE_SCENES or b.scene in CLIMATE_SCENES:
            reasons.append(IncompatibilityReason.ENTITY_ROLE_CONFLICT.value)
            return False, reasons
    if "climate" in overlap and a.action_domain != b.action_domain:
        if b.action_domain in {"light", "switch"} or a.action_domain in {"light", "switch"}:
            reasons.append(IncompatibilityReason.ENTITY_DOMAIN_CONFLICT.value)
            return False, reasons
    if overlap <= {"motion", "lux"}:
        return True, []
    if overlap == {"light"} and a.scene in LIGHTING_SCENES and b.scene in LIGHTING_SCENES:
        return True, []
    if overlap == {"climate"} and a.scene in CLIMATE_SCENES and b.scene in CLIMATE_SCENES:
        return True, []
    if overlap == {"contact", "window"} and a.scene in SECURITY_SCENES and b.scene in SECURITY_SCENES:
        return True, []
    return False, []

def classify_pair(a: BlueprintProfile, b: BlueprintProfile) -> PairComposability:
    reasons: list[str] = []
    bp_a, bp_b = a.blueprint_id, b.blueprint_id
    if a.sample_count == 0 or b.sample_count == 0:
        return PairComposability(
            blueprint_a=bp_a,
            blueprint_b=bp_b,
            scene_a=a.scene,
            scene_b=b.scene,
            classification=ComposabilityClass.INCOMPATIBLE.value,
            reasons=[IncompatibilityReason.INSUFFICIENT_GROUNDED_ALIGNMENT.value],
            shared_entity_opportunity=False,
            independent_entity_default=True,
            source_modes=[],
        )

    shared_ok, shared_block = _shared_entity_viable(a, b)
    source_modes = ["SAME_SOURCE", "CROSS_SOURCE"]
    ds_a = set(a.source_dataset_capabilities)
    ds_b = set(b.source_dataset_capabilities)
    if ds_a != ds_b or len(ds_a | ds_b) > 1:
        source_modes.append("CROSS_DATASET_SYNTHETIC")

    if a.blueprint_id == b.blueprint_id:
        if a.grounded_instance_count < 2:
            reasons.append(IncompatibilityReason.UNRESOLVABLE_SHARED_ENTITY.value)
            classification = ComposabilityClass.CONDITIONALLY_COMPOSABLE.value
        else:
            classification = ComposabilityClass.COMPOSABLE.value
        return PairComposability(
            blueprint_a=bp_a,
            blueprint_b=bp_b,
            scene_a=a.scene,
            scene_b=b.scene,
            classification=classification,
            reasons=reasons or ["SAME_BLUEPRINT_DIFFERENT_INSTANCES"],
            shared_entity_opportunity=False,
            independent_entity_default=True,
            source_modes=source_modes,
        )

    if a.action_domain == b.action_domain == "climate" and a.scene in CLIMATE_SCENES and b.scene in CLIMATE_SCENES:
        if set(a.action_path_signatures) & set(b.action_path_signatures):
            reasons.append(IncompatibilityReason.ACTION_PARAMETER_CONFLICT.value)

    if a.scene == "visual_fusion" or b.scene == "visual_fusion":
        if not reasons:
            classification = ComposabilityClass.CONDITIONALLY_COMPOSABLE.value
            return PairComposability(
                blueprint_a=bp_a,
                blueprint_b=bp_b,
                scene_a=a.scene,
                scene_b=b.scene,
                classification=classification,
                reasons=["VISUAL_FUSION_CROSS_DOMAIN"],
                shared_entity_opportunity=False,
                independent_entity_default=True,
                source_modes=source_modes,
            )

    if shared_block:
        if len(shared_block) >= 2:
            return PairComposability(
                blueprint_a=bp_a,
                blueprint_b=bp_b,
                scene_a=a.scene,
                scene_b=b.scene,
                classification=ComposabilityClass.INCOMPATIBLE.value,
                reasons=shared_block,
                shared_entity_opportunity=False,
                independent_entity_default=True,
                source_modes=source_modes,
            )

    if reasons:
        classification = ComposabilityClass.CONDITIONALLY_COMPOSABLE.value
    elif shared_ok:
        classification = ComposabilityClass.CONDITIONALLY_COMPOSABLE.value
    else:
        classification = ComposabilityClass.COMPOSABLE.value

    return PairComposability(
        blueprint_a=bp_a,
        blueprint_b=bp_b,
        scene_a=a.scene,
        scene_b=b.scene,
        classification=classification,
        reasons=reasons or (["SHARED_ENTITY_OPTIONAL"] if shared_ok else ["INDEPENDENT_ENTITIES"]),
        shared_entity_opportunity=shared_ok,
        independent_entity_default=True,
        source_modes=source_modes,
    )

def classify_triple(a: BlueprintProfile, b: BlueprintProfile, c: BlueprintProfile) -> dict[str, Any]:
    pairs = [
        classify_pair(a, b),
        classify_pair(a, c),
        classify_pair(b, c),
    ]
    classes = {p.classification for p in pairs}
    if ComposabilityClass.INCOMPATIBLE.value in classes:
        cls = ComposabilityClass.INCOMPATIBLE.value
    elif ComposabilityClass.CONDITIONALLY_COMPOSABLE.value in classes:
        cls = ComposabilityClass.CONDITIONALLY_COMPOSABLE.value
    else:
        cls = ComposabilityClass.COMPOSABLE.value
    return {
        "blueprints": sorted([a.blueprint_id, b.blueprint_id, c.blueprint_id]),
        "scenes": sorted({a.scene, b.scene, c.scene}),
        "classification": cls,
        "pair_results": [p.to_dict() for p in pairs],
    }

def build_composability_matrix(
    profiles: dict[str, BlueprintProfile],
    *,
    include_triples_summary: bool = True,
) -> dict[str, Any]:
    grounded = {k: v for k, v in profiles.items() if v.sample_count > 0}
    bp_ids = sorted(grounded.keys())
    pairs: list[dict] = []
    counts = {c.value: 0 for c in ComposabilityClass}
    reason_counts: dict[str, int] = {}

    for i, j in combinations(range(len(bp_ids)), 2):
        a, b = grounded[bp_ids[i]], grounded[bp_ids[j]]
        pc = classify_pair(a, b)
        pairs.append(pc.to_dict())
        counts[pc.classification] += 1
        for r in pc.reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1

    triple_summary = {"total": 0, "COMPOSABLE": 0, "CONDITIONALLY_COMPOSABLE": 0, "INCOMPATIBLE": 0}
    if include_triples_summary:
        for combo in combinations(bp_ids, 3):
            profs = [grounded[x] for x in combo]
            t = classify_triple(profs[0], profs[1], profs[2])
            triple_summary["total"] += 1
            triple_summary[t["classification"]] += 1

    scene_pair_coverage: dict[str, int] = {}
    shared_opportunities = 0
    source_mode_ops: dict[str, int] = {"SAME_SOURCE": 0, "CROSS_SOURCE": 0, "CROSS_DATASET_SYNTHETIC": 0}
    for p in pairs:
        key = f"{p['scene_a']}::{p['scene_b']}" if p["scene_a"] <= p["scene_b"] else f"{p['scene_b']}::{p['scene_a']}"
        scene_pair_coverage[key] = scene_pair_coverage.get(key, 0) + 1
        if p.get("shared_entity_opportunity"):
            shared_opportunities += 1
        for m in p.get("source_modes") or []:
            source_mode_ops[m] = source_mode_ops.get(m, 0) + 1

    generated_nodes = [profiles[k].to_dict() for k in sorted(profiles) if profiles[k].status == "GROUNDED_CAPABLE"]
    return {
        "generated_blueprint_nodes": generated_nodes,
        "generated_blueprint_count": len(generated_nodes),
        "grounded_capable_with_samples": len(grounded),
        "pair_combinations_total": len(pairs),
        "pair_counts": counts,
        "triple_summary": triple_summary,
        "top_incompatibility_reasons": sorted(reason_counts.items(), key=lambda x: -x[1])[:15],
        "scene_pair_coverage": dict(sorted(scene_pair_coverage.items())),
        "shared_entity_opportunities": shared_opportunities,
        "source_mode_opportunities": source_mode_ops,
        "pairs": pairs,
    }
