from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import OUTPUT_DIR
from smarthome_mdf.multi_action_frozen_v1.information_flow import run_information_flow_audit
from smarthome_mdf.multi_action_frozen_v1.sample_index import build_sample_index, estimate_capacities
from smarthome_mdf.multi_action_frozen_v1.blueprint_profile import build_blueprint_profiles
from smarthome_mdf.multi_action_frozen_v1.frozen_loader import load_frozen_samples

def audit_formal_corpus(
    samples_path: Path,
    *,
    capacities: dict[str, Any] | None = None,
    pilot_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples: list[dict] = []
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    bp_usage: Counter = Counter()
    scene_usage: Counter = Counter()
    pairs: Counter = Counter()
    triples: Counter = Counter()
    source_modes: Counter = Counter()
    temporal_modes: Counter = Counter()
    same_scene = 0
    cross_scene = 0
    shared = 0
    independent = 0
    sigs: set[str] = set()
    comp_tuples: set[frozenset] = set()
    dup_sigs = 0
    dup_tuples = 0
    provenance_invalid = 0
    temporal_invalid = 0
    entity_invalid = 0

    for s in samples:
        comps = s.get("components") or []
        bps = [c.get("blueprint_id") for c in comps]
        for bp in bps:
            bp_usage[bp] += 1
        for c in comps:
            scene_usage[c.get("scene", "")] += 1
        if len(bps) == 2:
            pairs[tuple(sorted(bps))] += 1
        elif len(bps) == 3:
            triples[tuple(sorted(bps))] += 1
        source_modes[s.get("source_mode") or "UNKNOWN"] += 1
        tm = (s.get("composition_timeline") or {}).get("alignment_mode") or "UNKNOWN"
        temporal_modes[tm] += 1
        if s.get("composition_type") == "SAME_SCENE":
            same_scene += 1
        else:
            cross_scene += 1
        if (s.get("shared_entities") or {}).get("shared_entity"):
            shared += 1
        else:
            independent += 1
        sig = s.get("canonical_parent_signature")
        if sig in sigs:
            dup_sigs += 1
        sigs.add(sig)
        ct = frozenset(c.get("single_scene_sample_id") for c in comps)
        if ct in comp_tuples:
            dup_tuples += 1
        comp_tuples.add(ct)
        val = s.get("construction_validator_result") or {}
        for r in val.get("reasons") or []:
            if "PROVENANCE" in r:
                provenance_invalid += 1
            elif "TEMPORAL" in r:
                temporal_invalid += 1
            elif "ENTITY" in r:
                entity_invalid += 1

    frozen, _ = load_frozen_samples(verify=True)
    profiles = build_blueprint_profiles(frozen)
    index = build_sample_index(frozen, profiles)
    if capacities is None:
        capacities = estimate_capacities(index)

    pair_freqs = list(pairs.values()) if pairs else [0]
    triple_freqs = list(triples.values()) if triples else [0]

    info_flow = run_information_flow_audit()
    grounded_bps = sorted(k for k, v in profiles.items() if v.sample_count > 0)

    report = {
        "sample_count": len(samples),
        "two_component_count": sum(1 for s in samples if s.get("component_count") == 2),
        "three_component_count": sum(1 for s in samples if s.get("component_count") == 3),
        "blueprint_usage": dict(bp_usage),
        "all_21_blueprint_usage": {bp: bp_usage.get(bp, 0) for bp in grounded_bps},
        "scene_usage": dict(scene_usage),
        "pair_coverage": {
            "eligible_pair_count": capacities.get("eligible_pair_count", len(index.legal_pairs)),
            "represented_pair_count": len(pairs),
            "coverage_rate": len(pairs) / max(1, len(index.legal_pairs)),
            "min_frequency": min(pair_freqs),
            "median_frequency": statistics.median(pair_freqs),
            "max_frequency": max(pair_freqs),
        },
        "triple_coverage": {
            "eligible_triple_count": capacities.get("eligible_triple_count", len(index.legal_triples)),
            "represented_triple_count": len(triples),
            "coverage_rate": len(triples) / max(1, len(index.legal_triples)),
            "min_frequency": min(triple_freqs),
            "median_frequency": statistics.median(triple_freqs),
            "max_frequency": max(triple_freqs),
        },
        "same_scene_count": same_scene,
        "cross_scene_count": cross_scene,
        "source_modes": dict(source_modes),
        "temporal_modes": dict(temporal_modes),
        "shared_entity_count": shared,
        "independent_entity_count": independent,
        "same_source_count": source_modes.get("SAME_SOURCE", 0),
        "same_source_capacity_estimate": capacities.get("same_source_capacity_estimate"),
        "shared_entity_capacity_estimate": capacities.get("shared_entity_capacity_estimate"),
        "hard_blockers": {
            "PROVENANCE_INVALID": provenance_invalid,
            "TEMPORAL_ALIGNMENT_INVALID": temporal_invalid,
            "ENTITY_BINDING_INVALID": entity_invalid,
            "DUPLICATE_PARENT_SIGNATURE": dup_sigs,
            "identical_component_tuple_duplicate": dup_tuples,
            "INFORMATION_FLOW_VIOLATION": info_flow["information_flow_violation_count"],
        },
        "information_flow_audit": info_flow,
        "remaining_capacity": {
            "remaining_pair_opportunities": capacities.get("eligible_pair_count", 0) - len(pairs),
            "remaining_triple_opportunities": capacities.get("eligible_triple_count", 0) - len(triples),
            "remaining_same_source_opportunities": max(
                0, capacities.get("same_source_capacity_estimate", 0) - source_modes.get("SAME_SOURCE", 0)
            ),
            "remaining_shared_entity_opportunities": max(
                0, capacities.get("shared_entity_capacity_estimate", 0) - shared
            ),
        },
        "pilot_regression": {
            "pilot": pilot_stats or {},
            "formal": {
                "samples": len(samples),
                "unique_pairs": len(pairs),
                "unique_triples": len(triples),
                "same_source": source_modes.get("SAME_SOURCE", 0),
                "shared_entity": shared,
            },
        },
    }

    blockers = report["hard_blockers"]
    all_bps_present = all(bp_usage.get(bp, 0) > 0 for bp in grounded_bps)
    target_ok = len(samples) == 2400
    if target_ok and all_bps_present and not any(v > 0 for v in blockers.values()):
        report["verdict"] = "READY_FOR_MULTI_ACTION_FINAL_VALIDATION"
    else:
        report["verdict"] = "BLOCKED_BEFORE_MULTI_ACTION_FINAL_VALIDATION"
        report["blockers"] = [
            k for k, v in blockers.items() if v > 0
        ] + ([] if target_ok else ["TARGET_NOT_REACHED"]) + ([] if all_bps_present else ["BLUEPRINT_COVERAGE_INCOMPLETE"])
    return report

def write_full_generation_md(report: dict[str, Any], path: Path, *, manifest: dict[str, Any]) -> None:
    lines = [
        "# Multi-action Frozen V1 Full Generation Report",
        "",
        f"**Verdict:** `{report.get('verdict')}`",
        f"**Stop reason:** `{manifest.get('stop_reason')}`",
        "",
        "## Generation",
        f"- Target: {manifest.get('target_total')}",
        f"- Accepted: {report['sample_count']}",
        f"- Attempts: {manifest.get('attempts')}",
        f"- Acceptance rate: {manifest.get('acceptance_rate'):.2%}",
        f"- Seed: {manifest.get('full_generation_seed')}",
        f"- Frozen SHA256: `{manifest.get('frozen_v1_sample_sha256')}`",
        "",
        "## Component distribution",
        f"- 2-component: {report['two_component_count']}",
        f"- 3-component: {report['three_component_count']}",
        "",
        "## Coverage",
        f"- Blueprint pairs: {report['pair_coverage']['represented_pair_count']} / {report['pair_coverage']['eligible_pair_count']} ({report['pair_coverage']['coverage_rate']:.1%})",
        f"- Blueprint triples: {report['triple_coverage']['represented_triple_count']} / {report['triple_coverage']['eligible_triple_count']} ({report['triple_coverage']['coverage_rate']:.1%})",
        f"- Same-scene: {report['same_scene_count']}",
        f"- Cross-scene: {report['cross_scene_count']}",
        "",
        "## Source modes",
    ]
    for k, v in report.get("source_modes", {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(
        [
            "",
            "## Entity modes",
            f"- SHARED_ENTITY: {report['shared_entity_count']}",
            f"- INDEPENDENT_ENTITY: {report['independent_entity_count']}",
            "",
            "## Hard blockers",
        ]
    )
    for k, v in report.get("hard_blockers", {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Pilot regression", ""])
    pr = report.get("pilot_regression", {})
    if pr.get("pilot"):
        lines.append(f"- Pilot: {pr['pilot']}")
    lines.append(f"- Formal: {pr.get('formal')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
