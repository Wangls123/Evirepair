from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.blueprint_profile import build_blueprint_profiles
from smarthome_mdf.multi_action_frozen_v1.composer import compose_multi_action
from smarthome_mdf.multi_action_frozen_v1.config import (
    COMPOSABILITY_POLICY_VERSION,
    CONSTRUCTION_VALIDATOR_VERSION,
    ENTITY_BINDING_VERSION,
    EXPECTED_FROZEN_SHA256,
    FORMAL_TARGET_TOTAL,
    FORMAL_TWO_COMPONENT_TARGET,
    PARENT_SIGNATURE_VERSION,
    SCHEDULER_VERSION,
    SCHEMA_VERSION,
    TEMPORAL_ALIGNMENT_VERSION,
)
from smarthome_mdf.multi_action_frozen_v1.construction_validator import validate_construction
from smarthome_mdf.multi_action_frozen_v1.coverage_scheduler import CoverageScheduler
from smarthome_mdf.multi_action_frozen_v1.frozen_loader import load_corpus_from_path, load_frozen_samples, verify_frozen_integrity
from smarthome_mdf.multi_action_frozen_v1.sample_index import CoverageState, build_sample_index, estimate_capacities

def _reject_category(reason: str) -> str:
    for prefix in (
        "DUPLICATE_PARENT_SIGNATURE",
        "COMPONENT_SET_DUPLICATE",
        "ENTITY_BINDING_INVALID",
        "SHARED_STATE_CONFLICT",
        "TEMPORAL_ALIGNMENT_INVALID",
        "PROVENANCE_INVALID",
        "MISSING_FROZEN_SAMPLE",
        "COMPONENT_REFERENCE_INVALID",
        "RUNTIME_INVALID",
        "SOURCE_MODE_INVALID",
        "INCOMPATIBLE_PAIR",
        "FORBIDDEN_FIELD",
    ):
        if reason.startswith(prefix) or prefix in reason:
            return prefix
    return "OTHER"

def run_generation(
    *,
    target: int,
    seed: int,
    output_dir: Path,
    resume: bool = False,
    checkpoint_interval: int = 100,
    upstream_path: Path | None = None,
    scene_filter: set[str] | frozenset[str] | None = None,
    two_component_target: int | None = None,
    skip_frozen_sha_check: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if upstream_path is not None:
        samples, meta = load_corpus_from_path(
            upstream_path,
            scene_filter=scene_filter,
            verify_sha=None,
        )
    else:
        if not skip_frozen_sha_check:
            meta = verify_frozen_integrity()
            if meta["sample_sha256"] != EXPECTED_FROZEN_SHA256:
                raise RuntimeError(
                    f"Frozen SHA256 mismatch: expected {EXPECTED_FROZEN_SHA256}, got {meta['sample_sha256']}"
                )
        samples, meta = load_frozen_samples(verify=not skip_frozen_sha_check)
        if scene_filter:
            samples, meta = load_corpus_from_path(
                Path(meta["path"]),
                meta=meta,
                expected_count=None,
                scene_filter=scene_filter,
            )
    profiles = build_blueprint_profiles(samples)
    index = build_sample_index(samples, profiles)
    capacities = estimate_capacities(index)

    samples_path = output_dir / "multi_action_samples.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"

    accepted: list[dict] = []
    state = CoverageState()
    reject_taxonomy: Counter = Counter()
    attempts = 0
    two_accepted = 0
    three_accepted = 0

    if resume and checkpoint_path.is_file():
        ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        state = CoverageState.from_dict(ck.get("coverage_state") or {})
        attempts = ck.get("attempts", 0)
        two_accepted = ck.get("two_accepted", 0)
        three_accepted = ck.get("three_accepted", 0)
        reject_taxonomy = Counter(ck.get("reject_taxonomy") or {})
        if samples_path.is_file():
            with samples_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        accepted.append(json.loads(line))
        if len(accepted) != two_accepted + three_accepted:
            two_accepted = sum(1 for a in accepted if a.get("component_count") == 2)
            three_accepted = sum(1 for a in accepted if a.get("component_count") == 3)

    rng = random.Random(seed + len(accepted))
    two_target = two_component_target if two_component_target is not None else int(target * 0.6)
    scheduler = CoverageScheduler(
        index,
        target_total=target,
        two_component_target=two_target,
        rng=rng,
        same_source_target=min(120, max(40, target // 20)),
        shared_entity_target=min(120, max(60, target // 20)),
    )

    seen_sigs = set(state.parent_signatures_seen)
    seen_tuples = set(state.component_tuple_seen)
    max_attempts = target * 50

    while len(accepted) < target and attempts < max_attempts:
        attempts += 1
        req = scheduler.next_request_balanced(state, two_accepted, three_accepted)
        comp_samples = scheduler.materialize_samples(req)
        if not comp_samples:
            reject_taxonomy["STRATUM_CAPACITY_EXHAUSTED"] += 1
            continue

        tuple_key = frozenset(s["sample_id"] for s in comp_samples)
        if tuple_key in seen_tuples:
            reject_taxonomy["COMPONENT_SET_DUPLICATE"] += 1
            continue

        record, trace = compose_multi_action(
            comp_samples,
            profiles,
            rng=rng,
            use_shared_entity=req.use_shared_entity,
            force_synthetic_temporal=req.force_synthetic,
        )
        if record is None:
            for t in trace:
                reject_taxonomy[_reject_category(t)] += 1
            continue

        val = validate_construction(
            record,
            frozen_index=index.by_id,
            seen_signatures=seen_sigs,
            seen_component_tuples=seen_tuples,
        )
        record["construction_validator_result"] = val
        if val["verdict"] != "CONSTRUCTION_ACCEPT":
            for r in val["reasons"]:
                reject_taxonomy[_reject_category(r)] += 1
            continue

        accepted.append(record)
        state.record_accept(record)
        if record["component_count"] == 2:
            two_accepted += 1
        else:
            three_accepted += 1

        with samples_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if len(accepted) % checkpoint_interval == 0:
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "accepted_count": len(accepted),
                        "two_accepted": two_accepted,
                        "three_accepted": three_accepted,
                        "attempts": attempts,
                        "coverage_state": state.to_dict(),
                        "reject_taxonomy": dict(reject_taxonomy),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    stop_reason = "GLOBAL_MULTI_ACTION_TARGET_REACHED" if len(accepted) >= target else "GROUNDED_COMPOSITION_SPACE_EXHAUSTED"

    return {
        "accepted": accepted,
        "attempts": attempts,
        "two_accepted": two_accepted,
        "three_accepted": three_accepted,
        "stop_reason": stop_reason,
        "reject_taxonomy": dict(reject_taxonomy),
        "coverage_state": state,
        "capacities": capacities,
        "frozen_sha256": meta.get("sample_sha256"),
        "upstream_path": meta.get("path"),
        "scene_filter": meta.get("scene_filter"),
        "seed": seed,
        "output_dir": str(output_dir),
        "samples_path": str(samples_path),
    }
