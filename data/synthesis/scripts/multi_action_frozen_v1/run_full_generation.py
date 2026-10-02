from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.multi_action_frozen_v1.config import (
    COMPOSABILITY_POLICY_VERSION,
    CONSTRUCTION_VALIDATOR_VERSION,
    ENTITY_BINDING_VERSION,
    EXPECTED_FROZEN_SHA256,
    FORMAL_TARGET_TOTAL,
    FULL_GENERATION_DIR,
    FULL_GENERATION_SEED,
    PARENT_SIGNATURE_VERSION,
    SCHEDULER_VERSION,
    SCHEMA_VERSION,
    SMOKE_DIR,
    SMOKE_TARGET,
    TEMPORAL_ALIGNMENT_VERSION,
    OUTPUT_DIR,
)
from smarthome_mdf.multi_action_frozen_v1.full_audit import audit_formal_corpus, write_full_generation_md
from smarthome_mdf.multi_action_frozen_v1.full_generator import run_generation
from smarthome_mdf.multi_action_frozen_v1.frozen_loader import verify_frozen_integrity
from smarthome_mdf.multi_action_frozen_v1.information_flow import run_information_flow_audit

PILOT_STATS = {
    "samples": 250,
    "unique_pairs": 111,
    "unique_triples": 95,
    "same_source": 0,
    "shared_entity": 4,
}

def _audit_smoke(smoke_dir: Path) -> dict:
    samples_path = smoke_dir / "multi_action_samples.jsonl"
    report = audit_formal_corpus(samples_path, pilot_stats=PILOT_STATS)
    blockers = report["hard_blockers"]
    report["smoke_pass"] = not any(v > 0 for v in blockers.values()) and report["sample_count"] >= SMOKE_TARGET * 0.85
    return report

def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)

def main() -> int:
    meta = verify_frozen_integrity()
    if meta["sample_sha256"] != EXPECTED_FROZEN_SHA256:
        print(f"ABORT: SHA256 mismatch expected {EXPECTED_FROZEN_SHA256} got {meta['sample_sha256']}")
        return 2

    info = run_information_flow_audit()
    if info["information_flow_violation_count"] > 0:
        print("ABORT: information flow violations", info)
        return 2

    print("=== SMOKE TEST ===")
    _clean_dir(SMOKE_DIR)
    smoke_result = run_generation(
        target=SMOKE_TARGET,
        seed=FULL_GENERATION_SEED + 1,
        output_dir=SMOKE_DIR,
        resume=False,
    )
    smoke_audit = _audit_smoke(SMOKE_DIR)
    (SMOKE_DIR / "smoke_audit.json").write_text(json.dumps(smoke_audit, indent=2), encoding="utf-8")
    print(f"Smoke accepted: {smoke_audit['sample_count']}")
    print(f"Smoke SAME_SOURCE: {smoke_audit.get('same_source_count', 0)}")
    print(f"Smoke shared_entity: {smoke_audit.get('shared_entity_count', 0)}")
    print(f"Smoke blockers: {smoke_audit['hard_blockers']}")

    if not smoke_audit.get("smoke_pass"):
        print("SMOKE FAILED — stopping before formal generation")
        return 1

    print("=== FORMAL GENERATION (clean from 0) ===")
    _clean_dir(FULL_GENERATION_DIR)
    formal_result = run_generation(
        target=FORMAL_TARGET_TOTAL,
        seed=FULL_GENERATION_SEED,
        output_dir=FULL_GENERATION_DIR,
        resume=False,
    )

    manifest = {
        "formal_target": FORMAL_TARGET_TOTAL,
        "full_generation_seed": FULL_GENERATION_SEED,
        "frozen_v1_sample_sha256": formal_result["frozen_sha256"],
        "schema_version": SCHEMA_VERSION,
        "composability_policy_version": COMPOSABILITY_POLICY_VERSION,
        "scheduler_version": SCHEDULER_VERSION,
        "entity_binding_version": ENTITY_BINDING_VERSION,
        "temporal_alignment_version": TEMPORAL_ALIGNMENT_VERSION,
        "parent_signature_version": PARENT_SIGNATURE_VERSION,
        "construction_validator_version": CONSTRUCTION_VALIDATOR_VERSION,
        "target_total": FORMAL_TARGET_TOTAL,
        "accepted_count": len(formal_result["accepted"]),
        "two_component_count": formal_result["two_accepted"],
        "three_component_count": formal_result["three_accepted"],
        "attempts": formal_result["attempts"],
        "acceptance_rate": len(formal_result["accepted"]) / max(1, formal_result["attempts"]),
        "stop_reason": formal_result["stop_reason"],
        "reject_taxonomy": formal_result["reject_taxonomy"],
        "pilot_samples_included": False,
        "smoke_samples_included": False,
        "formal_starting_count": 0,
    }
    (FULL_GENERATION_DIR / "full_generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    formal_audit = audit_formal_corpus(
        FULL_GENERATION_DIR / "multi_action_samples.jsonl",
        capacities=formal_result["capacities"],
        pilot_stats=PILOT_STATS,
    )
    (FULL_GENERATION_DIR / "full_generation_report.json").write_text(json.dumps(formal_audit, indent=2), encoding="utf-8")
    write_full_generation_md(formal_audit, FULL_GENERATION_DIR / "FULL_GENERATION_REPORT.md", manifest=manifest)

    capacity_report = {
        **formal_result["capacities"],
        "remaining_after_generation": formal_audit.get("remaining_capacity"),
        "note": "2400 is coverage-oriented benchmark target, not composition-space exhaustion",
    }
    (FULL_GENERATION_DIR / "composition_capacity_report.json").write_text(
        json.dumps(capacity_report, indent=2), encoding="utf-8"
    )

    print(f"Formal accepted: {formal_audit['sample_count']}")
    print(f"Stop reason: {formal_result['stop_reason']}")
    print(f"2-comp: {formal_audit['two_component_count']} 3-comp: {formal_audit['three_component_count']}")
    print(f"SAME_SOURCE: {formal_audit.get('same_source_count', 0)}")
    print(f"shared_entity: {formal_audit.get('shared_entity_count', 0)}")
    print(f"Verdict: {formal_audit['verdict']}")
    return 0 if formal_audit["verdict"] == "READY_FOR_MULTI_ACTION_FINAL_VALIDATION" else 1

if __name__ == "__main__":
    raise SystemExit(main())
