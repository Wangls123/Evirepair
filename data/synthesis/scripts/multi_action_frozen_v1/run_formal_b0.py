from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import audit_b0_information_flow
from smarthome_mdf.multi_action_frozen_v1.b0_blueprint_inventory_audit import (
    audit_blueprint_inventory,
    audit_construction_path_leakage,
    audit_runtime_projection_neutrality,
)
from smarthome_mdf.multi_action_frozen_v1.config import (
    EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    FORMAL_B0_V1_FROZEN_SHA256,
    FORMAL_B0_V2_DIR,
    FORMAL_B0_V2_FROZEN_DIR,
    FORMAL_B0_V2_PIPELINE_VERSION,
    FORMAL_TARGET_TOTAL,
    MULTI_ACTION_FROZEN_SAMPLES,
)
from smarthome_mdf.multi_action_frozen_v1.formal_b0_generator import pipeline_lineage, run_formal_b0_for_parent
from smarthome_mdf.multi_action_frozen_v1.formal_b0_semantic_audit import (
    audit_action_eligible_miss,
    audit_zero_action_blueprints_from_b0,
    run_counterfactual_tests,
)
from smarthome_mdf.multi_action_frozen_v1.formal_b0_validation import validate_formal_b0_outputs
from smarthome_mdf.multi_action_frozen_v1.frozen_loader import load_frozen_samples, load_multi_action_frozen, verify_multi_action_frozen_sha

IMPLEMENTATION_FILES = [
    "src/smarthome_mdf/synthesis_v3/frozen_blueprint_executor_registry.py",
    "src/smarthome_mdf/synthesis_v3/generic_blueprint_executor.py",
    "src/smarthome_mdf/synthesis_v3/blueprint_executor.py",
    "src/smarthome_mdf/synthesis_v3/phase2_executors.py",
    "src/smarthome_mdf/multi_action_frozen_v1/b0_adapter.py",
    "src/smarthome_mdf/multi_action_frozen_v1/frozen_instance_resolver.py",
    "src/smarthome_mdf/multi_action_frozen_v1/frozen_runtime_projector.py",
    "src/smarthome_mdf/multi_action_frozen_v1/formal_b0_generator.py",
    "src/smarthome_mdf/multi_action_frozen_v1/formal_b0_semantic_audit.py",
    "src/smarthome_mdf/multi_action_frozen_v1/formal_b0_validation.py",
    "scripts/multi_action_frozen_v1/run_formal_b0.py",
]

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _git_info() -> dict:
    info: dict = {"available": False}
    try:
        info["available"] = True
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        info["dirty"] = bool(dirty)
    except Exception as exc:
        info["error"] = str(exc)
    return info

def _implementation_hashes() -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in IMPLEMENTATION_FILES:
        p = ROOT / rel
        if p.is_file():
            out[rel] = _sha256(p)
    return out

def _load_frozen_index() -> dict[str, dict]:
    samples, _ = load_frozen_samples(verify=True)
    return {s["sample_id"]: s for s in samples}

def _compare_v1_v2(v1_path: Path, v2_records: list[dict]) -> dict:
    v1_by_id = {}
    with v1_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                v1_by_id[str(r["multi_action_id"])] = r

    v2_by_id = {str(r["multi_action_id"]): r for r in v2_records}

    comp_bp: dict[str, str] = {}
    for r in v2_records:
        for pc in r.get("per_component_b0") or []:
            cid = str(pc.get("component_id") or "")
            bid = str(pc.get("blueprint_id") or "")
            if cid and bid:
                comp_bp[cid] = bid
    parents_changed = 0
    components_changed = 0
    actions_added = 0
    actions_removed = 0
    actions_modified = 0
    by_bp: dict[str, dict[str, int]] = defaultdict(
        lambda: {"parents": 0, "components": 0, "actions_added": 0, "actions_removed": 0}
    )

    def _action_key(a: dict) -> tuple:
        target = a.get("target_entity")
        if isinstance(target, list):
            target = tuple(target)
        return (
            a.get("service"),
            target,
            json.dumps(a.get("parameters") or {}, sort_keys=True),
            a.get("component_origin"),
        )

    for mid, v2 in v2_by_id.items():
        v1 = v1_by_id.get(mid)
        if not v1:
            continue
        v1_actions = (v1.get("canonical_b0") or {}).get("semantic_actions") or []
        v2_actions = (v2.get("canonical_b0") or {}).get("semantic_actions") or []
        v1_set = Counter(_action_key(a) for a in v1_actions)
        v2_set = Counter(_action_key(a) for a in v2_actions)
        if v1_set != v2_set:
            parents_changed += 1
            added = sum((v2_set - v1_set).values())
            removed = sum((v1_set - v2_set).values())
            actions_added += added
            actions_removed += removed
            if added and removed:
                actions_modified += min(added, removed)

        v1_pc = {str(c.get("component_id")): c for c in v1.get("per_component_b0") or []}
        v2_pc = {str(c.get("component_id")): c for c in v2.get("per_component_b0") or []}
        for cid, v2c in v2_pc.items():
            v1c = v1_pc.get(cid, {})
            v1a = Counter(_action_key(a) for a in v1c.get("semantic_actions") or [])
            v2a = Counter(_action_key(a) for a in v2c.get("semantic_actions") or [])
            if v1a != v2a:
                components_changed += 1
                bid = str(v2c.get("blueprint_id") or v1c.get("blueprint_id") or comp_bp.get(cid) or "unknown")
                by_bp[bid]["components"] += 1
                by_bp[bid]["actions_added"] += sum((v2a - v1a).values())
                by_bp[bid]["actions_removed"] += sum((v1a - v2a).values())
                if v1a != v2a:
                    by_bp[bid]["parents"] += 1

    return {
        "parents_changed": parents_changed,
        "components_changed": components_changed,
        "actions_added": actions_added,
        "actions_removed": actions_removed,
        "actions_modified": actions_modified,
        "by_blueprint_id": dict(by_bp),
        "formal_b0_v1_sha256": _sha256(v1_path),
    }

def _write_freeze_artifacts(
    *,
    source_jsonl: Path,
    frozen_jsonl: Path,
    validation: dict,
    semantic: dict,
    manifest: dict,
    lineage: dict,
) -> dict:
    FORMAL_B0_V2_FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_jsonl, frozen_jsonl)
    source_sha = _sha256(source_jsonl)
    frozen_sha = _sha256(frozen_jsonl)
    if source_sha != frozen_sha:
        raise RuntimeError("Freeze copy SHA mismatch")

    freeze_meta = {
        "frozen_at": _utc_now(),
        "formal_b0_v2_frozen_sha256": frozen_sha,
        "formal_b0_v1_frozen_sha256": FORMAL_B0_V1_FROZEN_SHA256,
        "formal_b0_v1_status": "SEMANTICALLY_DEFECTIVE_HISTORICAL_B0",
        "source_b0_outputs_sha256": source_sha,
        "multi_action_frozen_corpus_sha256": lineage["multi_action_frozen_corpus_sha256"],
        "input_sample_count": FORMAL_TARGET_TOTAL,
        "validation_verdict": validation["verdict"],
        "semantic_verdict": semantic["verdict"],
        **lineage,
    }
    (FORMAL_B0_V2_FROZEN_DIR / "B0_FREEZE_METADATA.json").write_text(
        json.dumps(freeze_meta, indent=2), encoding="utf-8"
    )
    (FORMAL_B0_V2_FROZEN_DIR / "B0_FREEZE_SHA256_MANIFEST.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "b0_outputs.jsonl": frozen_sha,
                    "source_b0_outputs.jsonl": source_sha,
                    MULTI_ACTION_FROZEN_SAMPLES.name: lineage["multi_action_frozen_corpus_sha256"],
                    "formal_b0_v1_historical": FORMAL_B0_V1_FROZEN_SHA256,
                },
                "implementation_sha256": manifest.get("implementation_sha256"),
                "frozen_at": freeze_meta["frozen_at"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    spec_md = f"""# Formal B0 Frozen v2 Specification

## Identity
- **formal_b0_v2_frozen_sha256:** `{frozen_sha}`
- **multi_action_frozen_corpus_sha256:** `{lineage['multi_action_frozen_corpus_sha256']}`
- **formal_b0_v1 (historical, defective):** `{FORMAL_B0_V1_FROZEN_SHA256}`
- **input_sample_count:** {FORMAL_TARGET_TOTAL}
- **pipeline_version:** {lineage['pipeline_version']}

## Semantic repairs (v1 → v2)
- `appliance_power_google_sheets`: power-monitor executor (not periodic scheduler)
- `camera_frigate_*`: grounded `camera` input binding
- `lighting_motion_advanced_v22`: YAML default `light.turn_on` branch

## Canonical output
Each record contains `canonical_b0.semantic_actions[]` (ActionVNext structured actions).

## Immutability
Do not modify files under `formal_b0_v2/frozen_v1/` after freeze.
"""
    (FORMAL_B0_V2_FROZEN_DIR / "B0_FROZEN_V2_SPEC.md").write_text(spec_md, encoding="utf-8")
    (FORMAL_B0_V2_FROZEN_DIR / "B0_FREEZE_VALIDATION_SUMMARY.md").write_text(
        f"# B0 v2 Freeze Validation Summary\n\n"
        f"**Verdict:** `{validation['verdict']}`\n\n"
        f"**Semantic verdict:** `{semantic['verdict']}`\n\n"
        f"**formal_b0_v2_frozen_sha256:** `{frozen_sha}`\n\n"
        f"**B0_ACTION_ELIGIBLE_MISS:** {semantic.get('B0_ACTION_ELIGIBLE_MISS')}\n\n"
        f"**Blockers:** {validation.get('blockers') or 'none'}\n",
        encoding="utf-8",
    )
    (FORMAL_B0_V2_FROZEN_DIR / "FROZEN_DO_NOT_MODIFY.md").write_text(
        "# FROZEN — DO NOT MODIFY\n\n"
        f"Immutable formal B0 v2 artifact.\n\n"
        f"SHA256: `{frozen_sha}`\n\n"
        f"Historical v1 (`SEMANTICALLY_DEFECTIVE_HISTORICAL_B0`): `{FORMAL_B0_V1_FROZEN_SHA256}`\n",
        encoding="utf-8",
    )
    return {"formal_b0_v2_frozen_sha256": frozen_sha, "source_sha256": source_sha}

def main() -> int:
    FORMAL_B0_V2_DIR.mkdir(parents=True, exist_ok=True)
    val_dir = FORMAL_B0_V2_DIR / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    frozen_sha = verify_multi_action_frozen_sha()
    assert frozen_sha == EXPECTED_MULTI_ACTION_FROZEN_SHA256
    v1_frozen = ROOT / "data/multi_action_frozen_v1/formal_b0_v1/frozen_v1/b0_outputs.jsonl"
    assert v1_frozen.is_file(), "formal_b0_v1 frozen artifact required for delta audit"
    assert _sha256(v1_frozen) == FORMAL_B0_V1_FROZEN_SHA256

    corpus, _ = load_multi_action_frozen(verify=True)
    if len(corpus) != FORMAL_TARGET_TOTAL:
        print(json.dumps({"error": f"Expected {FORMAL_TARGET_TOTAL} samples, got {len(corpus)}"}))
        return 1

    frozen_index = _load_frozen_index()
    lineage = pipeline_lineage(pipeline_version=FORMAL_B0_V2_PIPELINE_VERSION)
    impl_hashes = _implementation_hashes()
    started = _utc_now()

    counterfactual = run_counterfactual_tests()
    cf_fail = sum(1 for v in counterfactual.values() if not v.get("pass"))
    pre_elig = audit_action_eligible_miss(corpus, frozen_index, focus_only=True)
    (val_dir / "targeted_pre_run_validation.json").write_text(
        json.dumps(
            {
                "counterfactual_tests": counterfactual,
                "COUNTERFACTUAL_ACTION_TEST_FAIL": cf_fail,
                "B0_ACTION_ELIGIBLE_MISS": pre_elig["B0_ACTION_ELIGIBLE_MISS"],
                "per_blueprint": pre_elig["per_blueprint"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if cf_fail > 0 or pre_elig["B0_ACTION_ELIGIBLE_MISS"] > 0:
        print(
            json.dumps(
                {
                    "verdict": "BLOCKED_BEFORE_FORMAL_B0_V2_FREEZE",
                    "reason": "targeted_pre_run_validation_failed",
                    "COUNTERFACTUAL_ACTION_TEST_FAIL": cf_fail,
                    "B0_ACTION_ELIGIBLE_MISS": pre_elig["B0_ACTION_ELIGIBLE_MISS"],
                },
                indent=2,
            )
        )
        return 1

    out_path = FORMAL_B0_V2_DIR / "b0_outputs.jsonl"
    status_counts: dict[str, int] = {}

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, ma in enumerate(corpus):
            record = run_formal_b0_for_parent(ma, frozen_index, pipeline_version=FORMAL_B0_V2_PIPELINE_VERSION)
            st = str(record.get("execution_status") or "UNKNOWN")
            status_counts[st] = status_counts.get(st, 0) + 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"Progress: {i + 1}/{FORMAL_TARGET_TOTAL}", file=sys.stderr)

    records = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    validation = validate_formal_b0_outputs(records, corpus, expected_count=FORMAL_TARGET_TOTAL)
    validation["verdict"] = (
        "FORMAL_B0_V2_READY" if validation["verdict"] == "FORMAL_B0_FROZEN_V1_READY" else validation["verdict"].replace("V1", "V2")
    )
    if validation["verdict"] == "BLOCKED_BEFORE_FORMAL_B0_FREEZE":
        validation["verdict"] = "BLOCKED_BEFORE_FORMAL_B0_V2_FREEZE"

    b0_sha = _sha256(out_path)
    semantic = audit_zero_action_blueprints_from_b0(corpus, records, frozen_index, formal_b0_sha256=b0_sha)
    delta = _compare_v1_v2(v1_frozen, records)

    (val_dir / "formal_b0_validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (val_dir / "zero_action_blueprint_audit.json").write_text(json.dumps(semantic, indent=2), encoding="utf-8")
    (val_dir / "b0_v1_v2_delta_audit.json").write_text(json.dumps(delta, indent=2), encoding="utf-8")

    inventory = audit_blueprint_inventory()
    flow = audit_b0_information_flow()
    constr = audit_construction_path_leakage()
    runtime = audit_runtime_projection_neutrality()

    gates = {
        **validation["gates"],
        "B0_ACTION_ELIGIBLE_MISS": semantic["B0_ACTION_ELIGIBLE_MISS"],
        "ZERO_ACTION_BLUEPRINT_UNEXPLAINED": semantic["ZERO_ACTION_BLUEPRINT_UNEXPLAINED"],
        "COUNTERFACTUAL_ACTION_TEST_FAIL": semantic["COUNTERFACTUAL_ACTION_TEST_FAIL"],
        "B0_INFORMATION_FLOW_VIOLATION": flow["B0_INFORMATION_FLOW_VIOLATION"],
        "B0_CONSTRUCTION_PATH_LEAKAGE": constr["B0_CONSTRUCTION_PATH_LEAKAGE"],
        "B0_RUNTIME_PROJECTION_DECISION_LEAKAGE": runtime["B0_RUNTIME_PROJECTION_DECISION_LEAKAGE"],
    }

    blockers = list(validation.get("blockers") or [])
    if semantic["B0_ACTION_ELIGIBLE_MISS"] > 0:
        blockers.append("B0_ACTION_ELIGIBLE_MISS")
    if semantic["ZERO_ACTION_BLUEPRINT_UNEXPLAINED"] > 0:
        blockers.append("ZERO_ACTION_BLUEPRINT_UNEXPLAINED")
    if semantic["COUNTERFACTUAL_ACTION_TEST_FAIL"] > 0:
        blockers.append("COUNTERFACTUAL_ACTION_TEST_FAIL")

    final_verdict = "FORMAL_B0_V2_FROZEN_READY" if not blockers else "BLOCKED_BEFORE_FORMAL_B0_V2_FREEZE"

    manifest = {
        "generated_at": started,
        "completed_at": _utc_now(),
        "multi_action_frozen_corpus_sha256": frozen_sha,
        "formal_b0_v1_frozen_sha256": FORMAL_B0_V1_FROZEN_SHA256,
        "formal_b0_v1_status": "SEMANTICALLY_DEFECTIVE_HISTORICAL_B0",
        "input_sample_count": FORMAL_TARGET_TOTAL,
        "output_record_count": len(records),
        **lineage,
        "implementation_sha256": impl_hashes,
        "git": _git_info(),
        "execution_status_counts": status_counts,
        "validation_verdict": validation["verdict"],
        "semantic_verdict": semantic["verdict"],
        "final_verdict": final_verdict,
        "validation_gates": gates,
        "delta_audit": delta,
        "blueprint_inventory": {
            "SCENE_FALLBACK_ONLY": inventory["SCENE_FALLBACK_ONLY"],
            "UNSUPPORTED_BLUEPRINT": inventory["UNSUPPORTED_BLUEPRINT"],
        },
        "output_path": str(out_path),
        "input_corpus_path": str(MULTI_ACTION_FROZEN_SAMPLES),
    }
    manifest["b0_outputs_sha256"] = b0_sha
    (FORMAL_B0_V2_DIR / "B0_GENERATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    freeze_info = None
    if final_verdict == "FORMAL_B0_V2_FROZEN_READY":
        freeze_info = _write_freeze_artifacts(
            source_jsonl=out_path,
            frozen_jsonl=FORMAL_B0_V2_FROZEN_DIR / "b0_outputs.jsonl",
            validation=validation,
            semantic=semantic,
            manifest=manifest,
            lineage=lineage,
        )
        manifest["formal_b0_v2_frozen_sha256"] = freeze_info["formal_b0_v2_frozen_sha256"]

    report = {
        "FINAL_VERDICT": final_verdict,
        "input_frozen_sha256": frozen_sha,
        "formal_b0_v1_sha256": FORMAL_B0_V1_FROZEN_SHA256,
        "formal_b0_v1_status": "SEMANTICALLY_DEFECTIVE_HISTORICAL_B0",
        "B0_ACTION_ELIGIBLE_MISS": semantic["B0_ACTION_ELIGIBLE_MISS"],
        "ZERO_ACTION_BLUEPRINT_UNEXPLAINED": semantic["ZERO_ACTION_BLUEPRINT_UNEXPLAINED"],
        "COUNTERFACTUAL_ACTION_TEST_FAIL": semantic["COUNTERFACTUAL_ACTION_TEST_FAIL"],
        "gates": gates,
        "execution_status_counts": status_counts,
        "parent_statistics": validation["parent_statistics"],
        "component_audit": validation["component_audit"],
        "per_blueprint": validation["per_blueprint"],
        "delta_audit": delta,
        "counterfactual_tests": semantic["counterfactual_tests"],
        "formal_b0_v2_output_path": str(out_path),
        "formal_b0_v2_frozen_path": str(FORMAL_B0_V2_FROZEN_DIR),
        "formal_b0_v2_sha256": b0_sha,
        "freeze": freeze_info,
        "confirmations": {
            "formal_b0_v1_not_modified": _sha256(v1_frozen) == FORMAL_B0_V1_FROZEN_SHA256,
            "y_not_run": True,
            "b0_vs_y_not_run": True,
            "conflict_not_run": True,
            "repair_not_run": True,
        },
    }
    (FORMAL_B0_V2_DIR / "b0_generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if final_verdict == "FORMAL_B0_V2_FROZEN_READY" else 1

if __name__ == "__main__":
    raise SystemExit(main())
