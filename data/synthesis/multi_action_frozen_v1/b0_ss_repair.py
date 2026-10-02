from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.b0_adapter import audit_b0_information_flow
from smarthome_mdf.multi_action_frozen_v1.b0_blueprint_inventory_audit import (
    audit_blueprint_inventory,
    audit_construction_path_leakage,
    audit_runtime_projection_neutrality,
)
from smarthome_mdf.multi_action_frozen_v1.config import (
    FORMAL_B0_V2_DIR,
    FORMAL_B0_V2_FROZEN_DIR,
    FORMAL_B0_V2_PIPELINE_VERSION,
    FORMAL_B0_SS_REPAIR_DIR,
    FORMAL_B0_SS_REPAIR_FROZEN_DIR,
    FORMAL_TARGET_TOTAL,
    MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES,
    SS_REPAIR_SAMPLES,
)
from smarthome_mdf.multi_action_frozen_v1.formal_b0_generator import pipeline_lineage, run_formal_b0_for_parent
from smarthome_mdf.multi_action_frozen_v1.formal_b0_semantic_audit import (
    audit_zero_action_blueprints_from_b0,
    run_counterfactual_tests,
)
from smarthome_mdf.multi_action_frozen_v1.formal_b0_validation import validate_formal_b0_outputs
from smarthome_mdf.multi_action_frozen_v1.frozen_instance_resolver import set_frozen_ss_root
from smarthome_mdf.multi_action_frozen_v1.single_scene_b0 import run_single_scene_b0
from smarthome_mdf.single_scene_blueprint_complete.blueprint_instance_defaults import REPAIR_BLUEPRINT_IDS
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _affected_ma_ids(ma_rows: list[dict], repair_bp_ids: set[str]) -> set[str]:
    affected: set[str] = set()
    for parent in ma_rows:
        for comp in parent.get("components") or []:
            if str(comp.get("blueprint_id") or "") in repair_bp_ids:
                affected.add(str(parent.get("multi_action_id")))
                break
    return affected

def run_b0_ss_repair(
    *,
    ma_jsonl: Path | None = None,
    ss_jsonl: Path | None = None,
    prior_ma_b0_jsonl: Path | None = None,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    ma_jsonl = ma_jsonl or MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES
    ss_jsonl = ss_jsonl or SS_REPAIR_SAMPLES
    prior_ma_b0_jsonl = prior_ma_b0_jsonl or (FORMAL_B0_V2_FROZEN_DIR / "b0_outputs.jsonl")
    out_dir = out_dir or FORMAL_B0_SS_REPAIR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    if not ma_jsonl.is_file():
        raise FileNotFoundError(f"Missing repaired MA corpus: {ma_jsonl}")
    if not ss_jsonl.is_file():
        raise FileNotFoundError(f"Missing repaired SS corpus: {ss_jsonl}")
    if not prior_ma_b0_jsonl.is_file():
        raise FileNotFoundError(f"Missing prior MA B0 frozen: {prior_ma_b0_jsonl}")

    ss_sha = _sha256(ss_jsonl)
    ma_sha = _sha256(ma_jsonl)
    set_frozen_ss_root(ss_jsonl.parent)

    ss_rows = read_jsonl(ss_jsonl)
    ma_rows = read_jsonl(ma_jsonl)
    ss_index = {str(s["sample_id"]): s for s in ss_rows}
    prior_b0 = {str(r["multi_action_id"]): r for r in read_jsonl(prior_ma_b0_jsonl)}

    if len(ma_rows) != FORMAL_TARGET_TOTAL:
        raise RuntimeError(f"Expected {FORMAL_TARGET_TOTAL} MA parents, got {len(ma_rows)}")
    if len(ss_rows) != 12000:
        raise RuntimeError(f"Expected 12000 SS samples, got {len(ss_rows)}")

    affected_ma = _affected_ma_ids(ma_rows, set(REPAIR_BLUEPRINT_IDS))
    if verbose:
        print(f"SS repair SHA: {ss_sha}", flush=True)
        print(f"MA repair SHA: {ma_sha}", flush=True)
        print(f"Regenerating MA B0 for {len(affected_ma)} / {len(ma_rows)} parents", flush=True)

    ma_b0_rows: list[dict] = []
    ma_status = Counter()
    for i, parent in enumerate(ma_rows):
        mid = str(parent.get("multi_action_id"))
        if mid in affected_ma:
            record = run_formal_b0_for_parent(parent, ss_index, pipeline_version=FORMAL_B0_V2_PIPELINE_VERSION)
            record["upstream_ss_repair_sha256"] = ss_sha
            record["upstream_ma_repair_sha256"] = ma_sha
            record["ss_repair_b0_sync"] = "regenerated"
        else:
            record = dict(prior_b0[mid])
            record["ss_repair_b0_sync"] = "retained_prior_b0_v2"
            record["upstream_ss_repair_sha256"] = ss_sha
            record["upstream_ma_repair_sha256"] = ma_sha
        ma_status[str(record.get("execution_status") or "UNKNOWN")] += 1
        ma_b0_rows.append(record)
        if verbose and (i + 1) % 400 == 0:
            print(f"  MA B0 progress: {i + 1}/{len(ma_rows)}", flush=True)

    ma_b0_path = out_dir / "multi_action_b0_outputs.jsonl"
    _write_jsonl(ma_b0_path, ma_b0_rows)
    ma_b0_sha = _sha256(ma_b0_path)

    if verbose:
        print(f"Generating single-scene B0 for {len(ss_rows)} samples...", flush=True)

    ss_b0_rows: list[dict] = []
    ss_status = Counter()
    prior_ss_b0_path = out_dir / "single_scene_b0_outputs.jsonl"
    prior_ss_b0 = {}
    if prior_ss_b0_path.is_file():
        prior_ss_b0 = {str(r["sample_id"]): r for r in read_jsonl(prior_ss_b0_path)}

    repair_ss_ids = {sid for sid, s in ss_index.items() if str((s.get("blueprint_binding") or {}).get("blueprint_id") or "") in REPAIR_BLUEPRINT_IDS}

    for i, sample in enumerate(ss_rows):
        sid = str(sample.get("sample_id"))
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if sid not in repair_ss_ids and sid in prior_ss_b0:
            record = dict(prior_ss_b0[sid])
            record["ss_repair_b0_sync"] = "retained_prior_ss_b0"
            record["upstream_ss_sha256"] = ss_sha
        else:
            record = run_single_scene_b0(sample, upstream_ss_sha256=ss_sha)
            record["ss_repair_b0_sync"] = "regenerated"
        ss_status[str(record.get("execution_status") or "UNKNOWN")] += 1
        ss_b0_rows.append(record)
        if verbose and (i + 1) % 1000 == 0:
            print(f"  SS B0 progress: {i + 1}/{len(ss_rows)}", flush=True)

    ss_b0_path = out_dir / "single_scene_b0_outputs.jsonl"
    _write_jsonl(ss_b0_path, ss_b0_rows)
    ss_b0_sha = _sha256(ss_b0_path)

    validation = validate_formal_b0_outputs(ma_b0_rows, ma_rows, expected_count=FORMAL_TARGET_TOTAL)
    validation["verdict"] = validation["verdict"].replace("V1", "V2").replace("FORMAL_B0_FROZEN_V1_READY", "FORMAL_B0_V2_SS_REPAIR_READY")
    semantic = audit_zero_action_blueprints_from_b0(ma_rows, ma_b0_rows, ss_index, formal_b0_sha256=ma_b0_sha)
    counterfactual = run_counterfactual_tests()

    flow = audit_b0_information_flow()
    inventory = audit_blueprint_inventory()
    blockers = list(validation.get("blockers") or [])
    if semantic["B0_ACTION_ELIGIBLE_MISS"] > 0:
        blockers.append("B0_ACTION_ELIGIBLE_MISS")
    if semantic["ZERO_ACTION_BLUEPRINT_UNEXPLAINED"] > 0:
        blockers.append("ZERO_ACTION_BLUEPRINT_UNEXPLAINED")
    if semantic["COUNTERFACTUAL_ACTION_TEST_FAIL"] > 0:
        blockers.append("COUNTERFACTUAL_ACTION_TEST_FAIL")

    verdict = "FORMAL_B0_V2_SS_REPAIR_READY" if not blockers else "BLOCKED_BEFORE_FORMAL_B0_SS_REPAIR_FREEZE"

    report = {
        "verdict": verdict,
        "generated_at": _utc_now(),
        "upstream_ss_repair_sha256": ss_sha,
        "upstream_ma_repair_sha256": ma_sha,
        "multi_action_b0": {
            "path": str(ma_b0_path),
            "sha256": ma_b0_sha,
            "total": len(ma_b0_rows),
            "regenerated_parents": len(affected_ma),
            "retained_parents": len(ma_rows) - len(affected_ma),
            "execution_status_counts": dict(ma_status),
        },
        "single_scene_b0": {
            "path": str(ss_b0_path),
            "sha256": ss_b0_sha,
            "total": len(ss_b0_rows),
            "execution_status_counts": dict(ss_status),
        },
        "validation": validation,
        "semantic_audit": {
            "B0_ACTION_ELIGIBLE_MISS": semantic["B0_ACTION_ELIGIBLE_MISS"],
            "ZERO_ACTION_BLUEPRINT_UNEXPLAINED": semantic["ZERO_ACTION_BLUEPRINT_UNEXPLAINED"],
            "COUNTERFACTUAL_ACTION_TEST_FAIL": semantic["COUNTERFACTUAL_ACTION_TEST_FAIL"],
            "per_blueprint": semantic.get("per_blueprint"),
        },
        "counterfactual_tests": counterfactual,
        "blockers": blockers,
        "lineage": pipeline_lineage(pipeline_version=FORMAL_B0_V2_PIPELINE_VERSION),
    }
    (val_dir / "b0_ss_repair_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    frozen_dir = FORMAL_B0_SS_REPAIR_FROZEN_DIR
    frozen_dir.mkdir(parents=True, exist_ok=True)
    frozen_ma = frozen_dir / "multi_action_b0_outputs.jsonl"
    frozen_ss = frozen_dir / "single_scene_b0_outputs.jsonl"
    shutil.copy2(ma_b0_path, frozen_ma)
    shutil.copy2(ss_b0_path, frozen_ss)
    manifest = {
        "verdict": verdict,
        "frozen_at": _utc_now(),
        "upstream_ss_repair_sha256": ss_sha,
        "upstream_ma_repair_sha256": ma_sha,
        "multi_action_b0_outputs.jsonl": _sha256(frozen_ma),
        "single_scene_b0_outputs.jsonl": _sha256(frozen_ss),
        "prior_formal_b0_v2_dir": str(FORMAL_B0_V2_DIR),
    }
    (frozen_dir / "B0_SS_REPAIR_SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report["frozen_manifest"] = manifest

    if verbose:
        print(json.dumps({"verdict": verdict, "ma_b0_sha256": ma_b0_sha, "ss_b0_sha256": ss_b0_sha}, indent=2), flush=True)
    return report
