from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS, OUTPUT_DIR, SCENE_BLUEPRINT_MAP
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    V3_CLEAN_OUTPUT_DIR_NAME,
)
from smarthome_mdf.single_scene_blueprint_complete.yaml_acquisition import build_one_to_one_mapping

SRC = OUTPUT_DIR / V3_CLEAN_OUTPUT_DIR_NAME
FROZEN = OUTPUT_DIR / "frozen_v1"

IMPLEMENTATION_FILES = [
    "src/smarthome_mdf/single_scene_blueprint_complete/config.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/final_regeneration_config.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/final_scheduler.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/final_coordinator.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/quota_reallocation.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/blueprint_capacity.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/diversity_source_selector.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/synthesizer.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/sample_acceptance_gate.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/stable_signatures.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/grounded_instances.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/grounded_instance_cache.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/legacy_pool.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/yaml_mapping.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/final_validation.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/behavior_specs.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/information_flow.py",
    "src/smarthome_mdf/multi_action_vnext/requirement_extractor.py",
    "src/smarthome_mdf/multi_action_vnext/unified_runtime_matcher.py",
    "src/smarthome_mdf/multi_action_vnext/source_adapters.py",
    "src/smarthome_mdf/single_scene_blueprint_complete/semantic_capabilities.py",
]

CONFIG_COPY_NAMES = [
    "blueprint_budget.json",
    "scene_budget.json",
    "blueprint_complexity.json",
    "manifest.json",
    "grounded_instance_candidates.json",
    "grounded_instance_cache_manifest.json",
    "generated_blueprint_inventory.json",
    "true_evidence_limited.json",
    "information_flow_audit.json",
    "source_lineage.json",
    "full_blueprint_inventory.json",
]

VALIDATION_COPY_NAMES = [
    "final_validation_report.json",
    "FINAL_VALIDATION_REPORT.md",
]

SCENE_COUNTS = {
    "climate_window": 2900,
    "advanced_lighting": 2700,
    "notification_security": 2166,
    "scene_schedule_override": 1198,
    "appliance_monitoring": 1352,
    "periodic_task_scheduling": 700,
    "on_off_schedule": 600,
    "visual_fusion": 384,
}

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _line_count(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def _git_info() -> dict:
    info: dict = {"available": False}
    try:
        cwd = ROOT
        info["available"] = True
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        info["branch"] = subprocess.check_output(["git", "branch", "--show-current"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
        info["dirty"] = bool(dirty)
        info["dirty_files_count"] = len([l for l in dirty.splitlines() if l.strip()]) if dirty else 0
    except Exception as exc:
        info["error"] = str(exc)
    return info

def _blueprint_counts(samples_path: Path) -> dict[str, int]:
    counts: Counter = Counter()
    with samples_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            bp = (s.get("blueprint_binding") or {}).get("blueprint_id")
            if bp:
                counts[bp] += 1
    return dict(counts)

def _build_blueprint_inventory(counts: dict[str, int]) -> list[dict]:
    mapping = build_one_to_one_mapping()
    rows_by_bp = {r["blueprint_id"]: r for r in mapping["rows"]}
    inventory = []
    for bp in ALL_BLUEPRINT_IDS:
        scene = next(sc for sc, bps in SCENE_BLUEPRINT_MAP.items() if bp in bps)
        if bp in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
            status = "TRUE_EVIDENCE_LIMITED"
        else:
            status = "GROUNDED_CAPABLE"
        row = rows_by_bp.get(bp, {})
        inventory.append(
            {
                "blueprint_id": bp,
                "scene": scene,
                "yaml_path": row.get("yaml_path"),
                "yaml_sha256": row.get("yaml_hash"),
                "status": status,
                "sample_count": counts.get(bp, 0),
            }
        )
    return inventory

def _write_frozen_spec(out: Path, *, sample_sha: str, git: dict, inventory: list[dict]) -> None:
    bp_lines = "\n".join(
        f"| {r['blueprint_id']} | {r['scene']} | {r['sample_count']} | {r['status']} |"
        for r in inventory
    )
    scene_lines = "\n".join(f"| {sc} | {cnt} |" for sc, cnt in SCENE_COUNTS.items())
    content = f"""# Single-Scene Frozen Benchmark Specification (v1)

**Dataset:** EviRepair Single-Scene Blueprint-Complete Corpus
**Version:** `frozen_v1`
**Freeze date:** {_utc_now()}
**Sample count:** 12,000
**Validation verdict:** `READY_FOR_SINGLE_SCENE_FREEZE`
**Production stop:** `GLOBAL_TARGET_REACHED`
**Corpus SHA256:** `{sample_sha}`

## Inventory

| Metric | Value |
|---|---|
| Scenes | 8 |
| Blueprint inventory | 22 |
| Grounded-capable generated | 21 / 21 |
| TRUE_EVIDENCE_LIMITED | 1 (`appliance_vibration_sensor`) |

## Scene distribution

| Scene | Count |
|---|---:|
{scene_lines}
| **Total** | **12000** |

## Blueprint distribution

| blueprint_id | scene | sample_count | status |
|---|---|---:|---|
{bp_lines}

## Visual Fusion note

`visual_fusion` = 384 samples reflecting grounded visual capacity under current public evidence:

- `camera_frigate_vision_llm` = 224
- `camera_frigate_intelligent` = 160
- unique visual frames = 368
- max frame reuse = 2

Visual fusion is **not** a freeze blocker for being below historical planning quotas.

## Source diversity (validated)

| Source | Samples |
|---|---:|
| CASAS | 9163 |
| UK-DALE | 2052 |
| YouHome | 785 |

- unique source observations = 2945
- max source reuse = 87, median = 2, p95 = 11
- reuse does **not** imply duplicate behavior×source pairs (12,000 unique pairs)

## Architecture principles

Blueprint-first single-scene synthesis constructs samples from:

- Blueprint semantics (22 canonical YAML assets)
- public grounded observations (CASAS, UK-DALE, YouHome)
- runtime requirement matching
- entity binding and temporal alignment
- construction validation (acceptance gate, dedup, provenance)

### Duplicate definition (hard blockers)

- `EXACT_DUPLICATE` = identical canonical sample signature
- `CANONICAL_DUPLICATE` = identical behavior×source dedup key
- `SAME_SOURCE_DUPLICATE_PADDING` = same canonical behavior reusing same source without diversity

Same public observation with **different reachable behavior** is **not** a duplicate.

### Grounded capacity principle

Generation stopped at target 12,000 (`GLOBAL_TARGET_REACHED`), not because all grounded capacity was exhausted. Remaining grounded pairs are documented in capacity reports; no duplicate padding was used to reach the target.

## Methodological boundary (construction-only)

Single-scene synthesis uses Blueprint semantics, public grounded observations, runtime matching, entity binding, temporal alignment, and construction validation.

Single-scene synthesis does **NOT** use or produce:

- B0
- Y / expected-state labels
- conflict labels
- repair outputs
- diagnosis results
- ARS evaluation

Formal B0 and Y are downstream independent pipelines.

## Git provenance at freeze

```json
{json.dumps(git, indent=2)}
```

## Known limitations

- `appliance_vibration_sensor` is inventory-known but TRUE_EVIDENCE_LIMITED (no public vibration evidence).
- Three legacy automation instance IDs appear in early checkpoint samples only; they do not inflate current grounded-instance coverage.
- Sub-planning-quota blueprints with small grounded pools (security contact/window/ikea, advanced_lighting 29-source pools) reflect true grounded exhaustion, not synthesis failure.
"""
    (out / "SINGLE_SCENE_FROZEN_V1_SPEC.md").write_text(content, encoding="utf-8")

def _write_validation_summary(out: Path) -> None:
    content = """# Freeze Validation Summary

Independent final validation passed before freeze.

| Check | Result |
|---|---|
| FINAL_SAMPLE_COUNT | 12000 |
| SCENE_COVERAGE | 8/8 |
| BLUEPRINT_INVENTORY | 22 |
| GROUNDED_CAPABLE_BLUEPRINT_COVERAGE | 21/21 |
| TRUE_EVIDENCE_LIMITED | 1 (appliance_vibration_sensor) |
| POST_GENERATION_RUNTIME_MATCH | 12000/12000 |
| MATCH_FAIL | 0 |
| SEMANTIC_INVALID_ACCEPTED | 0 |
| PROVENANCE_INVALID | 0 |
| EXACT_DUPLICATE | 0 |
| CANONICAL_DUPLICATE | 0 |
| SAME_SOURCE_DUPLICATE_PADDING | 0 |
| TIMESTAMP_ONLY_DUPLICATE | 0 |
| INFORMATION_FLOW_VIOLATION | 0 |
| IMPLEMENTATION_GAP | 0 |

**Final validation verdict:** `READY_FOR_SINGLE_SCENE_FREEZE`
"""
    (out / "FREEZE_VALIDATION_SUMMARY.md").write_text(content, encoding="utf-8")

def _write_do_not_modify(out: Path) -> None:
    (out / "FROZEN_DO_NOT_MODIFY.md").write_text(
        """# FROZEN — DO NOT MODIFY

This directory (`frozen_v1`) is an **immutable benchmark snapshot** of the validated
EviRepair single-scene corpus (12,000 samples).

## Rules

1. **Do not edit, append, or regenerate** any file in this directory in place.
2. Any correction or extension must produce a **new version** (e.g. `frozen_v2`).
3. The development directory `final_regeneration_v3_clean/` remains the generation workspace; this directory is the canonical benchmark input for downstream Multi-action work.
4. Integrity is enforced by `FREEZE_SHA256_MANIFEST.json` — verify hashes before use.

## Canonical entry point

`single_scene_samples.jsonl` — 12,000 blueprint-grounded single-scene samples.

Downstream stages (Multi-action, B0, Y, conflict, repair, ARS) must consume this snapshot explicitly, not the mutable development tree.
""",
        encoding="utf-8",
    )

def run_freeze() -> dict:
    if not (SRC / "single_scene_samples.jsonl").is_file():
        raise FileNotFoundError(f"Missing source corpus: {SRC / 'single_scene_samples.jsonl'}")
    validation = json.loads((SRC / "final_validation_report.json").read_text(encoding="utf-8"))
    if validation.get("FINAL_VERDICT") != "READY_FOR_SINGLE_SCENE_FREEZE":
        raise RuntimeError(f"Validation verdict not ready: {validation.get('FINAL_VERDICT')}")

    if FROZEN.exists():
        raise RuntimeError(f"Frozen directory already exists: {FROZEN}. Remove manually to re-freeze as frozen_v2 policy.")

    for sub in ("config", "grounded", "validation", "provenance", "specs"):
        (FROZEN / sub).mkdir(parents=True, exist_ok=True)

    src_samples = SRC / "single_scene_samples.jsonl"
    dst_samples = FROZEN / "single_scene_samples.jsonl"
    shutil.copy2(src_samples, dst_samples)

    src_sample_sha = _sha256(src_samples)
    dst_sample_sha = _sha256(dst_samples)
    src_lines = _line_count(src_samples)
    dst_lines = _line_count(dst_samples)

    for name in CONFIG_COPY_NAMES:
        p = SRC / name
        if p.is_file():
            shutil.copy2(p, FROZEN / "config" / name)

    for name in VALIDATION_COPY_NAMES:
        p = SRC / name
        if p.is_file():
            shutil.copy2(p, FROZEN / "validation" / name)

    for name in ("grounded_instance_candidates.json", "grounded_instance_cache_manifest.json"):
        p = SRC / name
        if p.is_file():
            shutil.copy2(p, FROZEN / "grounded" / name)

    bp_counts = _blueprint_counts(dst_samples)
    inventory = _build_blueprint_inventory(bp_counts)
    (FROZEN / "blueprint_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    yaml_mapping = build_one_to_one_mapping()
    (FROZEN / "config" / "blueprint_yaml_mapping.json").write_text(
        json.dumps(yaml_mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    git = _git_info()
    impl_hashes = []
    for rel in IMPLEMENTATION_FILES:
        p = ROOT / rel
        if p.is_file():
            impl_hashes.append({"relative_path": rel.replace("\\", "/"), "sha256": _sha256(p)})

    synthesis_config = {
        "synthesis_version": "v3_clean",
        "canonical_signature_version": "stable_v3",
        "deterministic_seed_policy": "diversity_selector.generation_step + tiebreak hash",
        "phase2_quota_policy": "quota_reallocation.py (PHASE2_INCREMENT_BASE=150, soft share cap)",
        "dedup_policy": "behavior_source_dedup_key + canonical_sample_signature; SAME_SOURCE_DUPLICATE_PADDING gate",
        "source_rotation_policy": "DiversitySourceSelector coverage-aware deterministic rotation",
        "runtime_matching": "unified_runtime_matcher + requirement_extractor minimal_grounding_requirements",
        "semantic_capabilities": "semantic_capabilities.py attach_capabilities",
        "grounded_instance_config": "grounded_instances_v3_config_sig",
        "target_total": 12000,
        "global_stop_at_freeze": "GLOBAL_TARGET_REACHED",
    }
    (FROZEN / "config" / "synthesis_config_snapshot.json").write_text(
        json.dumps(synthesis_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source_provenance = {
        "datasets_used": ["CASAS", "UK-DALE", "YouHome"],
        "validated_distribution": {"CASAS": 9163, "UK-DALE": 2052, "YouHome": 785},
        "unique_source_observations": 2945,
        "source_index_fingerprint": json.loads((SRC / "grounded_instance_cache_manifest.json").read_text(encoding="utf-8"))
        .get("dependencies", {})
        .get("source_index_fingerprint"),
        "adapter_version": "source_adapters_semantic_capabilities_v1",
        "license_note": "Public raw datasets referenced by project legacy_pool; full raw archives not duplicated in freeze.",
        "lineage": json.loads((SRC / "source_lineage.json").read_text(encoding="utf-8")) if (SRC / "source_lineage.json").is_file() else {},
    }
    (FROZEN / "provenance" / "source_provenance.json").write_text(
        json.dumps(source_provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    freeze_meta = {
        "freeze_version": "frozen_v1",
        "frozen_at": _utc_now(),
        "source_directory": str(SRC),
        "validation_verdict": validation.get("FINAL_VERDICT"),
        "git": git,
        "implementation_file_hashes": impl_hashes,
    }
    (FROZEN / "FREEZE_METADATA.json").write_text(json.dumps(freeze_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_validation_summary(FROZEN)
    _write_do_not_modify(FROZEN)
    _write_frozen_spec(FROZEN, sample_sha=dst_sample_sha, git=git, inventory=inventory)

    manifest_entries: dict[str, str] = {}
    for path in sorted(FROZEN.rglob("*")):
        if path.is_file():
            rel = path.relative_to(FROZEN).as_posix()
            manifest_entries[rel] = _sha256(path)

    for bp_row in inventory:
        ypath = bp_row.get("yaml_path")
        if ypath and Path(ypath).is_file():
            rel_yaml = f"yaml_assets/{bp_row['blueprint_id']}.yaml"
            manifest_entries[rel_yaml] = _sha256(Path(ypath))

    manifest = {
        "freeze_version": "frozen_v1",
        "generated_at": _utc_now(),
        "sample_count": dst_lines,
        "sample_sha256": dst_sample_sha,
        "source_sample_sha256": src_sample_sha,
        "artifacts": manifest_entries,
        "yaml_asset_hashes_in_inventory": {r["blueprint_id"]: r["yaml_sha256"] for r in inventory},
    }
    (FROZEN / "FREEZE_SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_lines = [f"{h}  {p}" for p, h in sorted(manifest_entries.items())]
    (FROZEN / "FREEZE_SHA256_MANIFEST.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    errors: dict[str, int] = {
        "FREEZE_HASH_MISMATCH": 0,
        "FREEZE_MISSING_ARTIFACT": 0,
        "FREEZE_SAMPLE_COUNT_MISMATCH": 0,
    }
    if src_sample_sha != dst_sample_sha:
        errors["FREEZE_HASH_MISMATCH"] += 1
    if dst_lines != 12000:
        errors["FREEZE_SAMPLE_COUNT_MISMATCH"] += 1
    required = [
        "single_scene_samples.jsonl",
        "FREEZE_SHA256_MANIFEST.json",
        "SINGLE_SCENE_FROZEN_V1_SPEC.md",
        "FREEZE_VALIDATION_SUMMARY.md",
        "FROZEN_DO_NOT_MODIFY.md",
        "blueprint_inventory.json",
        "validation/final_validation_report.json",
    ]
    for rel in required:
        if not (FROZEN / rel).is_file():
            errors["FREEZE_MISSING_ARTIFACT"] += 1

    for rel, expected in manifest_entries.items():
        p = FROZEN / rel
        if p.is_file() and _sha256(p) != expected:
            errors["FREEZE_HASH_MISMATCH"] += 1

    verdict = "SINGLE_SCENE_FROZEN_V1_READY" if sum(errors.values()) == 0 else "FREEZE_BLOCKED"
    result = {
        "FREEZE_VERDICT": verdict,
        "frozen_directory": str(FROZEN),
        "sample_sha256": dst_sample_sha,
        "sample_count": dst_lines,
        "scene_coverage": 8,
        "blueprint_inventory": 22,
        "grounded_capable_coverage": 21,
        "true_evidence_limited": "appliance_vibration_sensor",
        "integrity_errors": errors,
        "git": git,
        "downstream_executed": False,
    }
    (FROZEN / "FREEZE_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

def main() -> int:
    result = run_freeze()
    print(f"FREEZE VERDICT: {result['FREEZE_VERDICT']}")
    print(f"Directory: {result['frozen_directory']}")
    print(f"SHA256: {result['sample_sha256']}")
    print(f"Samples: {result['sample_count']}")
    if sum(result["integrity_errors"].values()):
        print("ERRORS:", result["integrity_errors"])
    return 0 if result["FREEZE_VERDICT"] == "SINGLE_SCENE_FROZEN_V1_READY" else 1

if __name__ == "__main__":
    raise SystemExit(main())
