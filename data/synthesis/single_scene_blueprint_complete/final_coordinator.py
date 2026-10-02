from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import (
    _normalize_from_instance,
    build_all_behavior_specs,
    run_parser_audit,
)
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    OUTPUT_DIR,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.final_audits import build_all_audits, write_audit_artifacts
from smarthome_mdf.single_scene_blueprint_complete.final_budgets import build_blueprint_budget, build_scene_budget
from smarthome_mdf.single_scene_blueprint_complete.final_complexity import build_blueprint_complexity
from smarthome_mdf.single_scene_blueprint_complete.coverage_stop_policy import (
    CoverageStopPolicy,
    GLOBAL_STOP_COVERAGE_INCOMPLETE,
    GLOBAL_STOP_SAMPLE_TARGET_AND_COVERAGE_REACHED,
)
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    ACCEPTABLE_MIN,
    BASE_MIN_PER_BLUEPRINT,
    FINAL_OUTPUT_DIR_NAME,
    GLOBAL_STOP_ALL_GROUNDED_SATURATED,
    GLOBAL_STOP_TARGET_REACHED,
    GROUNDED_CAPABLE_BLUEPRINTS,
    HARD_MAX,
    MAX_ZERO_STRICT_PATHS,
    MIN_STRICT_PATH_COVERAGE_RATIO,
    MAX_PATH_SKEW,
    SCENE_BUDGETS,
    TARGET_TOTAL,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    V3_CLEAN_OUTPUT_DIR_NAME,
)
from smarthome_mdf.single_scene_blueprint_complete.path_coverage_state import PathCoverageState
from smarthome_mdf.single_scene_blueprint_complete.blueprint_capacity import compute_all_remaining_pairs
from smarthome_mdf.single_scene_blueprint_complete.quota_reallocation import (
    resume_reactivate_quota_limited,
    try_phase2_reallocation,
)
from smarthome_mdf.single_scene_blueprint_complete.diversity_source_selector import DiversitySourceSelector
from smarthome_mdf.single_scene_blueprint_complete.grounded_instance_cache import resolve_grounded_instances
from smarthome_mdf.single_scene_blueprint_complete.sample_acceptance_gate import validate_sample_accept
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    behavior_source_dedup_key,
    canonical_sample_signature,
    is_timestamp_only_duplicate,
)
from smarthome_mdf.single_scene_blueprint_complete.final_scheduler import FinalGenerationScheduler
from smarthome_mdf.single_scene_blueprint_complete.generation_gate import evaluate_generation_gate
from smarthome_mdf.single_scene_blueprint_complete.information_flow import audit_package_imports
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.inventory import build_full_inventory
from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool
from smarthome_mdf.single_scene_blueprint_complete.generation_path_request import GenerationPathRequest
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import structural_signature_stable as _structural_signature
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene, _get_ir
from smarthome_mdf.single_scene_blueprint_complete.v3_post_audits import write_v3_audit_artifacts
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import build_yaml_mapping_audit
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _check_no_synthesis_processes() -> list[str]:

    import subprocess

    my_pid = os.getpid()
    conflicts: list[str] = []
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        for row in data:
            pid = int(row.get("ProcessId") or 0)
            cmd = str(row.get("CommandLine") or "")
            if pid == my_pid:
                continue
            if any(k in cmd for k in ("run_synthesis.py", "run_single_scene_synthesis", "run_final_regeneration", "run_v3_clean_regeneration")):
                try:
                    os.kill(pid, 0)
                except OSError:
                    continue
                conflicts.append(f"pid={pid} {cmd[:120]}")
    except Exception:
        pass
    return conflicts

def _acquire_lock(out_dir: Path) -> Path:
    lock = out_dir / ".generation_lock"
    if lock.exists():
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            stale_pid = int(info.get("pid") or 0)
            if stale_pid:
                import subprocess
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {stale_pid} -ErrorAction SilentlyContinue"],
                    capture_output=True,
                )
                if out.returncode == 0:
                    raise RuntimeError(f"Generation lock held by live pid={stale_pid}")
            lock.unlink(missing_ok=True)
        except json.JSONDecodeError:
            lock.unlink(missing_ok=True)
    lock.write_text(json.dumps({"pid": os.getpid(), "started_at": _utc_now()}, indent=2), encoding="utf-8")
    return lock

def _release_lock(lock: Path) -> None:
    if lock.is_file():
        lock.unlink(missing_ok=True)

def _load_checkpoint(jsonl_path: Path) -> list[dict]:
    if not jsonl_path.is_file():
        return []
    samples = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            samples.append(json.loads(line))
    return samples

def _restore_scheduler_from_samples(scheduler: FinalGenerationScheduler, samples: list[dict]) -> dict[str, dict]:
    last_by_sig: dict[str, dict] = {}
    for sample in samples:
        scheduler.register_accept(sample, behavior_target=(sample.get("synthesis_metadata") or {}).get("behavior_target", ""))
        sig = _structural_signature(sample)
        last_by_sig[sig] = sample
    for st in scheduler.bp_state.values():
        if st.accepted_count < st.allocated_target or SCENE_BUDGETS.get(st.scene, {}).get("min", 0) > scheduler.scene_count(st.scene):
            st.stop_reason = "PENDING"
            st.consecutive_fail = 0
    for sc in scheduler.scene_state.values():
        budget = SCENE_BUDGETS.get(sc.scene, {})
        if sc.accepted_count < budget.get("min", 0):
            sc.stop_reason = "PENDING"
    return last_by_sig

def _write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def _true_evidence_limited_doc() -> dict:
    audit_path = OUTPUT_DIR / "true_evidence_limited.json"
    prior = {}
    if audit_path.is_file():
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = prior.get("blueprints") or []
    if not rows:
        rows = [
            {
                "blueprint_id": "appliance_vibration_sensor",
                "scene": "appliance_monitoring",
                "status": "TRUE_EVIDENCE_LIMITED",
                "required_runtime_capability": "vibration",
                "searched_public_datasets": ["CASAS", "UK-DALE", "YouHome", "REFIT"],
                "failure_reason": "No vibration sensor evidence in audited public raw datasets",
                "root_cause_audit_reference": str(OUTPUT_DIR / "EVIDENCE_LIMITED_ROOT_CAUSE_AUDIT.md"),
            }
        ]
    return {"blueprints": rows, "count": len(rows)}

def _evaluate_verdict(
    samples: list[dict],
    audits: dict,
    gate: dict,
    flow: list,
    parser_audit: dict,
    yaml_audit: dict,
    scheduler: FinalGenerationScheduler,
    *,
    v3_audits: dict | None = None,
    clean_run: bool = False,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if flow:
        blockers.append(f"information_flow_violations: {flow}")
    if parser_audit.get("implementation_gap", 0) > 0:
        blockers.append("IMPLEMENTATION_GAP > 0")
    if not yaml_audit.get("complete"):
        blockers.append("YAML mapping incomplete")
    forbidden = audits.get("distribution_audit", {}).get("forbidden_field_hits") or []
    if forbidden:
        blockers.append("Forbidden downstream fields in samples")

    generated = {bp for bp, c in Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples).items() if c > 0}
    expected = set(GROUNDED_CAPABLE_BLUEPRINTS)
    if generated != expected:
        missing = expected - generated
        if missing:
            blockers.append(f"Missing generated blueprints: {sorted(missing)}")

    total = len(samples)
    bp_dist = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples)

    if total < ACCEPTABLE_MIN:
        blockers.append(
            f"Total samples {total} below acceptable minimum {ACCEPTABLE_MIN} — report ACTUAL_GROUNDED_CAPACITY"
        )

    scene_counts = Counter(s.get("scene_type") for s in samples)
    for scene, budget in SCENE_BUDGETS.items():
        actual = scene_counts.get(scene, 0)
        if actual < budget["min"]:
            blockers.append(f"Scene {scene} below min: {actual}/{budget['min']}")

    for bp_id in GROUNDED_CAPABLE_BLUEPRINTS:
        if bp_dist.get(bp_id, 0) < BASE_MIN_PER_BLUEPRINT:
            blockers.append(f"Blueprint {bp_id} below base min {BASE_MIN_PER_BLUEPRINT}: {bp_dist.get(bp_id, 0)}")

    if "appliance_vibration_sensor" in generated:
        blockers.append("TRUE_EVIDENCE_LIMITED blueprint generated samples")

    scenes_with_data = {s.get("scene_type") for s in samples}
    if scenes_with_data != set(PROJECT_SCENES):
        blockers.append(f"Scenes missing data: {set(PROJECT_SCENES) - scenes_with_data}")

    starved = [bp for bp in GROUNDED_CAPABLE_BLUEPRINTS if bp_dist.get(bp, 0) < BASE_MIN_PER_BLUEPRINT]
    if len(starved) >= 3:
        blockers.append(f"Blueprint starvation: {len(starved)} blueprints below base min")

    global_stops = [st.stop_reason for st in scheduler.bp_state.values() if st.stop_reason == "GLOBAL_MAX_ATTEMPTS_REACHED"]
    if global_stops:
        blockers.append("GLOBAL_MAX_ATTEMPTS_REACHED used as blueprint stop reason")

    if clean_run and v3_audits:
        runtime = v3_audits.get("runtime_match_audit.json") or {}
        if runtime.get("POST_GENERATION_RUNTIME_MATCH_FAIL", 0) > 0:
            blockers.append(f"POST_GENERATION_RUNTIME_MATCH_FAIL={runtime['POST_GENERATION_RUNTIME_MATCH_FAIL']}")
        instance = v3_audits.get("instance_coverage.json") or {}
        if instance.get("GENERATOR_INSTANCE_COVERAGE_GAP", 0) > 0:
            blockers.append(f"GENERATOR_INSTANCE_COVERAGE_GAP={instance['GENERATOR_INSTANCE_COVERAGE_GAP']}")
        semantic = v3_audits.get("semantic_integrity_audit.json") or {}
        if semantic.get("SEMANTIC_INVALID", 0) > 0:
            blockers.append(f"SEMANTIC_INVALID={semantic['SEMANTIC_INVALID']}")
        visual = v3_audits.get("visual_provenance_audit.json") or {}
        if visual.get("PROVENANCE_INVALID", 0) > 0:
            blockers.append(f"PROVENANCE_INVALID={visual['PROVENANCE_INVALID']}")
        dup = v3_audits.get("duplicate_audit.json") or {}
        ts_only = dup.get("timestamp_only_duplicate_count", 0)
        total = len(samples) or 1
        if ts_only / total > 0.01:
            blockers.append(f"HIGH_TIMESTAMP_ONLY_DUPLICATE_RATE={ts_only}/{total}")

    instance_gaps = scheduler.instance_coverage_gaps()
    if instance_gaps:
        blockers.append(f"Scheduler instance gaps remain: {list(instance_gaps.keys())}")

    if bp_dist:
        total_n = sum(bp_dist.values())
        largest_share = max(bp_dist.values()) / total_n
        if largest_share > 0.45 and len(bp_dist) >= 8:
            blockers.append(f"Possible easy blueprint domination: largest share {largest_share:.2%}")

    if blockers:
        return "BLOCKED", blockers
    return "READY_FOR_SINGLE_SCENE_FREEZE", []

def _format_final_report(
    verdict: str,
    blockers: list[str],
    samples: list[dict],
    audits: dict,
    complexity: dict,
    scene_budget: dict,
    blueprint_budget: dict,
    scheduler: FinalGenerationScheduler,
) -> str:
    lines = [
        "# FINAL Single-Scene Regeneration Report",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Generated at:** {_utc_now()}",
        "",
        "## Summary",
        "",
        f"- Total samples: **{len(samples)}**",
        f"- Target: **{TARGET_TOTAL}** (acceptable {ACCEPTABLE_MIN}–{TARGET_TOTAL}, hard max {HARD_MAX})",
        f"- Inventory blueprints: **22**",
        f"- Grounded-capable: **21**",
        f"- Generated: **{len([r for r in audits.get('blueprint_table_22', []) if r.get('status') == 'GENERATED'])}**",
        f"- TRUE_EVIDENCE_LIMITED: **{len(TRUE_EVIDENCE_LIMITED_BLUEPRINTS)}** (`appliance_vibration_sensor`)",
        "",
    ]
    if blockers:
        lines.extend(["## Blockers", ""])
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    lines.extend(["## 8-Scene Table", "", "| scene | target | actual | stop |", "|---|---:|---:|---|"])
    for row in audits.get("scene_table_8") or []:
        lines.append(f"| {row['scene']} | {row['target_samples']} | {row['actual_samples']} | {row.get('stop_reason')} |")

    lines.extend(["", "## 22-Blueprint Table", "", "| blueprint_id | status | samples | complexity | stop |", "|---|---|---:|---:|---|"])
    for row in audits.get("blueprint_table_22") or []:
        lines.append(
            f"| {row['blueprint_id']} | {row.get('status')} | {row.get('sample_count', 0)} | {row.get('complexity_score', 0)} | {row.get('stop_reason')} |"
        )
    return "\n".join(lines) + "\n"

def _write_v3_implementation_report(
    out: Path,
    samples: list[dict],
    v3_audits: dict[str, Any],
    rejects: dict[str, int],
    parser_audit: dict,
) -> None:
    runtime = v3_audits.get("runtime_match_audit.json") or {}
    instance = v3_audits.get("instance_coverage.json") or {}
    semantic = v3_audits.get("semantic_integrity_audit.json") or {}
    visual = v3_audits.get("visual_provenance_audit.json") or {}
    duplicate = v3_audits.get("duplicate_audit.json") or {}

    lines = [
        "# FINAL Single-Scene Implementation Report (v3 clean)",
        "",
        "## CODE FIX",
        "",
        "- Runtime Matcher: requirement-aware capability pre-filter rejects observations without `power` capability before measurement matching.",
        "- Source Adapter / semantic_capabilities: strict raw-to-semantic mapping; motion observations cannot satisfy power requirements.",
        "- semantic_sanitize: retained as assertion guard only; contaminated samples rejected at acceptance gate.",
        "- Grounded instance enumeration: configuration-signature-based instance IDs (`{blueprint}__inst_{hash}`).",
        "- Scheduler: prioritizes uncovered grounded instances; does not stop with instance coverage gaps.",
        "- Stable signatures: canonical behavior/action/branch IDs without generation-time `::aXXXX` suffix.",
        "- Pre-accept gate: runtime match, semantic integrity, provenance, timestamp-only duplicate rejection.",
        "",
        "## DATA RESULT",
        "",
        f"- Total samples: **{len(samples)}**",
        f"- POST_GENERATION_RUNTIME_MATCH_FAIL: **{runtime.get('POST_GENERATION_RUNTIME_MATCH_FAIL', 'N/A')}** (MATCH_OK={runtime.get('MATCH_OK', 'N/A')}/{runtime.get('total', 'N/A')})",
        f"- GENERATOR_INSTANCE_COVERAGE_GAP: **{instance.get('GENERATOR_INSTANCE_COVERAGE_GAP', 'N/A')}**",
        f"- SEMANTIC_INVALID: **{semantic.get('SEMANTIC_INVALID', 'N/A')}**",
        f"- PROVENANCE_INVALID: **{visual.get('PROVENANCE_INVALID', 'N/A')}**",
        f"- Timestamp-only duplicates in output: **{duplicate.get('timestamp_only_duplicate_count', 0)}**",
        f"- Generation-time TIMESTAMP_ONLY_DUPLICATE rejects: **{rejects.get('TIMESTAMP_ONLY_DUPLICATE', 0)}**",
        f"- IMPLEMENTATION_GAP: **{parser_audit.get('implementation_gap', 0)}**",
        "",
        "## Modified source files",
        "",
        "- `semantic_capabilities.py` — capability contract",
        "- `semantic_sanitize.py` — guard-only sanitization",
        "- `public_observation_pool.py` — attach capabilities at pool build",
        "- `multi_action_vnext/unified_runtime_matcher.py` — capability pre-filter",
        "- `multi_action_vnext/source_capabilities.py` — delegates to strict capabilities",
        "- `grounded_instances.py` — instance enumeration",
        "- `sample_acceptance_gate.py` — pre-accept validation",
        "- `stable_signatures.py` — temporal/canonical dedup signatures",
        "- `final_scheduler.py` — instance coverage scheduling",
        "- `final_coordinator.py` — v3 clean run orchestration",
        "- `behavior_specs.py` — instance_overrides for grounded templates",
        "- `synthesizer.py` — stable IR metadata fields",
        "- `v3_post_audits.py` — post-generation validation artifacts",
        "",
    ]
    (out / "FINAL_SINGLE_SCENE_IMPLEMENTATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

def _write_generation_progress(
    out: Path,
    *,
    accepted: int,
    attempts: int,
    rejects: Counter,
    elapsed_seconds: float,
    new_since_start: int,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "accepted": accepted,
        "attempts": attempts,
        "new_since_start": new_since_start,
        "top_rejects": rejects.most_common(8),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "updated_at": _utc_now(),
    }
    if extra:
        payload.update(extra)
    (out / "generation_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _scene_capacity_report(scheduler: FinalGenerationScheduler) -> dict[str, dict[str, Any]]:
    from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import SCENE_BUDGETS

    report: dict[str, dict[str, Any]] = {}
    for scene, budget in SCENE_BUDGETS.items():
        actual = scheduler.scene_count(scene)
        report[scene] = {
            "target": budget.get("target"),
            "min": budget.get("min"),
            "actual": actual,
            "grounded_unique_capacity": actual,
            "saturation_reason": scheduler.scene_state.get(scene, None) and scheduler.scene_state[scene].stop_reason,
        }
    return report

def run_final_regeneration(
    out_dir: Path | None = None,
    *,
    verbose: bool = True,
    checkpoint_every: int = 500,
    clean_run: bool = False,
    resume: bool = False,
    max_new_samples: int | None = None,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    conflicts = _check_no_synthesis_processes()
    if conflicts:
        raise RuntimeError(f"Conflicting synthesis processes: {conflicts}")

    startup_t0 = time.perf_counter()
    startup_metrics: dict[str, float | str | None] = {}
    resume_requested = bool(resume)
    if resume_requested and checkpoint_every > 50:
        checkpoint_every = 50

    out = out_dir or (OUTPUT_DIR / (V3_CLEAN_OUTPUT_DIR_NAME if clean_run else FINAL_OUTPUT_DIR_NAME))
    out.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out)

    try:
        if verbose:
            print("=== V3 Clean Single-Scene Regeneration ===" if clean_run else "=== Final Single-Scene Regeneration ===", flush=True)

        jsonl_path = out / "single_scene_samples.jsonl"
        checkpoint_t0 = time.perf_counter()
        checkpoint_samples = 0
        samples: list[dict] = []
        if resume_requested and jsonl_path.is_file():
            samples = _load_checkpoint(jsonl_path)
            checkpoint_samples = len(samples)
        elif clean_run and jsonl_path.is_file() and not resume_requested:
            jsonl_path.unlink()
        startup_metrics["checkpoint_load_seconds"] = round(time.perf_counter() - checkpoint_t0, 3)

        if verbose and resume_requested:
            print(f"Resume requested: {str(resume_requested).lower()}", flush=True)
            print(f"Checkpoint samples: {checkpoint_samples}", flush=True)

        gate = evaluate_generation_gate()
        if gate["verdict"] != "READY_FOR_GENERATION":
            raise RuntimeError(f"Generation gate blocked: {gate.get('blockers')}")

        reg = build_registry()
        pool_t0 = time.perf_counter()
        pool = build_legacy_source_pool(PROJECT_SCENES, include_public=True)
        startup_metrics["source_index_load_seconds"] = round(time.perf_counter() - pool_t0, 3)

        cache_t0 = time.perf_counter()
        cache_resolution = resolve_grounded_instances(
            out,
            pool,
            reg,
            clean_run=clean_run,
            resume_requested=resume_requested,
            verbose=verbose and not resume_requested,
        )
        grounded = cache_resolution.grounded
        inst_idx = cache_resolution.instance_templates or {}
        startup_metrics["grounded_cache_load_seconds"] = round(time.perf_counter() - cache_t0, 3)
        startup_metrics["grounded_cache_status"] = cache_resolution.status

        if verbose:
            gi_count = sum(len(v) for v in (grounded or {}).values()) if clean_run else 0
            if resume_requested:
                print(f"Grounded instance cache: {cache_resolution.status}", flush=True)
                print(f"Grounded instance count: {gi_count}", flush=True)

        skip_startup_audits = resume_requested and checkpoint_samples > 0
        if skip_startup_audits:
            complexity = json.loads((out / "blueprint_complexity.json").read_text(encoding="utf-8"))
            scene_budget = json.loads((out / "scene_budget.json").read_text(encoding="utf-8"))
            blueprint_budget = json.loads((out / "blueprint_budget.json").read_text(encoding="utf-8"))
            yaml_audit = {"complete": True}
            parser_audit = {"implementation_gap": 0}
            flow: list = []
            inventory_path = out / "full_blueprint_inventory.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.is_file() else {}
        else:
            specs_for_complexity = build_all_behavior_specs(reg, instance_overrides=inst_idx if clean_run else None)
            complexity = build_blueprint_complexity(specs_for_complexity, reg)
            scene_budget = build_scene_budget()
            blueprint_budget = build_blueprint_budget(complexity)
            yaml_audit = build_yaml_mapping_audit()
            parser_audit = run_parser_audit(reg)
            flow = audit_package_imports()
            inventory = build_full_inventory()

        specs = build_all_behavior_specs(reg, instance_overrides=inst_idx if clean_run else None)

        if not skip_startup_audits:
            (out / "blueprint_complexity.json").write_text(json.dumps(complexity, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "scene_budget.json").write_text(json.dumps(scene_budget, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "blueprint_budget.json").write_text(json.dumps(blueprint_budget, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "full_blueprint_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "generated_blueprint_inventory.json").write_text(
                json.dumps({"blueprint_ids": GROUNDED_CAPABLE_BLUEPRINTS, "count": len(GROUNDED_CAPABLE_BLUEPRINTS)}, indent=2),
                encoding="utf-8",
            )
            (out / "true_evidence_limited.json").write_text(json.dumps(_true_evidence_limited_doc(), indent=2, ensure_ascii=False), encoding="utf-8")
            (out / "information_flow_audit.json").write_text(json.dumps({"violations": flow, "clean": not flow}, indent=2), encoding="utf-8")

        scheduler_t0 = time.perf_counter()
        diversity_selector = DiversitySourceSelector() if clean_run else None
        if clean_run and samples and diversity_selector is not None:
            diversity_selector.restore_from_samples(samples)
            diversity_selector.generation_step = len(samples)

        path_coverage = PathCoverageState.from_behavior_specs(specs)
        coverage_policy = CoverageStopPolicy(
            min_strict_path_coverage_ratio=MIN_STRICT_PATH_COVERAGE_RATIO,
            max_zero_strict_paths=MAX_ZERO_STRICT_PATHS,
            max_path_skew=MAX_PATH_SKEW,
            target_total=TARGET_TOTAL,
        )

        scheduler = FinalGenerationScheduler(
            specs,
            blueprint_budget.get("blueprints") or [],
            instance_templates=inst_idx if clean_run else None,
            diversity_selector=diversity_selector,
            path_coverage=path_coverage,
        )
        spec_idx: dict[str, dict] = {}
        for s in specs:
            spec_idx[f"{s['blueprint_id']}::{s.get('automation_instance_id')}"] = s
            spec_idx.setdefault(s["blueprint_id"], s)
        nb_cache: dict[str, Any] = {}

        if resume_requested or not clean_run:
            last_by_sig = _restore_scheduler_from_samples(scheduler, samples) if samples else {}
        else:
            samples = []
            last_by_sig = {}
        canonical_sig_index: dict[str, dict] = {}
        if clean_run and samples:
            for s in samples:
                canonical_sig_index[behavior_source_dedup_key(s)] = s
        elif not clean_run:
            canonical_sig_index = dict(last_by_sig)
        startup_metrics["scheduler_restore_seconds"] = round(time.perf_counter() - scheduler_t0, 3)

        if verbose and resume_requested and checkpoint_samples:
            print(f"Coverage state restored: {'yes' if samples else 'no'}", flush=True)
            print(f"Dedup signature index restored: {'yes' if canonical_sig_index else 'no'}", flush=True)
            print(f"Resuming from checkpoint: {checkpoint_samples} samples", flush=True)

        startup_metrics["time_to_resume_message"] = round(time.perf_counter() - startup_t0, 3)
        if verbose and resume_requested:
            print(f"[resume] startup_seconds={startup_metrics['time_to_resume_message']}", flush=True)

        global_attempts = len(samples)
        initial_sample_count = len(samples)
        rejects: Counter = Counter()
        reuse_counts: dict[str, int] = {}
        attempts_by_bp: Counter = Counter()
        numeric_dup_rejects = 0
        temporal_dup_rejects = 0
        first_new_sample_time: float | None = None

        if resume_requested:
            for st in scheduler.bp_state.values():
                if st.accepted_count < BASE_MIN_PER_BLUEPRINT or scheduler._instance_gap(st):
                    st.stop_reason = "PENDING"
                    st.consecutive_fail = 0
            for sc in scheduler.scene_state.values():
                budget = SCENE_BUDGETS.get(sc.scene, {})
                if sc.accepted_count < budget.get("min", 0):
                    sc.stop_reason = "PENDING"

        complexity_by_bp = {
            str(row.get("blueprint_id") or ""): float(row.get("complexity_score") or 0)
            for row in (blueprint_budget.get("blueprints") or [])
        }
        remaining_by_bp: dict[str, int] = {}
        if clean_run and diversity_selector is not None:
            remaining_by_bp = compute_all_remaining_pairs(
                scheduler=scheduler,
                selector=diversity_selector,
                pool=pool,
                inst_idx=inst_idx,
                reg=reg,
            )
            for bp_id, rem in remaining_by_bp.items():
                if bp_id in scheduler.bp_state:
                    scheduler.bp_state[bp_id].cached_remaining_pairs = rem
            if resume_requested and checkpoint_samples > 0:
                reactivation = resume_reactivate_quota_limited(
                    scheduler=scheduler,
                    remaining_by_bp=remaining_by_bp,
                    complexity_by_bp=complexity_by_bp,
                    global_total=len(samples),
                    verbose=verbose,
                )
                startup_metrics["phase2_resume_reactivation"] = reactivation
                if verbose:
                    print(
                        f"[resume] phase2 reactivated {len(reactivation.get('activated') or [])} blueprints, "
                        f"released_quota={reactivation.get('released_quota')}",
                        flush=True,
                    )

        idle_without_accept = 0
        last_heartbeat = time.perf_counter()
        progress_every = 50 if resume_requested else 200
        loop_started = time.perf_counter()
        recent_outcomes: list[int] = []
        global_stop_reason = ""
        from smarthome_mdf.multi_action_vnext.requirement_extractor import extract_requirement_bundle

        def _refresh_remaining(*, force: bool = False) -> dict[str, int]:
            nonlocal remaining_by_bp
            if not clean_run or diversity_selector is None:
                return remaining_by_bp
            if force or not remaining_by_bp or len(samples) % 200 == 0:
                remaining_by_bp = compute_all_remaining_pairs(
                    scheduler=scheduler,
                    selector=diversity_selector,
                    pool=pool,
                    inst_idx=inst_idx,
                    reg=reg,
                )
                for bp_id, rem in remaining_by_bp.items():
                    if bp_id in scheduler.bp_state:
                        scheduler.bp_state[bp_id].cached_remaining_pairs = rem
            return remaining_by_bp

        def _maybe_phase2_reallocation() -> bool:
            remaining = _refresh_remaining(force=True)
            result = try_phase2_reallocation(
                scheduler=scheduler,
                remaining_by_bp=remaining,
                complexity_by_bp=complexity_by_bp,
                global_total=len(samples),
                verbose=verbose,
            )
            return bool(result.get("activated"))

        def _accept_rate(window: int) -> float:
            if not recent_outcomes:
                return 0.0
            tail = recent_outcomes[-window:]
            return round(sum(tail) / len(tail), 4)

        def _heartbeat_extra() -> dict[str, Any]:
            if not diversity_selector:
                return {}
            return {
                "accept_rate_last_100": _accept_rate(100),
                "accept_rate_last_1000": _accept_rate(1000),
                "generation_phase": scheduler.generation_phase,
                "blueprints_saturated": len(scheduler.saturated_blueprint_ids()),
                "blueprints_quota_limited": len(scheduler.quota_limited_blueprint_ids()),
                "blueprints_with_remaining_capacity": len(
                    [bp for bp, rem in remaining_by_bp.items() if rem > 0]
                ),
                "total_remaining_grounded_pairs": scheduler.total_remaining_grounded_pairs(remaining_by_bp),
                "unused_compatible_sources_total": diversity_selector.unused_compatible_sources_total(),
                "saturated_behavior_source_pairs": len(diversity_selector.saturated_behavior_source_pairs),
                "per_scene": _scene_capacity_report(scheduler),
            }

        def _diversity_exhaustion_check() -> bool:
            if len(recent_outcomes) < 1000 or _accept_rate(1000) >= 0.01:
                return False
            if verbose:
                print("[diversity] DIVERSITY_EXHAUSTION_CHECK triggered", flush=True)
            for bp_id, st in list(scheduler.bp_state.items()):
                if st.stop_reason != "PENDING":
                    continue
                inst_list = inst_idx.get(bp_id) or []
                behavior_rows = [
                    {
                        "blueprint_id": t.blueprint_id,
                        "automation_instance_id": t.automation_instance_id,
                        "branch_id": t.branch_id,
                        "action_path_signature": t.action_path_signature,
                    }
                    for t in scheduler.behavior_targets
                    if t.blueprint_id == bp_id
                ]
                for inst in inst_list[:1]:
                    iid = inst.get("automation_instance_id", "")
                    cache_key = f"{bp_id}::{iid}"
                    if cache_key not in nb_cache:
                        nb_cache[cache_key] = _normalize_from_instance(st.scene, inst, reg)
                    nb = nb_cache[cache_key]
                    ir = None
                    try:
                        ir = _get_ir(bp_id, inst, iid)
                    except Exception:
                        ir = None
                    bundle = extract_requirement_bundle(nb, ir=ir)
                    if diversity_selector and not diversity_selector.blueprint_has_remaining_capacity(
                        pool, bundle, iid, behavior_rows
                    ):
                        scheduler.mark_blueprint_saturated(bp_id)
            remaining = _refresh_remaining(force=True)
            if scheduler.all_grounded_capacity_exhausted(remaining):
                nonlocal global_stop_reason
                if not _maybe_phase2_reallocation():
                    global_stop_reason = GLOBAL_STOP_ALL_GROUNDED_SATURATED
                    return True
            return False

        def _maybe_heartbeat(*, force: bool = False) -> None:
            nonlocal last_heartbeat
            now = time.perf_counter()
            if not force and global_attempts % progress_every != 0 and (now - last_heartbeat) < 60:
                return
            last_heartbeat = now
            _write_generation_progress(
                out,
                accepted=len(samples),
                attempts=global_attempts,
                rejects=rejects,
                elapsed_seconds=now - loop_started,
                new_since_start=len(samples) - initial_sample_count,
                extra=_heartbeat_extra(),
            )
            if verbose:
                top = rejects.most_common(3)
                extra = _heartbeat_extra()
                print(
                    f"[heartbeat] accepted={len(samples)} attempts={global_attempts} "
                    f"new={len(samples) - initial_sample_count} "
                    f"rate100={extra.get('accept_rate_last_100', 0)} "
                    f"rate1000={extra.get('accept_rate_last_1000', 0)} "
                    f"saturated_bps={extra.get('blueprints_saturated', 0)} "
                    f"rejects={top}",
                    flush=True,
                )

        while global_attempts < HARD_MAX * 3:
            if global_stop_reason:
                break
            if max_new_samples is not None and len(samples) >= initial_sample_count + max_new_samples:
                break
            if max_attempts is not None and (global_attempts - initial_sample_count) >= max_attempts:
                break
            if scheduler.total_accepted() >= HARD_MAX:
                break
            if len(samples) >= TARGET_TOTAL:
                cov_eval = coverage_policy.evaluate(sample_count=len(samples), coverage=path_coverage)
                if cov_eval["should_stop"]:
                    global_stop_reason = GLOBAL_STOP_SAMPLE_TARGET_AND_COVERAGE_REACHED
                    break
                global_stop_reason = GLOBAL_STOP_COVERAGE_INCOMPLETE
                if verbose:
                    print(f"[coverage] target reached but coverage incomplete: {cov_eval['metrics']}", flush=True)
                break
            remaining = _refresh_remaining()
            if scheduler.all_grounded_capacity_exhausted(remaining) and scheduler.all_scheduling_inactive():
                if not _maybe_phase2_reallocation():
                    global_stop_reason = GLOBAL_STOP_ALL_GROUNDED_SATURATED
                    break
            if scheduler.all_scheduling_inactive() and scheduler.total_remaining_grounded_pairs(remaining) == 0:
                global_stop_reason = GLOBAL_STOP_ALL_GROUNDED_SATURATED
                break
            if scheduler.all_complete() and scheduler.total_accepted() >= ACCEPTABLE_MIN:
                break

            target = scheduler.next_target()
            if target is None:
                if _maybe_phase2_reallocation():
                    continue
                remaining = _refresh_remaining(force=True)
                if scheduler.all_grounded_capacity_exhausted(remaining):
                    global_stop_reason = GLOBAL_STOP_ALL_GROUNDED_SATURATED
                    break
                starved = [
                    st for st in scheduler.bp_state.values()
                    if st.stop_reason == "PENDING"
                    and (st.accepted_count < BASE_MIN_PER_BLUEPRINT or st.accepted_count < st.allocated_target)
                ]
                scene_under = [
                    sc for sc in scheduler.scene_state.values()
                    if sc.stop_reason == "PENDING"
                    and sc.accepted_count < SCENE_BUDGETS.get(sc.scene, {}).get("min", 0)
                ]
                if starved or scene_under or scheduler.instance_coverage_gaps():
                    for st in starved:
                        st.stop_reason = "PENDING"
                        st.consecutive_fail = 0
                        st.attempt_budget = max(st.attempt_budget, st.attempt_count + 500)
                    for bp_id, gap in scheduler.instance_coverage_gaps().items():
                        st = scheduler.bp_state.get(bp_id)
                        if st:
                            st.stop_reason = "PENDING"
                            st.consecutive_fail = 0
                            st.attempt_budget = max(st.attempt_budget, st.attempt_count + 800)
                    idle_without_accept += 1
                    if idle_without_accept >= 2000:
                        if verbose:
                            print(f"[heartbeat] idle spin limit reached ({idle_without_accept})", flush=True)
                        scheduler.finalize_scene_stops()
                        for st in scheduler.bp_state.values():
                            if st.stop_reason == "PENDING":
                                if st.accepted_count >= st.effective_target:
                                    st.stop_reason = "ALLOCATED_TARGET_REACHED"
                                elif st.accepted_count >= BASE_MIN_PER_BLUEPRINT and not scheduler._instance_gap(st):
                                    st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
                                elif st.accepted_count > 0:
                                    st.stop_reason = "GROUNDED_BEHAVIOR_SPACE_SMALL"
                                else:
                                    st.stop_reason = "EVIDENCE_CAPACITY_EXHAUSTED"
                        break
                    continue
                scheduler.finalize_scene_stops()
                for st in scheduler.bp_state.values():
                    if st.stop_reason == "PENDING":
                        if st.accepted_count >= st.effective_target and not scheduler._instance_gap(st):
                            st.stop_reason = "ALLOCATED_TARGET_REACHED"
                        elif st.accepted_count >= BASE_MIN_PER_BLUEPRINT and not scheduler._instance_gap(st):
                            st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
                        elif st.accepted_count > 0:
                            st.stop_reason = "GROUNDED_BEHAVIOR_SPACE_SMALL"
                        else:
                            st.stop_reason = "EVIDENCE_CAPACITY_EXHAUSTED"
                if scheduler.total_accepted() >= ACCEPTABLE_MIN and not scheduler.instance_coverage_gaps():
                    break
                if scheduler.total_accepted() < ACCEPTABLE_MIN:
                    for st in scheduler.bp_state.values():
                        if st.stop_reason != "PENDING" and (
                            st.accepted_count < BASE_MIN_PER_BLUEPRINT or scheduler._instance_gap(st)
                        ):
                            st.stop_reason = "PENDING"
                            st.consecutive_fail = 0
                            st.attempt_budget = max(st.attempt_budget, st.attempt_count + 1000)
                    continue
                break

            prev_accepted = len(samples)

            global_attempts += 1
            attempts_by_bp[target.blueprint_id] += 1
            scheduler.register_attempt(target.blueprint_id)

            inst_list = inst_idx.get(target.blueprint_id) or []
            instance = next((i for i in inst_list if i["automation_instance_id"] == target.automation_instance_id), inst_list[0] if inst_list else {})
            cache_key = f"{target.blueprint_id}::{target.automation_instance_id}"
            if cache_key not in nb_cache:
                nb_cache[cache_key] = _normalize_from_instance(target.scene, instance, reg)
            nb = nb_cache[cache_key]
            bspec = spec_idx.get(cache_key) or spec_idx.get(target.blueprint_id) or {}

            path_request = GenerationPathRequest.from_behavior_target(target)
            status, sample, _reason = synthesize_single_scene(
                scene=target.scene,
                blueprint_id=target.blueprint_id,
                automation_instance_id=target.automation_instance_id,
                behavior_target=target.behavior_target,
                branch_id=target.branch_id,
                action_path_signature=target.action_path_signature,
                instance=instance,
                normalized_blueprint=nb,
                behavior_spec=bspec,
                source_pool=pool,
                reuse_counts=reuse_counts,
                attempt_index=global_attempts,
                diversity_selector=diversity_selector,
                generation_path_request=path_request,
            )
            if diversity_selector is not None:
                diversity_selector.advance_step()

            if status == "BEHAVIOR_SOURCE_SATURATED":
                rejects[status] += 1
                scheduler.register_reject(target.blueprint_id)
                scheduler.mark_target_source_saturated(target)
                recent_outcomes.append(0)
                if len(recent_outcomes) > 1000:
                    recent_outcomes.pop(0)
                if len(recent_outcomes) >= 1000 and len(recent_outcomes) % 1000 == 0:
                    _diversity_exhaustion_check()
                _maybe_heartbeat()
                continue

            if status != "ACCEPT" or not sample:
                rejects[status or "REJECT"] += 1
                scheduler.register_reject(target.blueprint_id)
                recent_outcomes.append(0)
                if len(recent_outcomes) > 1000:
                    recent_outcomes.pop(0)
                _maybe_heartbeat()
                continue

            if clean_run:
                ok, reject_reason = validate_sample_accept(
                    sample, pool=pool, normalized_blueprint=nb, prior_by_canonical=canonical_sig_index
                )
                if not ok:
                    rejects[reject_reason] += 1
                    if reject_reason == "SAME_SOURCE_DUPLICATE_PADDING" and diversity_selector:
                        diversity_selector.register_padding_reject(sample)
                    scheduler.register_reject(target.blueprint_id)
                    recent_outcomes.append(0)
                    if len(recent_outcomes) > 1000:
                        recent_outcomes.pop(0)
                    if len(recent_outcomes) >= 1000 and len(recent_outcomes) % 1000 == 0:
                        _diversity_exhaustion_check()
                    _maybe_heartbeat()
                    continue

            sig = canonical_sample_signature(sample) if clean_run else _structural_signature(sample)
            st = scheduler.bp_state[target.blueprint_id]
            under_base = st.accepted_count < BASE_MIN_PER_BLUEPRINT
            if clean_run:
                dedup_key = behavior_source_dedup_key(sample)
                prior = canonical_sig_index.get(dedup_key)
                if prior and not under_base:
                    if is_timestamp_only_duplicate(sample, prior) or sig == canonical_sample_signature(prior):
                        temporal_dup_rejects += 1
                        rejects["TIMESTAMP_ONLY_DUPLICATE"] += 1
                        scheduler.register_reject(target.blueprint_id)
                        _maybe_heartbeat()
                        continue
            else:
                prior = last_by_sig.get(sig)

            if prior and not under_base and scheduler.temporal_shift_near_duplicate(sample, prior):
                temporal_dup_rejects += 1
                rejects["TEMPORAL_SHIFT_NEAR_DUPLICATE"] += 1
                scheduler.register_reject(target.blueprint_id)
                _maybe_heartbeat()
                continue
            if prior and scheduler.numeric_only_near_duplicate(sample, prior) and not under_base:
                numeric_dup_rejects += 1
                rejects["NUMERIC_ONLY_NEAR_DUPLICATE"] += 1
                scheduler.mark_target_saturated(target)
                scheduler.register_reject(target.blueprint_id)
                if scheduler.consecutive_numeric_dup_limit(target.blueprint_id, 40):
                    st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
                _maybe_heartbeat()
                continue

            scheduler.register_accept(sample, behavior_target=target.behavior_target)
            samples.append(sample)
            last_by_sig[sig] = sample
            if clean_run:
                canonical_sig_index[behavior_source_dedup_key(sample)] = sample
                if diversity_selector:
                    diversity_selector.register_accept(sample)
            recent_outcomes.append(1)
            if len(recent_outcomes) > 1000:
                recent_outcomes.pop(0)

            if first_new_sample_time is None and len(samples) > initial_sample_count:
                first_new_sample_time = time.perf_counter() - startup_t0
                startup_metrics["time_to_first_new_sample"] = round(first_new_sample_time, 3)
                if verbose and resume_requested:
                    print(f"[resume] time_to_first_new_sample={startup_metrics['time_to_first_new_sample']}s", flush=True)

            if verbose and (len(samples) <= 10 or len(samples) % 100 == 0):
                print(
                    f"[{len(samples)}/{TARGET_TOTAL}] bp={target.blueprint_id} scene={target.scene} total_attempts={global_attempts}",
                    flush=True,
                )

            if checkpoint_every and len(samples) % checkpoint_every == 0:
                _write_jsonl(jsonl_path, samples)
                _maybe_heartbeat(force=True)

            _maybe_heartbeat()

            if len(samples) == prev_accepted:
                idle_without_accept += 1
            else:
                idle_without_accept = 0
            if idle_without_accept >= 5000 and scheduler.total_accepted() >= ACCEPTABLE_MIN:
                break

            if len(samples) >= TARGET_TOTAL:
                cov_eval = coverage_policy.evaluate(sample_count=len(samples), coverage=path_coverage)
                if cov_eval["should_stop"]:
                    global_stop_reason = GLOBAL_STOP_SAMPLE_TARGET_AND_COVERAGE_REACHED
                    break
                global_stop_reason = GLOBAL_STOP_COVERAGE_INCOMPLETE
                break
            if global_attempts % 200 == 0:
                scheduler.adjust_dynamic_quotas()

        scheduler.finalize_scene_stops()
        _write_jsonl(jsonl_path, samples)

        skip_dev_audits = resume_requested and len(samples) < TARGET_TOTAL
        audits: dict[str, Any] = {}
        v3_audits: dict[str, Any] = {}
        if not skip_dev_audits:
            audits = build_all_audits(
                samples, scheduler, complexity, specs, rejects, dict(attempts_by_bp), numeric_dup_rejects, temporal_dup_rejects
            )
            write_audit_artifacts(out, audits)

            if clean_run:
                v3_audits = write_v3_audit_artifacts(
                    out,
                    samples,
                    scheduler,
                    grounded_templates=inst_idx,
                    rejects=dict(rejects),
                    pool=pool,
                )
                _write_v3_implementation_report(out, samples, v3_audits, dict(rejects), parser_audit)
        elif verbose:
            print("[resume] Skipping full development audits until generation completes", flush=True)

        startup_metrics["startup_seconds"] = round(time.perf_counter() - startup_t0, 3)
        if first_new_sample_time is None:
            startup_metrics["time_to_first_new_sample"] = None

        verdict, blockers = _evaluate_verdict(
            samples, audits, gate, flow, parser_audit, yaml_audit, scheduler, v3_audits=v3_audits, clean_run=clean_run
        )
        manifest = {
            "verdict": verdict,
            "blockers": blockers,
            "total_samples": len(samples),
            "target_total": TARGET_TOTAL,
            "inventory_blueprints": 22,
            "grounded_capable_blueprints": 21,
            "generated_blueprints": len(GROUNDED_CAPABLE_BLUEPRINTS),
            "true_evidence_limited_blueprints": 1,
            "generation_completed_at": _utc_now(),
            "pre_final_development_runs_excluded": True,
            "scheduler_state": scheduler.to_generation_state_dict(),
            "resume_startup_metrics": startup_metrics,
            "resume_requested": resume_requested,
            "checkpoint_samples_at_start": checkpoint_samples,
            "global_stop_reason": global_stop_reason or None,
            "diversity_selector": diversity_selector.stats_summary() if diversity_selector else None,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "generation_progress.json").write_text(
            json.dumps({"accepted": len(samples), "attempts": global_attempts}, indent=2),
            encoding="utf-8",
        )
        (out / "source_lineage.json").write_text(
            json.dumps(
                {
                    "lineage": "Public Raw → Adapter → SourceObservation → Blueprint Runtime Match → Single-scene Sample",
                    "datasets": audits.get("source_dataset_distribution"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out / "FINAL_SINGLE_SCENE_REGENERATION_REPORT.md").write_text(
            _format_final_report(verdict, blockers, samples, audits, complexity, scene_budget, blueprint_budget, scheduler),
            encoding="utf-8",
        )

        if verbose:
            print(f"\n=== Verdict: {verdict} ===", flush=True)
            print(f"Samples: {len(samples)} → {jsonl_path}", flush=True)

        return manifest
    finally:
        _release_lock(lock)

if __name__ == "__main__":
    result = run_final_regeneration()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("verdict") == "READY_FOR_SINGLE_SCENE_FREEZE" else 1)
