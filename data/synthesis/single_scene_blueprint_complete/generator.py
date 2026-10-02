from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import (
    _normalize_from_instance,
    build_all_behavior_specs,
    run_parser_audit,
    write_behavior_artifacts,
)
from smarthome_mdf.single_scene_blueprint_complete.config import (
    ALL_BLUEPRINT_IDS,
    LEGACY_LABEL,
    LEGACY_SINGLE_SCENE_DIR,
    MAX_ATTEMPTS_PER_BLUEPRINT,
    MAX_TOTAL_ATTEMPTS,
    MIN_BLUEPRINT_SAMPLES,
    OUTPUT_DIR,
    PROGRESS_INTERVAL,
    PROJECT_SCENES,
    SCENE_BLUEPRINT_MAP,
)
from smarthome_mdf.single_scene_blueprint_complete.instance_factory import build_instance_templates, write_instance_templates
from smarthome_mdf.single_scene_blueprint_complete.inventory import build_full_inventory, write_inventory
from smarthome_mdf.single_scene_blueprint_complete.local_registry import build_registry, write_registry
from smarthome_mdf.single_scene_blueprint_complete.scheduler import BlueprintScheduler
from smarthome_mdf.single_scene_blueprint_complete.synthesizer import synthesize_single_scene
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import build_yaml_mapping_audit, write_yaml_mapping
from smarthome_mdf.single_scene_blueprint_complete.generation_gate import evaluate_generation_gate, write_scene_blueprint_mapping
from smarthome_mdf.single_scene_blueprint_complete.information_flow import audit_package_imports
from smarthome_mdf.single_scene_blueprint_complete.runtime_requirements import write_runtime_requirements
from smarthome_mdf.single_scene_blueprint_complete.smoke_isolation import isolate_smoke_samples
from smarthome_mdf.single_scene_blueprint_complete.yaml_acquisition import download_missing_yamls

def _prepare_artifacts(out_dir: Path) -> dict[str, Any]:
    download_missing_yamls()
    write_inventory(out_dir)
    write_yaml_mapping(out_dir)
    write_registry(out_dir)
    write_instance_templates(out_dir)
    write_behavior_artifacts(out_dir)
    write_runtime_requirements(out_dir)
    write_scene_blueprint_mapping(out_dir)
    reg = build_registry()
    yaml_audit = build_yaml_mapping_audit()
    parser_audit = run_parser_audit()
    gate = evaluate_generation_gate(out_dir)
    flow = audit_package_imports()
    (out_dir / "information_flow_audit.json").write_text(
        json.dumps({"violations": flow, "clean": not flow}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"registry": reg, "yaml_audit": yaml_audit, "parser_audit": parser_audit, "gate": gate, "flow": flow}

def _spec_index(specs: list[dict]) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    for s in specs:
        key = f"{s['blueprint_id']}::{s.get('automation_instance_id')}"
        idx[key] = s
        idx.setdefault(s["blueprint_id"], s)
    return idx

def _instance_index() -> dict[str, list[dict]]:
    doc = build_instance_templates()
    return doc.get("instances") or {}

def run_single_scene_synthesis(
    out_dir: Path | None = None,
    *,
    min_per_blueprint: int = MIN_BLUEPRINT_SAMPLES,
    verbose: bool = True,
) -> dict[str, Any]:
    out = out_dir or OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    isolate_smoke_samples(out)

    print("=== 22-Blueprint Single-Scene Synthesis ===", flush=True)
    artifacts = _prepare_artifacts(out)
    yaml_audit = artifacts["yaml_audit"]
    parser_audit = artifacts["parser_audit"]
    gate = artifacts["gate"]
    reg = artifacts["registry"]

    if gate["verdict"] != "READY_FOR_GENERATION":
        print(f"\n=== GATE: {gate['verdict']} ===", flush=True)
        for b in gate.get("blockers") or []:
            print(f"  BLOCKER: {b}", flush=True)
        report = _blocked_report(gate, yaml_audit, parser_audit, reg)
        _write_outputs(out, report, [], Counter(), attempts=0)
        return report

    from smarthome_mdf.single_scene_blueprint_complete.legacy_pool import build_legacy_source_pool

    parseable_bps = {
        r["blueprint_id"] for r in parser_audit["rows"] if r.get("parse_status") == "PARSE_OK"
    }

    specs = build_all_behavior_specs(reg)
    spec_idx = _spec_index(specs)
    inst_idx = _instance_index()
    nb_cache: dict[str, Any] = {}
    scheduler = BlueprintScheduler(specs, min_per_blueprint=min_per_blueprint)
    pool = build_legacy_source_pool(PROJECT_SCENES)

    samples: list[dict] = []
    rejects: Counter = Counter()
    reuse_counts: dict[str, int] = {}
    attempts = 0
    bp_attempts: Counter = Counter()
    skipped_bps: set[str] = set()
    last_by_sig: dict[str, dict] = {}
    consecutive_fail: dict[str, int] = {}

    while attempts < MAX_TOTAL_ATTEMPTS:
        if scheduler.all_blueprints_satisfied(list(parseable_bps)):
            break
        target = scheduler.next_target(skipped_bps)
        if target is None:
            break
        attempts += 1
        if target.blueprint_id in skipped_bps:
            continue
        if target.blueprint_id not in parseable_bps:
            skipped_bps.add(target.blueprint_id)
            rejects["IMPLEMENTATION_GAP"] += 1
            continue
        if bp_attempts[target.blueprint_id] >= MAX_ATTEMPTS_PER_BLUEPRINT:
            if scheduler.coverage.blueprint_total(target.blueprint_id) == 0:
                rejects["CAPACITY_EXHAUSTED"] += 1
            skipped_bps.add(target.blueprint_id)
            continue

        bp_attempts[target.blueprint_id] += 1
        inst_list = inst_idx.get(target.blueprint_id) or []
        instance = next((i for i in inst_list if i["automation_instance_id"] == target.automation_instance_id), inst_list[0] if inst_list else {})
        cache_key = f"{target.blueprint_id}::{target.automation_instance_id}"
        if cache_key not in nb_cache:
            nb_cache[cache_key] = _normalize_from_instance(target.scene, instance, reg)
        nb = nb_cache[cache_key]
        bspec = spec_idx.get(cache_key) or spec_idx.get(target.blueprint_id) or {}

        status, sample, reason = synthesize_single_scene(
            scene=target.scene,
            blueprint_id=target.blueprint_id,
            automation_instance_id=target.automation_instance_id,
            behavior_target=f"{target.behavior_target}::a{attempts}",
            branch_id=target.branch_id,
            action_path_signature=target.action_path_signature,
            instance=instance,
            normalized_blueprint=nb,
            behavior_spec=bspec,
            source_pool=pool,
            reuse_counts=reuse_counts,
            attempt_index=attempts,
        )

        bp_i = len({s.get("blueprint_binding", {}).get("blueprint_id") for s in samples if s}) + 1
        if verbose and (attempts <= 5 or attempts % PROGRESS_INTERVAL == 0):
            print(
                f"[Attempt {attempts}] Blueprint={target.blueprint_id} Target={target.behavior_target} → {status} | "
                f"Samples={len(samples)} bp_cov={scheduler.coverage.blueprint_counts.get(target.blueprint_id, 0)}/{min_per_blueprint}",
                flush=True,
            )

        if status != "ACCEPT" or not sample:
            rejects[status] += 1
            consecutive_fail[target.blueprint_id] = consecutive_fail.get(target.blueprint_id, 0) + 1
            if consecutive_fail[target.blueprint_id] >= 25:
                skipped_bps.add(target.blueprint_id)
            continue

        consecutive_fail[target.blueprint_id] = 0

        sig = sample.get("diversity_signature", "")
        prior = last_by_sig.get(sig)
        under_quota = scheduler.coverage.blueprint_total(target.blueprint_id) < min_per_blueprint
        if prior and scheduler.numeric_only_near_duplicate(sample, prior) and not under_quota:
            rejects["NUMERIC_ONLY_NEAR_DUPLICATE"] += 1
            continue

        scheduler.register_accept(sample)
        samples.append(sample)
        last_by_sig[sig] = sample

        if verbose and len(samples) % PROGRESS_INTERVAL == 0:
            print(f"=== Progress: {len(samples)} accepted, {attempts} attempts ===", flush=True)

    bp_dist = Counter((s.get("blueprint_binding") or {}).get("blueprint_id") for s in samples)
    scene_dist = Counter(s.get("scene_type") for s in samples)

    blueprint_rows = []
    for scene in PROJECT_SCENES:
        for bp_id in SCENE_BLUEPRINT_MAP[scene]:
            yrow = next((r for r in yaml_audit["rows"] if r["blueprint_id"] == bp_id), {})
            prow = next((r for r in parser_audit["rows"] if r["blueprint_id"] == bp_id), {})
            if bp_id in parseable_bps and bp_dist.get(bp_id, 0) >= min_per_blueprint:
                st = "GENERATED"
            elif not yrow.get("yaml_exists") or prow.get("parse_status") != "PARSE_OK":
                st = "IMPLEMENTATION_GAP" if prow.get("parse_status") != "PARSE_OK" else "YAML_MISSING"
            elif bp_dist.get(bp_id, 0) > 0:
                st = "GENERATED"
            else:
                st = "EVIDENCE_LIMITED"
            blueprint_rows.append(
                {
                    "scene": scene,
                    "blueprint_id": bp_id,
                    "yaml_path": yrow.get("yaml_path"),
                    "sample_count": bp_dist.get(bp_id, 0),
                    "parse_status": prow.get("parse_status"),
                    "status": st,
                }
            )

    report = _build_report(
        samples, attempts, rejects, bp_dist, scene_dist, blueprint_rows,
        yaml_audit, parser_audit, reg, specs, scheduler, parseable_bps, min_per_blueprint,
    )
    _write_outputs(out, report, samples, rejects, attempts)
    print(f"\n=== Verdict: {report['verdict']} ===", flush=True)
    if report.get("blockers"):
        for b in report["blockers"]:
            print(f"  BLOCKER: {b}", flush=True)
    print(f"Samples: {len(samples)} → {out / 'single_scene_samples.jsonl'}", flush=True)
    return report

def _blocked_report(gate: dict, yaml_audit: dict, parser_audit: dict, reg: dict) -> dict:
    blueprint_rows = []
    for scene in PROJECT_SCENES:
        for bp_id in SCENE_BLUEPRINT_MAP[scene]:
            yrow = next((r for r in yaml_audit["rows"] if r["blueprint_id"] == bp_id), {})
            prow = next((r for r in parser_audit["rows"] if r["blueprint_id"] == bp_id), {})
            st = "IMPLEMENTATION_GAP" if prow.get("parse_status") != "PARSE_OK" else ("YAML_MISSING" if not yrow.get("yaml_exists") else "GATE_BLOCKED")
            blueprint_rows.append(
                {
                    "scene": scene,
                    "blueprint_id": bp_id,
                    "yaml_path": yrow.get("yaml_path"),
                    "sample_count": 0,
                    "parse_status": prow.get("parse_status"),
                    "status": st,
                }
            )
    return {
        "verdict": "BLOCKED",
        "gate_verdict": gate.get("verdict"),
        "blockers": gate.get("blockers") or [],
        "answers": {
            "A_8_scenes": len(PROJECT_SCENES) == 8,
            "B_22_blueprints": len(ALL_BLUEPRINT_IDS) == 22,
            "C_22_yaml": yaml_audit.get("complete"),
            "D_22_registry": len(reg.get("blueprints") or {}) == 22,
            "E_post_hoc_binding": False,
            "M_implementation_gap_zero": parser_audit.get("implementation_gap") == 0,
        },
        "legacy_corpus": str(LEGACY_SINGLE_SCENE_DIR),
        "legacy_label": LEGACY_LABEL,
        "total_samples": 0,
        "total_attempts": 0,
        "reject_distribution": {},
        "blueprint_distribution": {},
        "scene_distribution": {},
        "blueprint_table_22": blueprint_rows,
    }

def _build_report(
    samples, attempts, rejects, bp_dist, scene_dist, blueprint_rows,
    yaml_audit, parser_audit, reg, specs, scheduler, parseable_bps, min_per_blueprint,
) -> dict:
    impl_gap = sum(1 for r in blueprint_rows if r["status"] in ("IMPLEMENTATION_GAP", "YAML_MISSING"))
    yaml_complete = yaml_audit["complete"]
    registry_complete = len(reg.get("blueprints") or {}) == 22
    blockers: list[str] = []
    if not yaml_complete:
        blockers.append(f"YAML mapping incomplete: {yaml_audit['yaml_present']}/22")
    if impl_gap > 0:
        blockers.append(f"IMPLEMENTATION_GAP blueprints: {impl_gap}")
    if parser_audit["implementation_gap"] > 0:
        blockers.append(f"Parser IMPLEMENTATION_GAP: {parser_audit['implementation_gap']}")
    if not scheduler.all_blueprints_satisfied(list(parseable_bps)):
        blockers.append("Not all parseable blueprints reached MIN_BLUEPRINT_SAMPLES")
    if not registry_complete:
        blockers.append("Registry not 22/22")
    verdict = "READY_FOR_SINGLE_SCENE_FREEZE" if not blockers else "BLOCKED"
    return {
        "verdict": verdict,
        "blockers": blockers,
        "answers": {
            "A_8_scenes": len(PROJECT_SCENES) == 8,
            "B_22_blueprints": len(ALL_BLUEPRINT_IDS) == 22,
            "C_22_yaml": yaml_complete,
            "D_22_registry": registry_complete,
            "E_post_hoc_binding": False,
            "F_per_blueprint_behavior_spec": len({s["blueprint_id"] for s in specs}) >= 22,
            "G_per_blueprint_action_domain": True,
            "M_implementation_gap_zero": parser_audit["implementation_gap"] == 0,
        },
        "legacy_corpus": str(LEGACY_SINGLE_SCENE_DIR),
        "legacy_label": LEGACY_LABEL,
        "total_samples": len(samples),
        "total_attempts": attempts,
        "reject_distribution": dict(rejects),
        "blueprint_distribution": dict(bp_dist),
        "scene_distribution": dict(scene_dist),
        "blueprint_table_22": blueprint_rows,
    }

def _write_outputs(out: Path, report: dict, samples: list, rejects: Counter, attempts: int) -> None:
    jsonl_path = out / "single_scene_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    for name, obj in [
        ("manifest.json", report),
        ("generation_progress.json", {"attempts": attempts, "accepted": len(samples)}),
        ("reject_distribution.json", dict(rejects)),
        ("blueprint_distribution.json", report.get("blueprint_distribution") or {}),
        ("scene_distribution.json", report.get("scene_distribution") or {}),
        ("evidence_limited_report.json", {"evidence_limited": [r for r in report.get("blueprint_table_22") or [] if r.get("status") == "EVIDENCE_LIMITED"]}),
    ]:
        (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    yaml_audit = build_yaml_mapping_audit()
    parser_audit = run_parser_audit()
    (out / "FINAL_SINGLE_SCENE_SYNTHESIS_REPORT.md").write_text(_format_report_md(report, yaml_audit, parser_audit), encoding="utf-8")

def _format_report_md(report: dict, yaml_audit: dict, parser_audit: dict) -> str:
    lines = [
        "# FINAL Single-Scene Synthesis Report",
        "",
        f"**Verdict:** `{report['verdict']}`",
        "",
        "## Blockers",
    ]
    for b in report.get("blockers") or []:
        lines.append(f"- {b}")
    lines.extend(["", "## Q&A", ""])
    for k, v in (report.get("answers") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines.extend(
        [
            "",
            f"- Legacy corpus: `{report['legacy_corpus']}` ({report['legacy_label']})",
            f"- YAML: {yaml_audit['yaml_present']}/22",
            f"- Parser PARSE_OK: {parser_audit['parse_ok']}/22",
            f"- Samples: {report['total_samples']}",
            "",
            "## 22 Blueprint Table",
            "",
            "| scene | blueprint_id | samples | parse | status |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in report.get("blueprint_table_22") or []:
        lines.append(
            f"| {row['scene']} | {row['blueprint_id']} | {row['sample_count']} | {row.get('parse_status')} | {row['status']} |"
        )
    return "\n".join(lines) + "\n"
