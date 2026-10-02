from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.multi_action_frozen_v1.config import SS_REPAIR_DIR
from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    build_blueprint_action_inventory,
    enumerate_paths_from_spec,
    match_sample_to_path,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _path_row_key(path) -> str:
    return f"{path.branch_id}::{path.action_path_signature}::{path.action_template_key}"

def _inventory_path_key(action: dict) -> str:
    return f"{action['branch_id']}::{action['action_path_signature']}::{action['action_template_key']}"

def _b0_primary_label(sample: dict) -> tuple[str, str]:

    b0 = sample.get("b0_output") or {}
    semantic = list(b0.get("semantic_actions") or [])
    if not semantic:
        status = str(b0.get("execution_status") or "NO_ACTION")
        return "B0_NO_ACTION", status
    services = sorted({str(a.get("service") or "unknown") for a in semantic})
    if len(services) == 1:
        return services[0], services[0]
    return "B0_MULTI_ACTION", "|".join(services)

def _match_b0_to_path(sample: dict, paths: list, semantic: list[dict]):

    if not semantic:
        return set()
    path_by_branch_service: dict[tuple[str, str], list] = defaultdict(list)
    for p in paths:
        svc = p.action_path_signature.split("|")[0] if "|" in p.action_path_signature else p.action_path_signature
        path_by_branch_service[(str(p.branch_id), svc)].append(p)

    matched: set[str] = set()
    for act in semantic:
        prov = act.get("provenance") or {}
        branch = str(prov.get("source_branch") or prov.get("branch_id") or "")
        service = str(act.get("service") or "")
        if not service:
            continue
        cands = path_by_branch_service.get((branch, service), [])
        if not cands:
            cands = [p for p in paths if p.action_path_signature == service or p.action_path_signature.startswith(service)]
        if len(cands) == 1:
            matched.add(_path_row_key(cands[0]))
        elif cands:
            bt = str((sample.get("synthesis_metadata") or {}).get("behavior_target") or "")
            for p in cands:
                if p.behavior_target == bt:
                    matched.add(_path_row_key(p))
                    break
            else:
                matched.add(_path_row_key(cands[0]))
    return matched

def _scheduled_path_key(sample: dict, paths: list) -> str | None:
    matched = match_sample_to_path(sample, paths)
    if matched:
        return _path_row_key(matched)
    meta = sample.get("synthesis_metadata") or {}
    loose = f"{meta.get('branch_id', '?')}::{meta.get('action_path_signature', '?')}"
    return f"UNMATCHED::{loose}"

def run_export(*, samples_path: Path, out_dir: Path) -> dict:
    samples = read_jsonl(samples_path)
    specs = {s["blueprint_id"]: s for s in build_all_behavior_specs()}
    paths_by_bp = {bp: enumerate_paths_from_spec(spec) for bp, spec in specs.items()}
    inv = build_blueprint_action_inventory()
    inv_rows = {
        bp["blueprint_id"]: bp["executable_actions"] for bp in inv["blueprints"]
    }

    path_counts: Counter[tuple[str, str]] = Counter()
    b0_path_counts: Counter[tuple[str, str]] = Counter()
    b0_service_counts: Counter[tuple[str, str]] = Counter()
    sample_level = Counter()

    for sample in samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if not bp:
            continue
        paths = paths_by_bp.get(bp) or []
        sched_key = _scheduled_path_key(sample, paths) or "UNMATCHED::?"
        path_counts[(bp, sched_key)] += 1

        b0 = sample.get("b0_output") or {}
        semantic = list(b0.get("semantic_actions") or [])
        b0_bucket, b0_detail = _b0_primary_label(sample)
        b0_service_counts[(bp, b0_bucket)] += 1

        b0_path_keys = _match_b0_to_path(sample, paths, semantic)
        if not b0_path_keys:
            b0_path_counts[(bp, "B0_NO_MATCHING_PATH")] += 1
        else:
            for pk in b0_path_keys:
                b0_path_counts[(bp, pk)] += 1

        if not semantic:
            sample_level["scheduled_but_b0_empty"] += 1
        elif sched_key.startswith("UNMATCHED"):
            sample_level["unmatched_schedule"] += 1
        elif b0_path_keys and sched_key in b0_path_keys:
            sample_level["path_and_b0_align"] += 1
        elif b0_path_keys:
            sample_level["path_b0_mismatch"] += 1
            sched_svc = sched_key.split("::")[1] if "::" in sched_key else "?"
            b0_svc = b0_bucket if not b0_bucket.startswith("B0_") else b0_detail
            if sched_svc == b0_svc:
                sample_level["same_service_diff_path"] += 1
            else:
                sample_level["diff_service"] += 1
        else:
            sample_level["b0_unmapped"] += 1

    rows: list[dict] = []
    all_bps = sorted(set(bp for bp, _ in path_counts) | set(bp for bp, _ in b0_path_counts))
    for bp in all_bps:
        scene = next((b["scene"] for b in inv["blueprints"] if b["blueprint_id"] == bp), "")
        path_keys = sorted(
            set(k for b, k in path_counts if b == bp)
            | set(k for b, k in b0_path_counts if b == bp)
            | {_inventory_path_key(a) for a in inv_rows.get(bp, [])}
        )
        for pk in path_keys:
            pc = path_counts.get((bp, pk), 0)
            bc = b0_path_counts.get((bp, pk), 0)
            if pk.startswith("UNMATCHED") or pk.startswith("B0_"):
                service = pk
            else:
                service = pk.split("::")[1] if "::" in pk else pk
            rows.append(
                {
                    "scene": scene,
                    "blueprint_id": bp,
                    "path_key": pk,
                    "key_action": service,
                    "scheduled_path_count": pc,
                    "b0_path_count": bc,
                    "delta_b0_minus_path": bc - pc,
                }
            )

    service_rows: list[dict] = []
    for bp in all_bps:
        scene = next((b["scene"] for b in inv["blueprints"] if b["blueprint_id"] == bp), "")
        sched_by_svc: Counter[str] = Counter()
        for (b, pk), n in path_counts.items():
            if b != bp:
                continue
            if pk.startswith("UNMATCHED"):
                sched_by_svc["UNMATCHED"] += n
            else:
                sched_by_svc[pk.split("::")[1]] += n
        b0_by_svc: Counter[str] = Counter()
        for (b, svc), n in b0_service_counts.items():
            if b == bp:
                b0_by_svc[svc] += n
        for svc in sorted(set(sched_by_svc) | set(b0_by_svc)):
            sc = sched_by_svc.get(svc, 0)
            bc = b0_by_svc.get(svc, 0)
            service_rows.append(
                {
                    "scene": scene,
                    "blueprint_id": bp,
                    "key_action": svc,
                    "scheduled_path_count": sc,
                    "b0_service_count": bc,
                    "delta_b0_minus_path": bc - sc,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_now(),
        "corpus": str(samples_path),
        "total_samples": len(samples),
        "sample_level_agreement": dict(sample_level),
        "summary": {
            "align_rate": round(sample_level["path_and_b0_align"] / len(samples), 4),
            "mismatch_rate": round(sample_level["path_b0_mismatch"] / len(samples), 4),
            "scheduled_but_b0_empty_rate": round(sample_level["scheduled_but_b0_empty"] / len(samples), 4),
        },
        "path_level_rows": rows,
        "service_level_rows": service_rows,
    }
    json_path = out_dir / "path_b0_distribution_diff.json"
    csv_path = out_dir / "path_b0_distribution_diff.csv"
    md_path = out_dir / "path_b0_distribution_diff.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "blueprint_id",
                "path_key",
                "key_action",
                "scheduled_path_count",
                "b0_path_count",
                "delta_b0_minus_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Path 调度分布 vs B0 执行分布 对比",
        "",
        f"- 生成时间: {payload['generated_at']}",
        f"- 语料: `{samples_path.name}`",
        f"- 总样本: {len(samples)}",
        "",
        "## 样本级一致率",
        "",
        f"| 指标 | 数量 | 占比 |",
        f"|------|-----:|-----:|",
    ]
    for k, v in sample_level.items():
        lines.append(f"| {k} | {v} | {v/len(samples):.1%} |")
    lines.extend(
        [
            "",
            "## Service 级差异（scheduled path service vs B0 primary service）",
            "",
            "| 场景 | Blueprint | 关键动作 | Path样本数 | B0样本数 | Δ(B0-Path) |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for r in sorted(service_rows, key=lambda x: (x["scene"], x["blueprint_id"], x["key_action"])):
        if r["scheduled_path_count"] == 0 and r["b0_service_count"] == 0:
            continue
        lines.append(
            f"| {r['scene']} | {r['blueprint_id']} | `{r['key_action']}` | "
            f"{r['scheduled_path_count']} | {r['b0_service_count']} | {r['delta_b0_minus_path']:+d} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"total_samples={len(samples)}")
    print(f"align={sample_level['path_and_b0_align']} mismatch={sample_level['path_b0_mismatch']} b0_empty={sample_level['scheduled_but_b0_empty']}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return payload

def main() -> int:
    parser = argparse.ArgumentParser(description="Path vs B0 distribution diff")
    parser.add_argument(
        "--samples",
        type=Path,
        default=SS_REPAIR_DIR / "single_scene_samples.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SS_REPAIR_DIR / "validation",
    )
    args = parser.parse_args()
    run_export(samples_path=args.samples, out_dir=args.out_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
