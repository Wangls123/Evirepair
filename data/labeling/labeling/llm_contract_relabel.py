from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from smarthome_mdf.labeling.blueprint_constraints import audit_sample_decision_output
from smarthome_mdf.labeling.llm_auto_label import (
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MAX_PER_MINUTE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REPETITION_PENALTY,
    LLM_SLEEP_SEC,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SEC,
    LLM_TOP_K,
    LLM_TOP_P,
    LLM_RETRIES,
    SCENE_FILES,
    LlmClient,
    _load_registry,
    label_one_sample,
    print_failed_sample_ids,
)
from smarthome_mdf.paths import LLM_LABEL_SCENES, QUALITY_REPORT_DIR, SAMPLES_DIR

def _resolve_scenes(scene: str, include_visual: bool) -> List[str]:
    if scene == "all" and include_visual:
        return list(SCENE_FILES.keys())
    if scene == "all":
        return list(LLM_LABEL_SCENES)
    return [scene]

def _load_scene_samples(samples_dir: Path, scene: str) -> List[dict]:
    path = samples_dir / SCENE_FILES[scene]
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def audit_scene(samples_dir: Path, scene: str) -> Dict[str, Any]:
    samples = _load_scene_samples(samples_dir, scene)
    records = [audit_sample_decision_output(s) for s in samples]
    by_status = Counter(r["status"] for r in records)
    out_of_contract = [r for r in records if r["status"] == "out_of_contract"]
    return {
        "scene": scene,
        "total": len(samples),
        "by_status": dict(by_status),
        "out_of_contract_count": len(out_of_contract),
        "out_of_contract_ids": [r["sample_id"] for r in out_of_contract],
        "records": records,
    }

def audit_scenes(
    samples_dir: Path,
    scenes: List[str],
    *,
    write_report: bool = True,
) -> dict:
    scene_reports: Dict[str, Any] = {}
    all_out_ids: List[str] = []
    total_by_status: Counter = Counter()
    violation_counts: Counter = Counter()
    blueprint_counts: Counter = Counter()

    for scene in scenes:
        rep = audit_scene(samples_dir, scene)
        scene_reports[scene] = {
            k: rep[k]
            for k in ("scene", "total", "by_status", "out_of_contract_count", "out_of_contract_ids")
        }
        all_out_ids.extend(rep["out_of_contract_ids"])
        total_by_status.update(rep["by_status"])
        for rec in rep["records"]:
            if rec["status"] != "out_of_contract":
                continue
            blueprint_counts[rec.get("blueprint_id") or "?"] += 1
            for v in rec.get("violations") or []:
                if v.startswith("not_in_allowed_actions:"):
                    violation_counts[v] += 1

    report = {
        "generated_at": date.today().isoformat(),
        "audit_rule": "allowed_actions whitelist (blueprint_action_contracts.json)",
        "scenes": scene_reports,
        "summary": {
            "total_samples": sum(r["total"] for r in scene_reports.values()),
            "by_status": dict(total_by_status),
            "out_of_contract_total": len(all_out_ids),
            "top_violations": dict(violation_counts.most_common(20)),
            "by_blueprint_id": dict(blueprint_counts.most_common(20)),
        },
        "out_of_contract_ids": sorted(set(all_out_ids)),
    }

    if write_report:
        QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        out_json = QUALITY_REPORT_DIR / "llm_contract_audit.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        out_md = QUALITY_REPORT_DIR / "llm_contract_audit.md"
        out_md.write_text(_format_audit_md(report), encoding="utf-8")
        report["report_json"] = str(out_json)
        report["report_md"] = str(out_md)

    return report

def _format_audit_md(report: dict) -> str:
    lines = [
        "# LLM 标签 · Action Contract 审计",
        "",
        f"> 生成日期：{report['generated_at']}",
        f"> 规则：`{report['audit_rule']}`",
        "",
        "## 汇总",
        "",
        f"- 总样本：**{report['summary']['total_samples']}**",
        f"- 越界需重打：**{report['summary']['out_of_contract_total']}**",
        "",
        "| status | 条数 |",
        "|--------|-----:|",
    ]
    for st, n in sorted(report["summary"]["by_status"].items()):
        lines.append(f"| {st} | {n} |")
    lines.extend(["", "## 按 scene", ""])
    for scene, rep in report["scenes"].items():
        lines.append(f"### {scene} (n={rep['total']}, 越界={rep['out_of_contract_count']})")
        for st, n in sorted(rep["by_status"].items()):
            lines.append(f"- {st}: {n}")
        lines.append("")
    if report["summary"]["top_violations"]:
        lines.extend(["## 常见违规", ""])
        for v, n in report["summary"]["top_violations"].items():
            lines.append(f"- `{v}`: **{n}**")
        lines.append("")
    if report["summary"]["by_blueprint_id"]:
        lines.extend(["## 按 blueprint_id（越界）", ""])
        for bid, n in report["summary"]["by_blueprint_id"].items():
            lines.append(f"- `{bid}`: **{n}**")
        lines.append("")
    lines.extend(
        [
            "## 重打越界样本",
            "",
            "```powershell",
            "cd <this repository>",
            "$env:PYTHONPATH = \"$PWD\\src\"",
            "python -m smarthome_mdf.labeling.llm_contract_relabel --relabel --scene all --include-visual",
            "```",
        ]
    )
    return "\n".join(lines)

def _write_jsonl(path: Path, samples: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def relabel_out_of_contract(
    samples_dir: Path,
    scenes: List[str],
    client: LlmClient,
    *,
    limit: Optional[int] = None,
    sample_ids: Optional[Set[str]] = None,
    in_place: bool = True,
) -> dict:

    registry = _load_registry()
    audit = audit_scenes(samples_dir, scenes, write_report=True)
    relabel_ids: Set[str] = set(audit["out_of_contract_ids"])
    if sample_ids:
        relabel_ids &= sample_ids

    logs: Dict[str, Any] = {
        "mode": "relabel_out_of_contract",
        "audit_summary": audit["summary"],
        "scenes": {},
        "failed_sample_ids": [],
        "checkpoint": "after_each_successful_relabel",
    }
    relabeled_total = 0

    for scene in scenes:
        in_path = samples_dir / SCENE_FILES[scene]
        out_path = in_path if in_place else samples_dir / SCENE_FILES[scene].replace(
            ".jsonl", "_relabel.jsonl"
        )
        samples = _load_scene_samples(samples_dir, scene)
        scene_log = {
            "total": len(samples),
            "target_relabel": sum(1 for s in samples if s.get("sample_id") in relabel_ids),
            "relabeled": 0,
            "skipped_ok": 0,
            "skipped_not_target": 0,
            "errors": [],
        }

        for i, sample in enumerate(samples):
            sid = sample.get("sample_id")
            rec = audit_sample_decision_output(sample)

            if sid not in relabel_ids:
                scene_log["skipped_not_target"] += 1
                continue

            if limit is not None and relabeled_total >= limit:
                continue

            print(
                f"  [{scene} {i+1}/{len(samples)}] {sid} relabel "
                f"(violations={rec.get('violations')})",
                flush=True,
            )
            try:
                samples[i] = label_one_sample(sample, client, registry)
                scene_log["relabeled"] += 1
                relabeled_total += 1
                _write_jsonl(out_path, samples)
                print(f"  [{scene}] {sid} OK (checkpoint saved)", flush=True)
            except Exception as e:
                scene_log["errors"].append({"sample_id": sid, "error": str(e)})
                logs["failed_sample_ids"].append(sid)
                print(f"  [{scene}] {sid} ERROR: {e}", flush=True)

        scene_log["output"] = str(out_path)
        logs["scenes"][scene] = scene_log

    logs["relabeled_total"] = relabeled_total
    print_failed_sample_ids(logs["failed_sample_ids"], title="Relabel failed sample_ids")

    QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    relog = QUALITY_REPORT_DIR / "llm_contract_relabel_log.json"
    relog.write_text(json.dumps(logs, indent=2, ensure_ascii=False), encoding="utf-8")
    logs["report"] = str(relog)
    return logs

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit LLM labels vs Action Contract; relabel only out-of-contract samples"
    )
    parser.add_argument("--audit-only", action="store_true", help="Scan only, no LLM calls")
    parser.add_argument("--relabel", action="store_true", help="Relabel out-of-contract samples via LLM")
    parser.add_argument("--scene", default="all", choices=list(SCENE_FILES.keys()) + ["all"])
    parser.add_argument("--include-visual", action="store_true")
    parser.add_argument("--samples-dir", default=str(SAMPLES_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Max samples to relabel (global)")
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="Optional: restrict relabel to these sample_ids (intersect with out-of-contract)",
    )
    parser.add_argument("--no-in-place", action="store_true")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--max-per-minute", type=int, default=LLM_MAX_PER_MINUTE)
    parser.add_argument("--sleep", type=float, default=LLM_SLEEP_SEC)
    args = parser.parse_args()

    if not args.audit_only and not args.relabel:
        parser.error("Specify --audit-only and/or --relabel")

    scenes = _resolve_scenes(args.scene, args.include_visual)
    samples_dir = Path(args.samples_dir)

    if args.audit_only or not args.relabel:
        report = audit_scenes(samples_dir, scenes, write_report=True)
        print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
        print(f"\nWrote {report.get('report_json')} , {report.get('report_md')}")
        print(f"Out-of-contract: {report['summary']['out_of_contract_total']} samples")

    if args.relabel:
        if not LLM_API_URL.strip() or not LLM_API_KEY.strip():
            parser.error("Set LLM_API_URL / LLM_API_KEY (or V4_LLM_API_KEY / DASHSCOPE_API_KEY) via environment")
        sample_ids: Optional[Set[str]] = None
        if args.sample_ids_file:
            text = args.sample_ids_file.read_text(encoding="utf-8").strip()
            if text.startswith("["):
                sample_ids = set(json.loads(text))
            else:
                sample_ids = {ln.strip() for ln in text.splitlines() if ln.strip()}

        client = LlmClient(
            LLM_API_KEY,
            model=args.model,
            api_url=LLM_API_URL,
            max_per_minute=args.max_per_minute,
            sleep_sec=args.sleep,
            timeout_sec=LLM_TIMEOUT_SEC,
            retries=LLM_RETRIES,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=LLM_TOP_P,
            top_k=LLM_TOP_K,
            repetition_penalty=LLM_REPETITION_PENALTY,
        )
        logs = relabel_out_of_contract(
            samples_dir,
            scenes,
            client,
            limit=args.limit,
            sample_ids=sample_ids,
            in_place=not args.no_in_place,
        )
        print(json.dumps(logs, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
