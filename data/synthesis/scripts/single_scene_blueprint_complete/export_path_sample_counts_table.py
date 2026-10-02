from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    build_blueprint_action_inventory,
    enumerate_paths_from_spec,
    match_sample_to_path,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs
from smarthome_mdf.single_scene_blueprint_complete.config import OUTPUT_DIR
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _key_action(action: dict) -> str:
    return str(action.get("service") or "valid.no_action")

def _unmatched_key_action(meta: dict) -> str:
    sig = str(meta.get("action_path_signature") or "")
    if sig and "|" not in sig:
        return sig
    parts = sig.split("|")
    return parts[0] if parts else "unknown"

def _write_markdown_table(rows: list[dict], path: Path, *, generated_at: str, total_samples: int) -> None:
    lines = [
        "# 单场景合成语料 Path 样本分布表",
        "",
        f"- 生成时间: {generated_at}",
        f"- 语料: `frozen_v1_ss_repair/single_scene_samples.jsonl`",
        f"- 总样本数: {total_samples}",
        f"- 总行数 (path): {len(rows)}",
        "",
        "| 场景 | Blueprint | 关键动作 | 样本数 |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['blueprint_id']} | `{row['key_action']}` | {row['sample_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> int:
    samples_path = OUTPUT_DIR / "frozen_v1_ss_repair" / "single_scene_samples.jsonl"
    samples = read_jsonl(samples_path)
    specs = build_all_behavior_specs()
    spec_by_bp = {s["blueprint_id"]: s for s in specs}
    paths_by_bp = {bp: enumerate_paths_from_spec(spec_by_bp[bp]) for bp in spec_by_bp}

    counts: Counter[tuple[str, str]] = Counter()
    unmatched_by_bp: Counter[str] = Counter()
    for sample in samples:
        bp = str((sample.get("blueprint_binding") or {}).get("blueprint_id") or "")
        if not bp:
            continue
        paths = paths_by_bp.get(bp) or []
        matched = match_sample_to_path(sample, paths)
        if matched:
            counts[(bp, matched.path_key)] += 1
        else:
            meta = sample.get("synthesis_metadata") or {}
            loose = f"{meta.get('branch_id', '?')}::{meta.get('action_path_signature', '?')}"
            counts[(bp, f"UNMATCHED::{loose}")] += 1
            unmatched_by_bp[bp] += 1

    inv = build_blueprint_action_inventory()
    rows: list[dict] = []
    for bp_entry in inv["blueprints"]:
        bp = bp_entry["blueprint_id"]
        scene = bp_entry["scene"]
        seen_keys: set[str] = set()
        for action in bp_entry["executable_actions"]:
            pk = f"{action['branch_id']}::{action['action_path_signature']}::{action['action_template_key']}"
            seen_keys.add(pk)
            rows.append(
                {
                    "scene": scene,
                    "blueprint_id": bp,
                    "key_action": _key_action(action),
                    "branch_id": action["branch_id"],
                    "sample_count": counts.get((bp, pk), 0),
                }
            )
        for (b, pk), n in counts.items():
            if b == bp and pk.startswith("UNMATCHED::") and pk not in seen_keys:
                meta_part = pk.replace("UNMATCHED::", "")
                sig = meta_part.split("::", 1)[-1] if "::" in meta_part else meta_part
                rows.append(
                    {
                        "scene": scene,
                        "blueprint_id": bp,
                        "key_action": sig.split("|")[0] if sig else "unknown",
                        "branch_id": meta_part.split("::")[0] if "::" in meta_part else "?",
                        "sample_count": n,
                    }
                )

    rows.sort(key=lambda r: (r["scene"], r["blueprint_id"], r["key_action"], r.get("branch_id", "")))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    validation = OUTPUT_DIR / "frozen_v1_ss_repair" / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    json_path = validation / "path_sample_counts_table.json"
    csv_path = validation / "path_sample_counts_table.csv"
    md_path = validation / "path_sample_counts_table.md"

    payload = {
        "generated_at": generated_at,
        "total_samples": len(samples),
        "row_count": len(rows),
        "unmatched_by_blueprint": dict(unmatched_by_bp),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene", "blueprint_id", "key_action", "sample_count"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    _write_markdown_table(rows, md_path, generated_at=generated_at, total_samples=len(samples))

    print(f"total_samples={len(samples)} rows={len(rows)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")
    print(f"md={md_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
