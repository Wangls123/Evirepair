from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import Y_PROMPT_VERSION
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def checkpoint_row_to_final_action(rec: dict[str, Any]) -> dict[str, Any]:

    actions = list(rec.get("normalized_structured_y") or [])
    slim_actions: list[dict[str, Any]] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        row: dict[str, Any] = {
            "service": a.get("service"),
            "target_entity": a.get("target_entity"),
            "component_origin": a.get("component_origin"),
            "execution_order": a.get("execution_order"),
        }
        params = dict(a.get("parameters") or {})
        if params:
            row["parameters"] = params
        slim_actions.append(row)

    return {
        "multi_action_id": rec.get("multi_action_id"),
        "component_count": rec.get("component_count"),
        "label_decision": rec.get("label_decision"),
        "expected_actions": slim_actions,
    }

def export_ma_final_actions_from_checkpoint(
    checkpoint: Path,
    out_jsonl: Path,
    *,
    id_key: str = "multi_action_id",
    success_only: bool = True,
) -> dict[str, Any]:

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    written = 0
    skipped = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        with checkpoint.open(encoding="utf-8") as in_f:
            for line in in_f:
                text = line.strip()
                if not text:
                    continue
                try:
                    rec = json.loads(text)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if success_only and rec.get("label_status") != "LLM_LABEL_SUCCESS":
                    skipped += 1
                    continue
                if id_key not in rec and rec.get("sample_id"):
                    rec[id_key] = rec["sample_id"]
                final = checkpoint_row_to_final_action(rec)
                if not final.get(id_key):
                    skipped += 1
                    continue
                out_f.write(json.dumps(final, ensure_ascii=False) + "\n")
                written += 1

    summary = {
        "exported_at": _utc_now(),
        "checkpoint": str(checkpoint),
        "out_jsonl": str(out_jsonl),
        "written": written,
        "skipped": skipped,
        "success_only": success_only,
        "prompt_version": Y_PROMPT_VERSION,
    }
    (out_jsonl.parent / "ma_y_final_actions_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary

def load_final_actions_index(path: Path, id_key: str = "multi_action_id") -> dict[str, dict[str, Any]]:
    return {str(r[id_key]): r for r in read_jsonl(path) if r.get(id_key)}
