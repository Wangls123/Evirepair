from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from smarthome_mdf.multi_action_frozen_v1.action_type_classifier import CLASSIFIER_VERSION
from smarthome_mdf.multi_action_frozen_v1 import config as y_config
from smarthome_mdf.multi_action_frozen_v1 import y_incremental_batch as y_batch
from smarthome_mdf.multi_action_frozen_v1 import y_labeling_view as y_view
from smarthome_mdf.multi_action_frozen_v1.single_scene_y_labeling_view import (
    build_single_scene_y_labeling_view,
)
from smarthome_mdf.multi_action_frozen_v1.y_final_ss import (
    FINAL_EVENT_SYSTEM_PROMPT,
    FINAL_STATE_SYSTEM_PROMPT,
    Y_FINAL_PROMPT_VERSION,
    audit_y_final_payload,
    build_y_final_payload,
    build_y_final_user_prompt,
    select_system_prompt,
    y_final_prompt_hash,
)
from smarthome_mdf.multi_action_frozen_v1.y_incremental_batch import (
    CHECKPOINT_NAME,
    load_completed_ids,
    run_incremental_y_labeling,
)
from smarthome_mdf.multi_action_frozen_v1.y_labeling_view import label_one_independent_y
from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

DEFAULT_CORPUS = REPO / "data" / "benchmarks" / "single_scene.jsonl"
DEFAULT_OUT = REPO / "runs" / "y_labeling_single_scene"

DROP_ORACLE = {
    "formal_b0",
    "b0_output",
    "formal_y",
    "y_label_status",
    "y_output",
    "repaired_actions",
    "repair_result",
    "eval_y",
    "synthesis_construction_trace",
}


def _strip_oracle(sample: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in sample.items() if k not in DROP_ORACLE}


def build_view(sample: dict[str, Any]) -> dict[str, Any]:
    return build_single_scene_y_labeling_view(_strip_oracle(sample))


def build_prompt(view: dict[str, Any], sample: dict[str, Any] | None = None) -> str:
    return build_y_final_user_prompt(view, _strip_oracle(sample) if sample else None)


def label_final_y(view, *, system_prompt, user_prompt, **kwargs):
    action_type = "STATE_ACTION"
    declared = ""
    if "\nCLASSIFIER\n" in (user_prompt or ""):
        tail = user_prompt.split("\nCLASSIFIER\n", 1)[1]
        for line in tail.splitlines():
            if line.startswith("action_type="):
                action_type = line.split("=", 1)[1].strip()
            elif line.startswith("declared_effect_service="):
                declared = line.split("=", 1)[1].strip()
    chosen = select_system_prompt(action_type)
    rec = label_one_independent_y(
        view,
        system_prompt=chosen,
        user_prompt=user_prompt,
        **kwargs,
    )
    rec["action_type"] = action_type
    rec["declared_effect_service"] = declared
    rec["action_type_source"] = CLASSIFIER_VERSION
    rec["prompt_version"] = Y_FINAL_PROMPT_VERSION
    rec["prompt_hash"] = y_final_prompt_hash()
    return rec


def _stamp() -> None:
    y_config.Y_PROMPT_VERSION = Y_FINAL_PROMPT_VERSION
    y_view.Y_PROMPT_VERSION = Y_FINAL_PROMPT_VERSION
    y_batch.Y_PROMPT_VERSION = Y_FINAL_PROMPT_VERSION
    y_view.prompt_version_hash = y_final_prompt_hash
    y_batch.prompt_version_hash = y_final_prompt_hash


def _load_id_list(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Final single-scene Gold Y labeling (final_gold_y)")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Build the final labeling view and do not call a model")
    ap.add_argument("--llm", action="store_true", help="Run real LLM labeling")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--min-interval-sec", type=float, default=2.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-rpm", type=float, default=60.0)
    ap.add_argument("--sample-ids-file", type=Path, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.corpus)
    if args.sample_ids_file:
        wanted = _load_id_list(args.sample_ids_file)
        index = {str(r.get("sample_id")): r for r in rows}
        missing = [sid for sid in wanted if sid not in index]
        if missing:
            raise SystemExit(f"sample ids not in corpus: {missing[:5]}")
        selected = [index[sid] for sid in wanted]
    else:
        selected = rows
    if args.limit:
        selected = selected[: args.limit]

    entry = {
        "corpus": str(args.corpus),
        "out_dir": str(args.out_dir),
        "selected_count": len(selected),
        "prompt_version": Y_FINAL_PROMPT_VERSION,
        "prompt_hash": y_final_prompt_hash(),
        "classifier_version": CLASSIFIER_VERSION,
    }
    if selected:
        clean = _strip_oracle(selected[0])
        view = build_view(clean)
        payload = build_y_final_payload(clean, view)
        prompt = build_prompt(view, clean)
        audit = audit_y_final_payload(payload, FINAL_STATE_SYSTEM_PROMPT + FINAL_EVENT_SYSTEM_PROMPT + "\n" + prompt)
        entry["probe_sample_id"] = selected[0].get("sample_id")
        entry["probe_leakage"] = bool(audit.get("Y_LEAKAGE"))
        if audit.get("Y_LEAKAGE"):
            raise RuntimeError(f"Y leakage on probe: {audit}")

    if args.dry_run or not args.llm:
        print(json.dumps({"dry_run": True, **entry}, indent=2))
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed_before = load_completed_ids(args.out_dir / CHECKPOINT_NAME)
    entry["completed_before"] = len(completed_before)
    (args.out_dir / "y_labeling_entry.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _stamp()
    summary = run_incremental_y_labeling(
        items=selected,
        out_dir=args.out_dir,
        build_view=build_view,
        build_prompt=build_prompt,
        system_prompt=FINAL_STATE_SYSTEM_PROMPT,
        label_fn=label_final_y,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        min_interval_sec=args.min_interval_sec,
        max_consecutive_failures=args.max_consecutive_failures,
        workers=args.workers,
        target_rpm=args.target_rpm,
    )
    summary["prompt_version"] = Y_FINAL_PROMPT_VERSION
    summary["prompt_hash"] = y_final_prompt_hash()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
