from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from smarthome_mdf.multi_action_frozen_v1.config import (
    EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    FROZEN_SAMPLES,
    MULTI_ACTION_FROZEN_SAMPLES,
    Y_PROMPT_VERSION,
)
from smarthome_mdf.multi_action_frozen_v1.frozen_loader import load_frozen_samples, load_multi_action_frozen
from smarthome_mdf.multi_action_frozen_v1.single_scene_y_labeling_view import (
    build_ma_y_labeling_view,
    build_multi_action_y_user_prompt,
)
from smarthome_mdf.multi_action_frozen_v1.y_incremental_batch import (
    ACTIVE_FAILURES_NAME,
    CHECKPOINT_NAME,
    load_completed_ids,
    run_incremental_y_labeling,
)
from smarthome_mdf.multi_action_frozen_v1.y_labeling_view import (
    FROZEN_INDEPENDENT_Y_SYSTEM_PROMPT,
    audit_y_information_flow,
    audit_y_view,
)
from smarthome_mdf.multi_action_frozen_v1.y_llm_config import y_llm_configured
from smarthome_mdf.multi_action_frozen_v1.y_ma_final_export import export_ma_final_actions_from_checkpoint
from smarthome_mdf.paths import MDF_ROOT

DEFAULT_OUT = MDF_ROOT / "runs" / "y_labeling_multi_action"
FINAL_ACTIONS_NAME = "ma_y_final_actions.jsonl"
ID_KEY = "multi_action_id"

def _load_id_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        data = json.loads(text)
        ids = data.get("sample_ids") or data.get("multi_action_ids") or []
        return [str(x) for x in ids]
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]

def _load_ss_index(ss_corpus: Path) -> dict[str, dict]:
    from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

    return {str(r["sample_id"]): r for r in read_jsonl(ss_corpus)}

def main() -> None:
    ap = argparse.ArgumentParser(description="Final multi-action Y labeling with checkpoint/resume")
    ap.add_argument("--ma-corpus", type=Path, default=MULTI_ACTION_FROZEN_SAMPLES)
    ap.add_argument(
        "--ss-corpus",
        type=Path,
        default=FROZEN_SAMPLES,
        help="Parent single-scene frozen corpus for runtime evidence lookup",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--llm", action="store_true", help="Run real LLM labeling")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--min-interval-sec", type=float, default=2.0)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4, help="Parallel LLM workers (1=legacy serial)")
    ap.add_argument("--target-rpm", type=float, default=60.0, help="Global API RPM cap when workers>1")
    ap.add_argument("--sample-ids-file", type=Path, default=None)
    ap.add_argument("--failures-only", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument(
        "--export-final-only",
        action="store_true",
        help=f"Export {FINAL_ACTIONS_NAME} from existing checkpoint and exit",
    )
    ap.add_argument(
        "--include-failed-in-export",
        action="store_true",
        help="Export all checkpoint rows, not only LLM_LABEL_SUCCESS",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / CHECKPOINT_NAME
    final_out = args.out_dir / FINAL_ACTIONS_NAME

    if args.export_final_only:
        summary = export_ma_final_actions_from_checkpoint(
            checkpoint,
            final_out,
            id_key=ID_KEY,
            success_only=not args.include_failed_in_export,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.ma_corpus == MULTI_ACTION_FROZEN_SAMPLES:
        corpus, meta = load_multi_action_frozen(verify=True)
        verify_sha = meta.get("multi_action_frozen_corpus_sha256")
    else:
        from smarthome_mdf.synthesis_v3.merge_scenes import read_jsonl

        corpus = read_jsonl(args.ma_corpus)
        verify_sha = None
        meta = {"path": str(args.ma_corpus), "sample_count": len(corpus)}

    ss_index = _load_ss_index(args.ss_corpus)
    index = {str(r.get(ID_KEY)): r for r in corpus}

    if args.failures_only:
        active_path = args.out_dir / ACTIVE_FAILURES_NAME
        if not active_path.is_file():
            raise SystemExit(f"--failures-only requires {active_path}")
        ids = _load_id_list(active_path)
        selected = [index[sid] for sid in ids if sid in index]
    elif args.sample_ids_file:
        ids = _load_id_list(args.sample_ids_file)
        selected = [index[sid] for sid in ids if sid in index]
    else:
        selected = corpus

    if args.limit:
        selected = selected[: args.limit]

    frozen_index_holder: dict[str, dict] = {"index": ss_index}

    def build_view(ma: dict) -> dict:
        return build_ma_y_labeling_view(ma, frozen_index_holder["index"])

    def build_prompt(view: dict, ma: dict | None = None) -> str:
        return build_multi_action_y_user_prompt(view, ma or {})

    if selected:
        probe_view = build_view(selected[0])
        flow = audit_y_information_flow(probe_view)
        view_audit = audit_y_view(probe_view)
        if flow["Y_INFORMATION_FLOW_VIOLATION"] or view_audit["Y_VIEW_INFORMATION_FLOW_VIOLATION"]:
            raise RuntimeError(f"Y information flow violation: flow={flow} view={view_audit}")

    completed_before = load_completed_ids(checkpoint, id_key=ID_KEY)
    entry = {
        "ma_corpus": str(args.ma_corpus),
        "ss_corpus": str(args.ss_corpus),
        "out_dir": str(args.out_dir),
        "corpus_count": len(corpus),
        "selected_count": len(selected),
        "multi_action_frozen_sha256": verify_sha or EXPECTED_MULTI_ACTION_FROZEN_SHA256,
        "prompt_version": Y_PROMPT_VERSION,
        "id_key": ID_KEY,
        "final_actions_file": FINAL_ACTIONS_NAME,
        "checkpoint_file": CHECKPOINT_NAME,
        "resume": not args.no_resume,
        "completed_before": len(completed_before),
        "llm_configured": y_llm_configured(),
        "workers": args.workers,
        "target_rpm": args.target_rpm,
    }
    (args.out_dir / "y_labeling_entry.json").write_text(
        json.dumps(entry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.dry_run or not args.llm:
        report = {
            "dry_run": True,
            "would_process": len(selected),
            "would_skip_completed": sum(
                1 for s in selected if str(s.get(ID_KEY)) in completed_before
            )
            if not args.no_resume
            else 0,
            **entry,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if not y_llm_configured():
        raise SystemExit(
            "LLM not configured — edit src/smarthome_mdf/multi_action_frozen_v1/y_llm_config.py (Y_LLM_API_KEY)"
        )

    summary = run_incremental_y_labeling(
        items=selected,
        out_dir=args.out_dir,
        build_view=build_view,
        build_prompt=build_prompt,
        system_prompt=FROZEN_INDEPENDENT_Y_SYSTEM_PROMPT,
        id_key=ID_KEY,
        resume=not args.no_resume,
        batch_size=args.batch_size,
        min_interval_sec=args.min_interval_sec,
        max_consecutive_failures=args.max_consecutive_failures,
        workers=args.workers,
        target_rpm=args.target_rpm,
    )

    export_summary = export_ma_final_actions_from_checkpoint(
        checkpoint,
        final_out,
        id_key=ID_KEY,
        success_only=True,
    )
    summary["final_actions_export"] = export_summary
    (args.out_dir / "y_labeling_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
