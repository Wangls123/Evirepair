from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "scripts" / "eval_llm_repair_baselines.py"
QO = ROOT / "baselines" / "qwen_open"
LOG = QO / "orchestrator_log.jsonl"

GENERATE_JOBS = [
    ("qwen3-14b", "direct", "ss", 12021),
    ("qwen3-14b", "constrained", "ss", 12021),
    ("qwen3-32b", "direct", "ss", 12021),
    ("qwen3-32b", "constrained", "ss", 12021),
]
EVAL_PHASES = ["evaluate", "runtime", "statistics", "timing", "build-tables", "report"]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def log(event: dict) -> None:
    QO.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_iso(), **event}, ensure_ascii=False) + "\n")
    print(json.dumps({"ts": now_iso(), **event}, ensure_ascii=False), flush=True)

def pred_path(model: str, baseline: str, dataset: str) -> Path:
    size = "14b" if "14b" in model else "32b"
    return QO / size / baseline / dataset / "predictions.jsonl"

TERMINAL_OK = {
    "VALID",
    "INVALID_JSON",
    "SCHEMA_VIOLATION",
    "MISSING_FIELD",
    "EMPTY_RESPONSE",
    "TRUNCATED_RESPONSE",
}

def count_status(path: Path) -> dict[str, int]:

    by_id: dict[str, str] = {}
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("id") or "")
            st = str(rec.get("parse_status") or "UNKNOWN")
            if not rid:
                continue
            prev = by_id.get(rid)
            if prev is None:
                by_id[rid] = st
                continue
            prev_ok = prev in TERMINAL_OK
            new_ok = st in TERMINAL_OK
            if new_ok and not prev_ok:
                by_id[rid] = st
            elif new_ok == prev_ok:
                by_id[rid] = st
    out: dict[str, int] = {}
    for st in by_id.values():
        out[st] = out.get(st, 0) + 1
    return out

def run_cmd(args: list[str]) -> int:
    print("\n>>> " + " ".join(args), flush=True)
    proc = subprocess.run(args, cwd=str(ROOT))
    return int(proc.returncode)

def require_key() -> None:
    if not str(os.environ.get("DASHSCOPE_API_KEY") or "").strip():
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set. Set it in this PowerShell session only. Do not write it to a file."
        )

def generate_one(model: str, baseline: str, dataset: str, expect_n: int, concurrency: int, rpm: float, keep_going: bool) -> None:
    cmd = [
        sys.executable,
        str(EVAL),
        "--phase", "generate",
        "--model", model,
        "--baseline", baseline,
        "--dataset", dataset,
        "--resume",
        "--concurrency", str(concurrency),
        "--target-rpm", str(rpm),
    ]
    code = run_cmd(cmd)
    counts = count_status(pred_path(model, baseline, dataset))
    n = sum(counts.values())
    valid = counts.get("VALID", 0)
    fail = counts.get("API_FAILURE", 0)
    log({
        "event": "generate_done",
        "model": model,
        "baseline": baseline,
        "dataset": dataset,
        "exit_code": code,
        "n": n,
        "expect_n": expect_n,
        "counts": counts,
    })
    if code != 0:
        raise SystemExit(f"generate failed exit={code} {model} {baseline} {dataset}")
    if n == 0:
        raise SystemExit(f"no predictions written for {model} {baseline} {dataset}")
    if valid == 0 and fail > 0:
        raise SystemExit(f"all API_FAILURE for {model} {baseline} {dataset}: {counts}")
    if fail and not keep_going:
        raise SystemExit(
            f"API_FAILURE={fail} in {model} {baseline} {dataset}. "
            "Re-run with --keep-going to continue, or resume after fixing the API."
        )
    if n < expect_n and not keep_going:
        raise SystemExit(
            f"incomplete {model} {baseline} {dataset}: {n}/{expect_n}. "
            "Re-run this orchestrator; generate uses --resume. Use --keep-going to score partial files."
        )

def main() -> None:
    ap = argparse.ArgumentParser(description="Run Qwen3-14B and Qwen3-32B baselines end to end")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--target-rpm", type=float, default=60.0)
    ap.add_argument("--generate-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="Continue after API_FAILURE or incomplete N")
    ap.add_argument(
        "--skip-32b-ma",
        action="store_true",
        help="Do not generate qwen3-32b MA jobs; score completed groups and leave 32B MA for later resume",
    )
    args = ap.parse_args()
    if args.generate_only and args.eval_only:
        raise SystemExit("choose at most one of --generate-only / --eval-only")
    if not args.eval_only:
        require_key()
    log({
        "event": "orchestrator_start",
        "jobs": [f"{m}/{b}/{d}" for m, b, d, _ in GENERATE_JOBS],
        "concurrency": args.concurrency,
        "target_rpm": args.target_rpm,
        "generate_only": args.generate_only,
        "eval_only": args.eval_only,
        "skip_32b_ma": args.skip_32b_ma,
    })
    if not args.eval_only:
        for model, baseline, dataset, expect_n in GENERATE_JOBS:
            if args.skip_32b_ma and "32b" in model and dataset == "ma":
                log({"event": "generate_skipped_deferred", "model": model, "baseline": baseline, "dataset": dataset})
                print(f"\n===== SKIP GENERATE {model} {baseline} {dataset} (deferred) =====", flush=True)
                continue
            print(f"\n===== GENERATE {model} {baseline} {dataset} =====", flush=True)
            generate_one(model, baseline, dataset, expect_n, args.concurrency, args.target_rpm, args.keep_going)
    if not args.generate_only:
        for phase in EVAL_PHASES:
            print(f"\n===== {phase.upper()} =====", flush=True)
            extra = ["--all"] if phase != "report" else []
            code = run_cmd([sys.executable, str(EVAL), "--phase", phase, *extra])
            log({"event": "eval_phase_done", "phase": phase, "exit_code": code})
            if code != 0:
                raise SystemExit(f"{phase} failed exit={code}")
    log({"event": "orchestrator_complete"})
    print("\nAll 14B/32B jobs finished. See baselines/qwen_open/qwen_baseline_final_report.md", flush=True)

if __name__ == "__main__":
    main()
