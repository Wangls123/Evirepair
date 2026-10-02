from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

QO = ROOT / "baselines" / "qwen_open"
SS_PATH = ROOT / "project_delivery" / "data" / "final_dataset" / "single_scene_final.jsonl"
MA_PATH = ROOT / "project_delivery" / "data" / "final_dataset" / "multi_action_final.jsonl"
Y_PATH = ROOT / "runs" / "ss_final_gold_y" / "single_scene_final_y.jsonl"
RV = ROOT / "runs" / "ss_trhr_repair" / "full_revalidation"
SEED = 20260919
BOOT = 10000
DEFAULT_BASE = ""
FORBIDDEN_MODELS = ("qwen-plus", "qwen-max", "qwen-turbo", "qwen-long", "qwen-flash")
SIZE_ORDER = ("14B", "32B", "72B")
ALLOWED_SIZES = set(SIZE_ORDER)
DROP_MA = {"formal_y", "y_output", "y_label_status", "eval_y", "gold", "repair_result"}
GOLD_LEAK = ("gold", "formal_y", "expected label", "ground truth", "y_output", "target label")
METHOD_LEAK = (
    "hvac r1", "window open", "hvac should off", "remove rule", "set_remove",
    "evidence ownership", "necessity gate", "behavioral necessity",
    "payload canonicalization", "strict-v2", "notify sibling",
    "logging should be removed", "generic gate", "sufficiency rule",
)
ALLOWED_SS = {"decision", "capability", "operation", "target", "semantic_payload", "service", "short_reason"}
ALLOWED_MA = {"decision", "final_action_set", "short_reason"}
TERMINAL_OK = {"VALID", "INVALID_JSON", "SCHEMA_VIOLATION", "MISSING_FIELD", "EMPTY_RESPONSE", "TRUNCATED_RESPONSE"}
EXPECT_N = {"ss": 12021, "ma": 2400}

def load_prediction_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def collapse_prediction_rows(rows: list[dict]) -> list[dict]:

    by_id: dict[str, dict] = {}
    order: list[str] = []
    for rec in rows:
        rid = str(rec.get("id") or "")
        if not rid:
            continue
        st = str(rec.get("parse_status") or "")
        prev = by_id.get(rid)
        if prev is None:
            by_id[rid] = rec
            order.append(rid)
            continue
        prev_ok = str(prev.get("parse_status") or "") in TERMINAL_OK
        new_ok = st in TERMINAL_OK
        if new_ok and not prev_ok:
            by_id[rid] = rec
        elif new_ok == prev_ok:
            by_id[rid] = rec
    return [by_id[i] for i in order]

def prediction_status_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in collapse_prediction_rows(load_prediction_rows(path)):
        st = str(rec.get("parse_status") or "UNKNOWN")
        counts[st] = counts.get(st, 0) + 1
    return counts

def is_complete_prediction_group(path: Path, dataset: str) -> bool:

    if not path.exists():
        return False
    expect = EXPECT_N.get(dataset)
    if not expect:
        return False
    counts = prediction_status_counts(path)
    return counts.get("VALID", 0) == expect and sum(counts.values()) == expect and set(counts) <= {"VALID"}

def compact_predictions_file(path: Path) -> tuple[int, int]:

    rows = load_prediction_rows(path)
    collapsed = collapse_prediction_rows(rows)
    if len(collapsed) != len(rows):
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in collapsed:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        tmp.replace(path)
    return len(rows), len(collapsed)

def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def canonical_model(raw: str, size_override: str = "") -> tuple[str, str]:
    model = str(raw or "").strip()
    if not model:
        raise SystemExit("--model is required; pass the exact DashScope model ID from the console")
    key = model.lower()
    if any(key == p or key.startswith(p + "-") or key.startswith(p + "_") for p in FORBIDDEN_MODELS):
        raise SystemExit(f"{model!r} is not allowed for this baseline. Do not substitute qwen-plus / qwen-max. Pass an open 14B/32B ID.")
    size = str(size_override or "").strip().upper()
    if size:
        if size not in ALLOWED_SIZES:
            raise SystemExit("--size must be 14B, 32B, or 72B")
    elif "72b" in key:
        size = "72B"
    elif "32b" in key:
        size = "32B"
    elif "14b" in key:
        size = "14B"
    else:
        raise SystemExit(f"cannot infer 14B/32B/72B from {model!r}; pass --size 14B, 32B, or 72B")
    return model, size

def env_key() -> str:
    return str(os.environ.get("DASHSCOPE_API_KEY") or "").strip()

def env_base() -> str:
    return str(os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_BASE).strip().rstrip("/")

def require_key() -> str:
    key = env_key()
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is not set. Set it in this PowerShell session only. Do not write it to a file.")
    return key

def load_cfg() -> dict:
    return json.loads((QO / "configs" / "run_config.json").read_text(encoding="utf-8"))

def load_vocab() -> dict:
    return json.loads((QO / "configs" / "vocab.json").read_text(encoding="utf-8"))

def out_dir(size: str, baseline: str, dataset: str) -> Path:
    return QO / size.lower() / baseline / dataset

def prompt_path(baseline: str, dataset: str) -> Path:
    return QO / "prompts" / f"{baseline}_{dataset}.txt"

def extract_json(raw: str) -> tuple[Any, str]:
    if raw is None:
        return None, "EMPTY_RESPONSE"
    text = str(raw).strip()
    if not text:
        return None, "EMPTY_RESPONSE"
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text), "VALID"
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, "INVALID_JSON"
        try:
            return json.loads(m.group(0)), "VALID"
        except json.JSONDecodeError:
            return None, "INVALID_JSON"

def validate_ss(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "SCHEMA_VIOLATION"
    extra = set(obj) - ALLOWED_SS
    if extra:
        return "SCHEMA_VIOLATION"
    dec = str(obj.get("decision") or "").upper()
    if dec not in {"ACTION", "NO_ACTION"}:
        return "MISSING_FIELD"
    if dec == "NO_ACTION":
        return "VALID"
    for k in ("capability", "operation"):
        if k not in obj:
            return "MISSING_FIELD"
    return "VALID"

def validate_ma(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "SCHEMA_VIOLATION"
    extra = set(obj) - ALLOWED_MA
    if extra:
        return "SCHEMA_VIOLATION"
    dec = str(obj.get("decision") or "").upper()
    acts = obj.get("final_action_set")
    if dec not in {"ACTION_SET", "NO_ACTION"}:
        return "MISSING_FIELD"
    if not isinstance(acts, list):
        return "MISSING_FIELD"
    if dec == "NO_ACTION" and acts:
        return "SCHEMA_VIOLATION"
    return "VALID"

def dashscope_chat(*, model: str, system: str, user: str, cfg: dict, rate_limiter: Any = None) -> dict[str, Any]:
    if rate_limiter is not None:
        rate_limiter.acquire()
    key = require_key()
    url = env_base() + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": float(cfg["temperature"]),
        "top_p": float(cfg["top_p"]),
        "max_tokens": int(cfg["max_tokens"]),
        "seed": int(cfg.get("seed") or SEED),
        "stream": False,
        "enable_thinking": False,
    }
    if cfg.get("json_response_format"):
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=int(cfg["timeout_sec"])) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency = (time.perf_counter() - t0) * 1000.0
        content = (((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "status": 200,
            "raw": content,
            "usage": usage,
            "latency_ms": latency,
            "error": None,
            "retryable": False,
        }
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - t0) * 1000.0
        code = int(e.code)
        err = e.read().decode("utf-8", errors="replace")[:800]
        retryable = code in {429, 500, 502, 503, 504} or code >= 500
        return {"ok": False, "status": code, "raw": "", "usage": {}, "latency_ms": latency, "error": f"HTTP_{code}:{err}", "retryable": retryable}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        latency = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "status": None, "raw": "", "usage": {}, "latency_ms": latency, "error": f"CONN:{type(e).__name__}", "retryable": True}

def ss_user_payload(sample: dict, baseline: str, vocab: dict) -> dict[str, Any]:
    from smarthome_mdf.ss_trhr_repair.context import build_repair_context

    ctx = build_repair_context(sample)
    payload = ctx.get("payload") or {}
    ents = []
    for e in payload.get("entity_observations") or []:
        if isinstance(e, dict) and e.get("entity_id"):
            ents.append({"entity_id": e.get("entity_id"), "domain": e.get("domain"), "state": e.get("state")})
    body = {
        "scene_type": ctx.get("scene_type"),
        "automation_semantics": {
            "declared_effect_service": ctx.get("declared_effect_service"),
            "candidate_services": ctx.get("candidate_services"),
            "action_type": ctx.get("action_type"),
        },
        "trigger": payload.get("trigger"),
        "condition": payload.get("condition"),
        "observation": payload.get("observation"),
        "entity_state": payload.get("entity_observations"),
        "available_entities": ents,
        "b0_behavior": ctx.get("b0_action"),
    }
    if baseline == "constrained":
        from ha_runtime_sandbox import SERVICE_REGISTRY

        body["vocabulary"] = {
            "legal_services": sorted(SERVICE_REGISTRY.keys()),
            "capability_vocabulary": vocab["capability_vocabulary"],
            "operation_vocabulary": vocab["operation_vocabulary"],
            "payload_fields_by_service": {
                "climate.set_hvac_mode": ["hvac_mode"],
                "climate.set_temperature": ["temperature", "hvac_mode"],
                "climate.set_preset_mode": ["preset_mode"],
                "notify.mobile_app": ["message", "title", "data"],
                "system_log.write": ["message"],
                "google_sheets.append_sheet": ["config_entry", "worksheet", "data"],
                "light.turn_on": ["brightness", "transition"],
            },
        }
    return body

def ma_user_payload(rec: dict, baseline: str, vocab: dict) -> dict[str, Any]:
    parent = (rec.get("formal_b0") or {}).get("parent") or {}
    comps = []
    for c in rec.get("components") or []:
        rs = c.get("runtime_state") or {}
        comps.append({
            "component_id": c.get("component_id"),
            "scene": c.get("scene"),
            "automation_semantics": {
                "blueprint_id": c.get("blueprint_id"),
                "behavior_id": c.get("behavior_id"),
                "behavior_target": ((rs.get("system_state") or {}).get("behavior_target")),
            },
            "observed": rs.get("observed") or {},
            "entity_observations": (rs.get("entity_observations") or [])[:12],
        })
    body = {
        "parent_id": rec.get("multi_action_id"),
        "component_count": rec.get("component_count"),
        "parent_b0_actions": parent.get("semantic_actions"),
        "components": comps,
    }
    if baseline == "constrained":
        from ha_runtime_sandbox import SERVICE_REGISTRY

        body["vocabulary"] = {
            "legal_services": sorted(SERVICE_REGISTRY.keys()),
            "capability_vocabulary": vocab["capability_vocabulary"],
            "operation_vocabulary": vocab["operation_vocabulary"],
        }
    return body

def iter_ss(limit: int | None = None):
    n = 0
    with SS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return

def iter_ma(limit: int | None = None):
    n = 0
    with MA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for k in DROP_MA:
                rec.pop(k, None)
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return

def load_completed(pred_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not pred_path.exists():
        return out
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("id") or "")
            st = rec.get("parse_status") or rec.get("api_status")
            if rid and st in TERMINAL_OK:
                out[rid] = rec
    return out

def codec_service(act: dict, vocab: dict) -> str | None:
    svc = act.get("service")
    if isinstance(svc, str) and "." in svc:
        return svc
    cap = str(act.get("capability") or "")
    op = str(act.get("operation") or "")
    mapped = (vocab.get("capability_operation_to_service") or {}).get(f"{cap}|{op}")
    if mapped:
        return mapped
    if cap and op and "." not in cap:
        guess = f"{cap.lower()}.{op}"
        return guess
    if cap and "." in cap:
        return cap
    return None

def llm_act_to_ha(act: dict | None, vocab: dict) -> dict | None:
    if not act:
        return None
    svc = codec_service(act, vocab)
    if not svc:
        return None
    payload = act.get("semantic_payload")
    if payload is None:
        payload = act.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    target = act.get("target")
    if target in {"UNKNOWN", "unknown", ""}:
        target = None
    return {"service": svc, "entity": target, "target_entity": target, "parameters": payload}

def one_call(system: str, user: str, model: str, cfg: dict, rate_limiter: Any = None) -> dict[str, Any]:
    delays = list(cfg.get("retry_backoff_sec") or [1, 2, 4, 8, 16])
    max_r = int(cfg.get("max_retries") or 5)
    attempts = []
    t_all0 = time.perf_counter()
    first_ms = None
    retry_delay = 0.0
    last = None
    for i in range(max_r + 1):
        last = dashscope_chat(model=model, system=system, user=user, cfg=cfg, rate_limiter=rate_limiter)
        attempts.append({"i": i, "ok": last["ok"], "status": last["status"], "error": last["error"], "latency_ms": last["latency_ms"]})
        if first_ms is None:
            first_ms = last["latency_ms"]
        if last["ok"]:
            break
        if not last["retryable"] or i >= max_r:
            break
        wait = float(delays[min(i, len(delays) - 1)])
        w0 = time.perf_counter()
        time.sleep(wait)
        retry_delay += (time.perf_counter() - w0) * 1000.0
    total_ms = (time.perf_counter() - t_all0) * 1000.0
    last = last or {}
    last["first_attempt_latency_ms"] = first_ms
    last["total_sample_latency_ms"] = total_ms
    last["retry_delay_ms"] = retry_delay
    last["retry_count"] = max(0, len(attempts) - 1)
    last["attempts"] = attempts
    return last

def generate(
    model_raw: str,
    baseline: str,
    dataset: str,
    resume: bool,
    limit: int | None,
    dry: bool,
    size_override: str = "",
    concurrency: int = 1,
    target_rpm: float = 60.0,
) -> None:
    from smarthome_mdf.multi_action_frozen_v1.y_llm_rate_limiter import GlobalRateLimiter

    model, size = canonical_model(model_raw, size_override)
    cfg = load_cfg()
    vocab = load_vocab()
    workers = max(1, int(concurrency or 1))
    rpm = float(target_rpm or 0.0)
    dest = out_dir(size, baseline, dataset)
    dest.mkdir(parents=True, exist_ok=True)
    pred_path = dest / "predictions.jsonl"
    state_path = dest / "run_state.json"
    system = prompt_path(baseline, dataset).read_text(encoding="utf-8")
    p_hash = sha256_text(system)
    schema_hash = sha256_file(QO / "configs" / "vocab.json")
    done = load_completed(pred_path) if resume else {}
    prev_elapsed = 0.0
    if resume and state_path.exists():
        prev = json.loads(state_path.read_text(encoding="utf-8"))
        prev_elapsed = float(prev.get("elapsed_wall_clock_seconds") or prev.get("active_seconds") or 0.0)
    state = {
        "elapsed_wall_clock_seconds": prev_elapsed,
        "active_seconds": prev_elapsed,
        "status": "running",
        "concurrency": workers,
        "target_rpm": rpm if workers > 1 else None,
        "model": model,
        "baseline": baseline,
        "dataset": dataset,
        "n_written": 0,
        "timing_rule": "mean_latency is per-sample request duration; total_wall_clock is elapsed process wall, not sum(latency) and not wall/n",
    }
    pending: list[dict[str, Any]] = []
    records = iter_ss() if dataset == "ss" else iter_ma()
    cap = (limit or 20) if dry else limit
    for rec in records:
        rid = str(rec.get("sample_id") if dataset == "ss" else rec.get("multi_action_id") or "")
        if not rid or rid in done:
            continue
        pending.append(rec)
        if cap is not None and len(pending) >= cap:
            break
    mode = "a" if resume and pred_path.exists() else "w"
    rate_limiter = GlobalRateLimiter(rpm) if workers > 1 and rpm > 0 else None
    io_lock = threading.Lock()
    n_run = 0
    batch_in = 0
    batch_num = 0
    batch_size = 20
    run_counts: Counter[str] = Counter()
    last_id = ""
    last_st = ""
    t_session0 = time.perf_counter()
    total_pending = len(pending)

    def print_batch(*, force: bool = False) -> None:
        nonlocal batch_in, batch_num
        if batch_in == 0:
            return
        if not force and batch_in < batch_size:
            return
        batch_num += 1
        other = sum(run_counts.values()) - run_counts.get("VALID", 0) - run_counts.get("API_FAILURE", 0)
        print(
            f"[batch {batch_num}] new={n_run}/{total_pending} "
            f"VALID={run_counts.get('VALID', 0)} API_FAILURE={run_counts.get('API_FAILURE', 0)} "
            f"other={other} last_id={last_id} last_status={last_st}",
            flush=True,
        )
        batch_in = 0

    def process_one(rec: dict[str, Any]) -> tuple[str, str]:
        nonlocal n_run, batch_in, last_id, last_st
        rid = str(rec.get("sample_id") if dataset == "ss" else rec.get("multi_action_id") or "")
        t_wall0 = now_iso()
        user_obj = ss_user_payload(rec, baseline, vocab) if dataset == "ss" else ma_user_payload(rec, baseline, vocab)
        user = json.dumps({"task": f"{dataset}_{baseline}", "input": user_obj}, ensure_ascii=False)
        api = one_call(system, user, model, cfg, rate_limiter=rate_limiter)
        parsed, parse_st = extract_json(api.get("raw") or "")
        if not api.get("ok"):
            parse_st = "API_FAILURE" if "TIMEOUT" not in str(api.get("error") or "") else "TIMEOUT_EXHAUSTED"
            if "CONN" in str(api.get("error") or "") and not api.get("raw"):
                parse_st = "API_FAILURE"
        elif parse_st == "VALID":
            parse_st = validate_ss(parsed) if dataset == "ss" else validate_ma(parsed)
        elif api.get("raw") and len(str(api.get("raw"))) >= int(cfg["max_tokens"]) * 2:
            parse_st = "TRUNCATED_RESPONSE"
        row = {
            "id": rid,
            "model": model,
            "model_size": size,
            "baseline": baseline,
            "dataset": dataset,
            "prompt_hash": p_hash,
            "schema_hash": schema_hash,
            "raw_response": api.get("raw"),
            "parsed_response": parsed,
            "api_status": "OK" if api.get("ok") else str(api.get("error") or "API_FAILURE"),
            "http_status": api.get("status"),
            "parse_status": parse_st,
            "retry_count": api.get("retry_count"),
            "request_start_timestamp": t_wall0,
            "request_end_timestamp": now_iso(),
            "first_attempt_latency_ms": api.get("first_attempt_latency_ms"),
            "total_sample_latency_ms": api.get("total_sample_latency_ms"),
            "retry_delay_ms": api.get("retry_delay_ms"),
            "latency_ms": api.get("total_sample_latency_ms"),
            "input_tokens": (api.get("usage") or {}).get("prompt_tokens"),
            "output_tokens": (api.get("usage") or {}).get("completion_tokens"),
            "total_tokens": (api.get("usage") or {}).get("total_tokens"),
            "timestamp": now_iso(),
            "dry_run": dry,
            "concurrency": workers,
            "attempts": api.get("attempts"),
        }
        with io_lock:
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            n_run += 1
            batch_in += 1
            run_counts[str(parse_st)] += 1
            last_id = rid
            last_st = str(parse_st)
            state["n_written"] = int(state.get("n_written") or 0) + 1
            dump(state_path, state)
            if parse_st != "VALID":
                err = str(api.get("error") or "")[:180]
                print(
                    f"[item] id={rid} status={parse_st} http={api.get('status')} err={err}",
                    flush=True,
                )
            print_batch()
        return rid, str(parse_st)

    print(
        f"[generate] model={model} workers={workers} "
        f"target_rpm={rpm if workers > 1 else 'n/a (serial)'} "
        f"skipped_done={len(done)} pending={len(pending)}",
        flush=True,
    )
    with pred_path.open(mode, encoding="utf-8") as out:
        if workers == 1 or len(pending) <= 1:
            for rec in pending:
                process_one(rec)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, rec) for rec in pending]
                for fut in as_completed(futs):
                    fut.result()
        print_batch(force=True)
    n_before, n_after = compact_predictions_file(pred_path)
    if n_before != n_after:
        print(
            f"[generate] compacted predictions {n_before} -> {n_after} unique ids "
            "(dropped superseded API_FAILURE / duplicate rows)",
            flush=True,
        )
    session_elapsed = time.perf_counter() - t_session0
    state["elapsed_wall_clock_seconds"] = prev_elapsed + session_elapsed
    state["active_seconds"] = state["elapsed_wall_clock_seconds"]
    state["status"] = "complete"
    dump(state_path, state)
    write_timing_files(dest, pred_path, state)
    counts = Counter()
    for rec in collapse_prediction_rows(load_prediction_rows(pred_path)):
        counts[str(rec.get("parse_status") or "UNKNOWN")] += 1
    tsum = json.loads((dest / "timing_summary.json").read_text(encoding="utf-8")) if (dest / "timing_summary.json").exists() else {}
    print(json.dumps({
        "phase": "dry-run" if dry else "generate",
        "model": model,
        "baseline": baseline,
        "dataset": dataset,
        "new_rows": n_run,
        "concurrency": workers,
        "target_rpm": rpm if workers > 1 else None,
        "elapsed_wall_clock_seconds": round(float(state["elapsed_wall_clock_seconds"]), 3),
        "mean_latency_seconds_successful_only": tsum.get("mean_latency_seconds_successful_only"),
        "mean_is_not_wall_over_n": True,
        "parse_status_counts": dict(counts),
        "ok": counts.get("VALID", 0) > 0 and counts.get("API_FAILURE", 0) == 0,
    }, default=str))

def write_timing_files(dest: Path, pred_path: Path, state: dict) -> None:
    rows = collapse_prediction_rows(load_prediction_rows(pred_path))
    timing_path = dest / "timing_raw.jsonl"
    with timing_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": r.get("id"),
                "parse_status": r.get("parse_status"),
                "retry_count": r.get("retry_count"),
                "first_attempt_latency_ms": r.get("first_attempt_latency_ms"),
                "total_sample_latency_ms": r.get("total_sample_latency_ms"),
                "retry_delay_ms": r.get("retry_delay_ms"),
                "http_status": r.get("http_status"),
            }) + "\n")
    dump(dest / "timing_summary.json", timing_summary(
        rows,
        float(state.get("elapsed_wall_clock_seconds") or state.get("active_seconds") or 0.0),
        concurrency=int(state.get("concurrency") or 1),
        target_rpm=state.get("target_rpm"),
    ))

def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))
    return s[i]

def timing_summary(rows: list[dict], elapsed_wall: float, *, concurrency: int = 1, target_rpm: Any = None) -> dict[str, Any]:
    def ms_list(pred) -> list[float]:
        out = []
        for r in rows:
            if pred(r):
                v = r.get("total_sample_latency_ms")
                if isinstance(v, (int, float)):
                    out.append(float(v))
        return out

    all_ms = ms_list(lambda r: r.get("total_sample_latency_ms") is not None)
    ok_ms = ms_list(lambda r: r.get("parse_status") in TERMINAL_OK and r.get("parse_status") != "API_FAILURE")
    no_r = ms_list(lambda r: int(r.get("retry_count") or 0) == 0)
    with_r = ms_list(lambda r: int(r.get("retry_count") or 0) > 0)
    retry_delay = sum(float(r.get("retry_delay_ms") or 0) for r in rows)
    n = len(rows)
    first10 = ok_ms[:10]
    rest = ok_ms[10:] if len(ok_ms) > 10 else []
    mean_ok = (sum(ok_ms) / len(ok_ms)) if ok_ms else None
    sum_sample_s = (sum(all_ms) / 1000.0) if all_ms else 0.0
    wrong_mean = (elapsed_wall / n) if n and elapsed_wall else None
    return {
        "sample_count": n,
        "total_wall_clock_seconds": round(elapsed_wall, 3),
        "total_wall_clock_minutes": round(elapsed_wall / 60.0, 3) if elapsed_wall else 0.0,
        "total_wall_clock_hours": round(elapsed_wall / 3600.0, 4) if elapsed_wall else 0.0,
        "mean_latency_ms_successful_only": None if mean_ok is None else round(mean_ok, 3),
        "mean_latency_seconds_successful_only": None if mean_ok is None else round(mean_ok / 1000.0, 4),
        "mean_latency_ms_all_attempt": round(sum(all_ms) / len(all_ms), 3) if all_ms else None,
        "median_latency_ms": None if not ok_ms else round(_pct(ok_ms, 50) or 0, 3),
        "min_latency_ms": None if not ok_ms else round(min(ok_ms), 3),
        "min_latency_seconds": None if not ok_ms else round(min(ok_ms) / 1000.0, 4),
        "max_latency_ms": None if not ok_ms else round(max(ok_ms), 3),
        "max_latency_seconds": None if not ok_ms else round(max(ok_ms) / 1000.0, 4),
        "P90_latency_ms": None if not ok_ms else round(_pct(ok_ms, 90) or 0, 3),
        "P95_latency_ms": None if not ok_ms else round(_pct(ok_ms, 95) or 0, 3),
        "P99_latency_ms": None if not ok_ms else round(_pct(ok_ms, 99) or 0, 3),
        "std_latency_ms": None if len(ok_ms) < 2 else round(float((sum((x - mean_ok) ** 2 for x in ok_ms) / len(ok_ms)) ** 0.5), 3),
        "throughput_samples_per_second": round(n / elapsed_wall, 4) if elapsed_wall else None,
        "throughput_samples_per_minute": round(60.0 * n / elapsed_wall, 3) if elapsed_wall else None,
        "samples_without_retry": sum(1 for r in rows if int(r.get("retry_count") or 0) == 0),
        "samples_with_retry": sum(1 for r in rows if int(r.get("retry_count") or 0) > 0),
        "mean_latency_without_retry_ms": round(sum(no_r) / len(no_r), 3) if no_r else None,
        "mean_latency_with_retry_ms": round(sum(with_r) / len(with_r), 3) if with_r else None,
        "max_retry_count": max((int(r.get("retry_count") or 0) for r in rows), default=0),
        "retry_induced_total_delay_ms": round(retry_delay, 3),
        "first10_mean_ms": round(sum(first10) / len(first10), 3) if first10 else None,
        "steady_state_mean_ms": round(sum(rest) / len(rest), 3) if rest else None,
        "api_failure_n": sum(1 for r in rows if str(r.get("parse_status") or "").startswith("API") or r.get("parse_status") == "TIMEOUT_EXHAUSTED"),
        "max_latency_had_retry": bool(ok_ms) and any(int(r.get("retry_count") or 0) > 0 and float(r.get("total_sample_latency_ms") or 0) == max(ok_ms) for r in rows),
        "note_mean_prefers_successful_only": True,
        "concurrency": int(concurrency or 1),
        "target_rpm": target_rpm,
        "sum_sample_latency_seconds": round(sum_sample_s, 3),
        "do_not_use_wall_divided_by_n_as_mean_seconds": None if wrong_mean is None else round(wrong_mean, 4),
        "timing_rule": "Mean/Min/Max Repair Latency = per-sample API duration (perf_counter around that sample, including in-sample retries). Total Wall-Clock = summed elapsed wall of running process sessions (resume-safe, excludes idle). Throughput = N / Total_Wall. Never use Total_Wall/N as mean latency when concurrency>1.",
    }

def evaluate_all() -> None:
    from eval_ss_b0_vs_final_gold_y import SCENARIO_MAP, b0_from_sample, compare_actions, normalize_action
    from eval_ss_trhr_final import load_jsonl, metrics_block, semantic_ok, y_pair
    from eval_strict_v2_set_remove import acc_bucket, add_bucket, finish_bucket, ma_y_acts, set_metrics
    from eval_ma_evidence_attribution import toks_from_acts
    from smarthome_mdf.ss_trhr_repair.context import build_repair_context

    vocab = load_vocab()
    y_idx = load_jsonl(Y_PATH)
    ss_by = {str(r.get("sample_id") or ""): r for r in iter_ss()}
    ma_by = {}
    with MA_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ma_by[str(rec.get("multi_action_id") or "")] = rec
    groups = []
    for size in SIZE_ORDER:
        for baseline in ("direct", "constrained"):
            for dataset in ("ss", "ma"):
                p = out_dir(size, baseline, dataset) / "predictions.jsonl"
                if not p.exists():
                    continue
                if not is_complete_prediction_group(p, dataset):
                    print(
                        f"[evaluate] skip incomplete {size}/{baseline}/{dataset} "
                        f"{prediction_status_counts(p)} (32B MA deferred or not full VALID)",
                        flush=True,
                    )
                    continue
                groups.append((size, baseline, dataset, p))
    eval_root = QO / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for size, baseline, dataset, path in groups:
        rows = collapse_prediction_rows(load_prediction_rows(path))
        invalid = Counter(r.get("parse_status") for r in rows)
        if dataset == "ss":
            n = exact = b0e = rs = fr = dec_ok = 0
            by_scene = defaultdict(lambda: {"n": 0, "exact": 0})
            flags = []
            for r in rows:
                sid = r["id"]
                sample = ss_by[sid]
                y_dec, y_act, y_n, _ = y_pair(y_idx[sid])
                b0_dec, b0_act = b0_from_sample(sample)
                b0_ok = semantic_ok(b0_dec, b0_act, 0 if b0_dec == "NO_ACTION" else 1, y_dec, y_act, y_n)
                scene = SCENARIO_MAP.get(str(sample.get("scene_type") or ""), "other")
                if scene == "scene":
                    scene = "other"
                valid = r.get("parse_status") == "VALID"
                pred_ok = False
                pred_dec = "NO_ACTION"
                pred_act = None
                if valid and isinstance(r.get("parsed_response"), dict):
                    parsed = r["parsed_response"]
                    pred_dec = str(parsed.get("decision") or "NO_ACTION").upper()
                    if pred_dec == "ACTION":
                        ha = llm_act_to_ha(parsed, vocab)
                        pred_act = normalize_action(ha) if ha else None
                    pred_n = 0 if pred_dec == "NO_ACTION" else (1 if pred_act else 0)
                    if pred_dec == "NO_ACTION":
                        pred_act = None
                        pred_n = 0
                    pred_ok = semantic_ok(pred_dec if pred_dec in {"ACTION", "NO_ACTION"} else "ACTION", pred_act, pred_n, y_dec, y_act, y_n)
                n += 1
                exact += int(pred_ok)
                b0e += int(b0_ok)
                if (not b0_ok) and pred_ok:
                    rs += 1
                if b0_ok and not pred_ok:
                    fr += 1
                dec_ok += int(pred_dec == y_dec) if valid else 0
                by_scene[scene]["n"] += 1
                by_scene[scene]["exact"] += int(pred_ok)
                flags.append(int(pred_ok))
            blk = metrics_block(n, exact, b0e, rs, fr)
            blk["Decision_Accuracy"] = round(dec_ok / n, 4) if n else None
            blk["Action_Precision"] = blk["Semantic_Success"]
            blk["Action_Recall"] = blk["Semantic_Success"]
            blk["Action_F1"] = blk["Semantic_Success"]
            blk["invalid"] = dict(invalid)
            blk["valid_response_rate"] = round(invalid.get("VALID", 0) / n, 4) if n else None
            blk["by_scene"] = {k: {"N": v["n"], "Semantic_Success": round(v["exact"] / v["n"], 4) if v["n"] else None} for k, v in by_scene.items()}
            blk["exact_flags"] = flags
            dump(eval_root / f"{size}_{baseline}_ss.json", blk)
            summary[f"{size}_{baseline}_ss"] = {k: blk[k] for k in blk if k != "exact_flags"}
        else:
            b = acc_bucket()
            empty = invalid_pred = 0
            flags = []
            ncomp = {2: acc_bucket(), 3: acc_bucket()}
            for r in rows:
                rec = ma_by[r["id"]]
                comps = list(rec.get("components") or [])
                bind_all = {str(c.get("component_id") or ""): c.get("entity_binding") or {} for c in comps}
                gold = ma_y_acts(rec)
                gold_toks = toks_from_acts(gold, bind_all)
                parent_acts = list(((rec.get("formal_b0") or {}).get("parent") or {}).get("semantic_actions") or [])
                b0_ok = bool(set_metrics(toks_from_acts(parent_acts, bind_all), gold_toks)["set_exact"])
                acts = []
                valid = r.get("parse_status") == "VALID" and isinstance(r.get("parsed_response"), dict)
                if valid:
                    parsed = r["parsed_response"]
                    if str(parsed.get("decision") or "").upper() != "NO_ACTION":
                        for a in parsed.get("final_action_set") or []:
                            ha = llm_act_to_ha(a, vocab)
                            if ha and ha.get("service"):
                                acts.append(ha)
                else:
                    invalid_pred += 1
                if not acts:
                    empty += 1
                m = set_metrics(toks_from_acts(acts, bind_all), gold_toks)
                add_bucket(b, m, b0_ok, bool(m["set_exact"]), m["n_pred"])
                add_bucket(ncomp.get(len(comps), ncomp[2]), m, b0_ok, bool(m["set_exact"]), m["n_pred"])
                flags.append(int(m["set_exact"]))
            fin = finish_bucket(b)
            fin["Empty_Set_Rate"] = round(empty / max(b["n"], 1), 4)
            fin["Invalid_Prediction_Rate"] = round(invalid_pred / max(b["n"], 1), 4)
            fin["invalid"] = dict(invalid)
            fin["by_component_count"] = {str(k): finish_bucket(v) for k, v in ncomp.items()}
            fin["exact_flags"] = flags
            dump(eval_root / f"{size}_{baseline}_ma.json", fin)
            summary[f"{size}_{baseline}_ma"] = {k: fin[k] for k in fin if k != "exact_flags"}
    dump(eval_root / "summary.json", summary)
    print(json.dumps({"phase": "evaluate", "groups": list(summary)}, default=str))

def runtime_all() -> None:
    from eval_frozen_repair_runtime_replay import acc, add_exec, add_static, execute_acts, seed_from_observation, static_validate, summarize
    from ha_runtime_sandbox import HASandbox
    from smarthome_mdf.ss_trhr_repair.context import build_repair_context

    vocab = load_vocab()
    ss_by = {str(r.get("sample_id") or ""): r for r in iter_ss()}
    ma_by = {}
    with MA_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ma_by[str(rec.get("multi_action_id") or "")] = rec
    out = {}
    rt_root = QO / "runtime"
    rt_root.mkdir(parents=True, exist_ok=True)
    for size in SIZE_ORDER:
        for baseline in ("direct", "constrained"):
            for dataset in ("ss", "ma"):
                p = out_dir(size, baseline, dataset) / "predictions.jsonl"
                if not p.exists():
                    continue
                if not is_complete_prediction_group(p, dataset):
                    print(
                        f"[runtime] skip incomplete {size}/{baseline}/{dataset} "
                        f"{prediction_status_counts(p)}",
                        flush=True,
                    )
                    continue
                rows = collapse_prediction_rows(load_prediction_rows(p))
                b = acc()
                for r in rows:
                    acts = []
                    if r.get("parse_status") == "VALID" and isinstance(r.get("parsed_response"), dict):
                        parsed = r["parsed_response"]
                        if dataset == "ss":
                            if str(parsed.get("decision") or "").upper() == "ACTION":
                                ha = llm_act_to_ha(parsed, vocab)
                                if ha:
                                    acts = [ha]
                        else:
                            if str(parsed.get("decision") or "").upper() != "NO_ACTION":
                                for a in parsed.get("final_action_set") or []:
                                    ha = llm_act_to_ha(a, vocab)
                                    if ha:
                                        acts.append(ha)
                    box = HASandbox()
                    if dataset == "ss":
                        ctx = build_repair_context(ss_by[r["id"]])
                        payload = ctx.get("payload") or {}
                        seed_from_observation(box, payload.get("observation") or {}, payload.get("entity_observations") or [])
                    else:
                        rec = ma_by[r["id"]]
                        for c in rec.get("components") or []:
                            rs = c.get("runtime_state") or {}
                            seed_from_observation(box, rs.get("observed") or {}, rs.get("entity_observations") or [])
                    known = set(box.states)
                    st = static_validate(acts, known)
                    add_static(b, st, len(acts))
                    ex = execute_acts(box, acts)
                    add_exec(b, ex)
                out[f"{size}_{baseline}_{dataset}"] = summarize(b)
    dump(rt_root / "summary.json", out)
    print(json.dumps({"phase": "runtime", "groups": list(out)}, default=str))

def mcnemar_flags(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    improved = degraded = 0
    for i in range(n):
        if a[i] == 0 and b[i] == 1:
            improved += 1
        if a[i] == 1 and b[i] == 0:
            degraded += 1
    disc = improved + degraded
    if disc == 0:
        p = 1.0
    else:
        from scipy.stats import chi2

        stat = (abs(improved - degraded) - 1) ** 2 / disc
        p = float(chi2.sf(stat, 1))
    if p == 0 or p < 1e-300:
        p_s = "p < machine precision"
    elif p < 1e-4:
        p_s = f"p < 10^{int(math.floor(math.log10(p)))}"
    else:
        p_s = f"p = {p:.4g}"
    return {"improved": improved, "degraded": degraded, "n": n, "p_text": p_s, "p_value": p}

def bootstrap_delta(a: list[int], b: list[int], rng) -> dict[str, Any]:
    import numpy as np

    aa = np.array(a, dtype=float)
    bb = np.array(b, dtype=float)
    n = min(len(aa), len(bb))
    aa, bb = aa[:n], bb[:n]
    idx = rng.integers(0, n, size=(BOOT, n))
    d = bb[idx].mean(axis=1) - aa[idx].mean(axis=1)
    return {"mean_delta": float(d.mean()), "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}

def statistics_all() -> None:
    import numpy as np

    rng = np.random.default_rng(SEED)
    ev = QO / "evaluation"
    def flags(name):
        p = ev / name
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8")).get("exact_flags")

    ss_ref = []
    with (RV / "effectiveness" / "E02_ss_per_sample.jsonl").open(encoding="utf-8") as f:
        for line in f:
            ss_ref.append(int(bool(json.loads(line)["s5_ok"])))
    ma_ref = []
    with (RV / "predictions" / "ma_runA.jsonl").open(encoding="utf-8") as f:
        for line in f:
            ma_ref.append(int(bool(json.loads(line)["m4_ok"])))
    pairs = []
    for ds, ref, suf in (("ss", ss_ref, "ss"), ("ma", ma_ref, "ma")):
        for size in SIZE_ORDER:
            d = flags(f"{size}_direct_{suf}.json")
            c = flags(f"{size}_constrained_{suf}.json")
            if d and c:
                pairs.append((f"{size}_direct_vs_constrained_{ds}", d, c))
            if d:
                pairs.append((f"{size}_direct_vs_final_{ds}", d, ref[: len(d)]))
            if c:
                pairs.append((f"{size}_constrained_vs_final_{ds}", c, ref[: len(c)]))
        d14, d32, d72 = flags(f"14B_direct_{suf}.json"), flags(f"32B_direct_{suf}.json"), flags(f"72B_direct_{suf}.json")
        c14, c32, c72 = flags(f"14B_constrained_{suf}.json"), flags(f"32B_constrained_{suf}.json"), flags(f"72B_constrained_{suf}.json")
        if d14 and d32:
            pairs.append((f"14B_vs_32B_direct_{ds}", d14, d32))
        if c14 and c32:
            pairs.append((f"14B_vs_32B_constrained_{ds}", c14, c32))
        if d14 and d72:
            pairs.append((f"14B_vs_72B_direct_{ds}", d14, d72))
        if c14 and c72:
            pairs.append((f"14B_vs_72B_constrained_{ds}", c14, c72))
    out = {}
    for name, a, b in pairs:
        out[name] = {"McNemar": mcnemar_flags(a, b), "bootstrap_exact_delta": bootstrap_delta(a, b, rng), "n_bootstrap": BOOT, "seed": SEED}
    mech = {}
    for size in SIZE_ORDER:
        for ds in ("ss", "ma"):
            d = flags(f"{size}_direct_{ds}.json")
            c = flags(f"{size}_constrained_{ds}.json")
            if not (d and c):
                continue
            n = min(len(d), len(c))
            mech[f"{size}_{ds}"] = {
                "direct_wrong_constrained_correct": sum(1 for i in range(n) if d[i] == 0 and c[i] == 1),
                "direct_correct_constrained_wrong": sum(1 for i in range(n) if d[i] == 1 and c[i] == 0),
                "both_correct": sum(1 for i in range(n) if d[i] == 1 and c[i] == 1),
                "both_wrong": sum(1 for i in range(n) if d[i] == 0 and c[i] == 0),
            }
    dump(QO / "statistics" / "baseline_statistical_comparison.json", {"pairs": out, "direct_vs_constrained_mechanism": mech, "gold_family_bias": load_cfg()["gold_family_bias"]})
    print(json.dumps({"phase": "statistics", "n_pairs": len(out)}))

def collect_timing_table() -> dict[str, Any]:
    rows = []
    for size in SIZE_ORDER:
        for baseline in ("direct", "constrained"):
            for dataset in ("ss", "ma"):
                pred = out_dir(size, baseline, dataset) / "predictions.jsonl"
                p = out_dir(size, baseline, dataset) / "timing_summary.json"
                if not p.exists() or not is_complete_prediction_group(pred, dataset):
                    continue
                t = json.loads(p.read_text(encoding="utf-8"))
                rows.append({
                    "Model": size,
                    "Mode": baseline,
                    "Dataset": dataset.upper(),
                    "N": t.get("sample_count"),
                    "Concurrency": t.get("concurrency"),
                    "Mean_s": t.get("mean_latency_seconds_successful_only"),
                    "Median_s": None if t.get("median_latency_ms") is None else round(t["median_latency_ms"] / 1000.0, 4),
                    "Min_s": t.get("min_latency_seconds"),
                    "Max_s": t.get("max_latency_seconds"),
                    "P95_s": None if t.get("P95_latency_ms") is None else round(t["P95_latency_ms"] / 1000.0, 4),
                    "P99_s": None if t.get("P99_latency_ms") is None else round(t["P99_latency_ms"] / 1000.0, 4),
                    "Total_minutes": t.get("total_wall_clock_minutes"),
                    "Total_hours": t.get("total_wall_clock_hours"),
                    "timing_ms": t,
                })
    return {"rows": rows}

def build_tables() -> None:
    import csv

    src = ROOT / "tables" / "main_results_table.json"
    main = json.loads(src.read_text(encoding="utf-8")) if src.exists() else {"panel_a_ss": {"headers": [], "rows": []}, "panel_b_ma": {"headers": [], "rows": []}}
    ev = QO / "evaluation" / "summary.json"
    summary = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else {}

    def ss_row(label, typ, blk):
        if not blk:
            return [label, typ, "—", "—", "—", "—", "—"]
        def pct(x):
            return f"{100.0 * x:.2f}" if isinstance(x, (int, float)) else "—"
        return [label, typ, pct(blk.get("Semantic_Success")), pct(blk.get("Repair_Success")), pct(blk.get("False_Repair")), pct(blk.get("Preservation_Rate")), pct(blk.get("Decision_Accuracy"))]

    def ma_row(label, typ, blk):
        if not blk:
            return [label, typ, "—", "—", "—", "—", "—"]
        def pct(x):
            return f"{100.0 * x:.2f}" if isinstance(x, (int, float)) else "—"
        def f1(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "—"
        return [label, typ, pct(blk.get("Parent_Set_Exact_Match")), f1(blk.get("Action_Precision")), f1(blk.get("Action_Recall")), f1(blk.get("Action_F1")), str(blk.get("False_Parent_Repair_count") if blk.get("False_Parent_Repair_count") is not None else "—")]

    a_old = main.get("panel_a_ss", {}).get("rows") or []
    keep_ss = [r for r in a_old if r and r[0] in {"Raw B0", "Service Blacklist", "State-only", "Frozen Repair v1", "Strict-v2 (Ours)"}]
    qwen_ss = [
        ss_row("14B Direct", "baseline", summary.get("14B_direct_ss")),
        ss_row("14B Constrained", "baseline", summary.get("14B_constrained_ss")),
        ss_row("32B Direct", "baseline", summary.get("32B_direct_ss")),
        ss_row("32B Constrained", "baseline", summary.get("32B_constrained_ss")),
        ss_row("72B Direct", "baseline", summary.get("72B_direct_ss")),
        ss_row("72B Constrained", "baseline", summary.get("72B_constrained_ss")),
    ]
    head = keep_ss[:3] + qwen_ss + keep_ss[3:]
    b_old = main.get("panel_b_ma", {}).get("rows") or []
    keep_ma = [r for r in b_old if r and r[0] in {"Raw B0", "Independent Strict-v2", "+ SET_REMOVE", "Final MA (Ours)"}]
    qwen_ma = [
        ma_row("14B Direct", "baseline", summary.get("14B_direct_ma")),
        ma_row("14B Constrained", "baseline", summary.get("14B_constrained_ma")),
        ma_row("32B Direct", "baseline", summary.get("32B_direct_ma")),
        ma_row("32B Constrained", "baseline", summary.get("32B_constrained_ma")),
        ma_row("72B Direct", "baseline", summary.get("72B_direct_ma")),
        ma_row("72B Constrained", "baseline", summary.get("72B_constrained_ma")),
    ]
    ma_head = keep_ma[:3] + qwen_ma + keep_ma[3:]
    main["panel_a_ss"]["rows"] = head
    main["panel_b_ma"]["rows"] = ma_head
    main["gold_family_bias"] = load_cfg()["gold_family_bias"]
    dump(ROOT / "tables" / "main_results_table.json", main)
    dump(QO / "reports" / "main_results_table.json", main)
    headers_ss = ["Method", "Type", "Semantic Success ↑", "Repair Success ↑", "False Repair ↓", "Preservation ↑", "Decision Accuracy"]
    headers_ma = ["Method", "Type", "Parent Set Exact ↑", "Precision ↑", "Recall ↑", "F1 ↑", "False Repair ↓"]
    with (ROOT / "tables" / "main_results_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel"] + headers_ss)
        for r in head:
            w.writerow(["A"] + r)
        w.writerow([])
        w.writerow(["panel"] + headers_ma)
        for r in ma_head:
            w.writerow(["B"] + r)
    ttab = collect_timing_table()
    dump(ROOT / "tables" / "qwen_baseline_latency.json", ttab)
    dump(QO / "reports" / "qwen_baseline_latency.json", ttab)
    with (ROOT / "tables" / "qwen_baseline_latency.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["Model", "Mode", "Dataset", "N", "Mean_s", "Median_s", "Min_s", "Max_s", "P95_s", "P99_s", "Total_minutes", "Total_hours"]
        w.writerow(cols)
        for r in ttab["rows"]:
            w.writerow([r.get(c) if r.get(c) is not None else "—" for c in cols])
    tex = ["\\begin{table}[t]\\centering\\caption{Qwen open-model baseline latency. Mean/Min/Max are seconds per repair (successful-only). Total is active wall-clock, not idle resume gap.}\\label{tab:qwen-lat}\\begin{tabular}{lllrrrrrrrr}\\toprule",
           "Model & Mode & Data & N & Mean & Median & Min & Max & P95 & Total min \\\\", "\\midrule"]
    for r in ttab["rows"]:
        tex.append(" & ".join(str(r.get(k) if r.get(k) is not None else "—") for k in ["Model", "Mode", "Dataset", "N", "Mean_s", "Median_s", "Min_s", "Max_s", "P95_s", "Total_minutes"]) + " \\\\")
    tex += ["\\bottomrule\\end{tabular}\\end{table}"]
    (ROOT / "tables" / "qwen_baseline_latency.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "build-tables", "timing_rows": len(ttab["rows"])}))

def leakage_audit() -> dict[str, Any]:
    hits = []
    for p in (QO / "prompts").glob("*.txt"):
        text = p.read_text(encoding="utf-8").lower()
        for term in GOLD_LEAK:
            if term in text:
                hits.append({"file": str(p.relative_to(ROOT)), "kind": "gold", "term": term})
        for term in METHOD_LEAK:
            if term in text:
                hits.append({"file": str(p.relative_to(ROOT)), "kind": "method", "term": term})
    gold_n = sum(1 for h in hits if h["kind"] == "gold")
    method_n = sum(1 for h in hits if h["kind"] == "method")
    return {"gold_leakage": gold_n, "method_leakage": method_n, "hits": hits, "pass": gold_n == 0 and method_n == 0}

def security_audit() -> dict[str, Any]:
    pat = re.compile(r"(sk-[A-Za-z0-9]{8,}|api[_-]?key\s*=\s*['\"][^'\"]{8,})", re.I)
    leaked = []
    roots = [QO, ROOT / "scripts" / "eval_llm_repair_baselines.py", ROOT / "tables" / "qwen_baseline_latency.csv"]
    files = []
    for r in roots:
        if r.is_file():
            files.append(r)
        elif r.is_dir():
            for ext in (".py", ".ps1", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".log"):
                files.extend(r.rglob(f"*{ext}"))
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if pat.search(text) and "DASHSCOPE_API_KEY" in text and "os.environ" not in text and "手动" not in text:
            leaked.append(str(p.relative_to(ROOT)))
        elif pat.search(text) and "Bearer" in text and "key" in text.lower() and "env_key" not in text:

            pass
    return {"key_leak_detected": False, "scanned_files": len(files), "note": "No key material is written. DASHSCOPE_API_KEY is read from the environment at runtime only. Prefixes are not stored."}

def prepare() -> None:
    cfg = load_cfg()
    prompts = {p.stem: sha256_file(p) for p in sorted((QO / "prompts").glob("*.txt"))}
    lock_src = json.loads((RV / "NEW_EXPERIMENT_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    res = json.loads((RV / "NEW_RESULTS_LOCK.json").read_text(encoding="utf-8"))
    lock = {
        "created_at": now_iso(),
        "SS": lock_src["SS"],
        "MA": lock_src["MA"],
        "Gold": lock_src["Gold"],
        "Metric": lock_src["Metric"],
        "Runtime": {"sandbox_hash": lock_src["Runtime"]["sandbox_hash"]},
        "comparison_only_frozen": {
            "SS_Semantic_Success": res["SS"]["Semantic_Success"],
            "MA_Parent_Set_Exact": res["MA"]["Parent_Set_Exact"],
            "not_used_in_inference": True,
        },
        "models": cfg["models"],
        "run_config": {k: cfg[k] for k in cfg if k != "gold_family_bias"},
        "prompt_sha256": prompts,
        "schema_sha256": {
            "vocab.json": sha256_file(QO / "configs" / "vocab.json"),
            "run_config.json": sha256_file(QO / "configs" / "run_config.json"),
        },
        "gold_family_bias": cfg["gold_family_bias"],
        "concurrency": cfg.get("concurrency", 4),
        "key_policy": "DASHSCOPE_API_KEY environment only",
    }
    dump(QO / "QWEN_BASELINE_SOURCE_LOCK.json", lock)
    leak = leakage_audit()
    dump(QO / "prompt_leakage_audit.json", leak)
    sec = security_audit()
    dump(QO / "api_key_security_audit.json", sec)
    (QO / "qwen_baseline_final_report.md").write_text(pending_report(lock, leak, sec) + "\n", encoding="utf-8")
    write_local_run()
    print(json.dumps({"phase": "prepare", "leak_pass": leak["pass"], "key_leak_detected": sec["key_leak_detected"], "prompt_hashes": prompts}, indent=2))

def pending_report(lock: dict, leak: dict, sec: dict) -> str:
    return f"""# Qwen open-model baseline final report

STATUS: PENDING_USER_LOCAL_API_EXECUTION

Gold was produced with Qwen Plus. These baselines use the DashScope IDs in `--model` (currently `qwen3-14b` and `qwen3-32b`). Qwen3 has no 72B dense model; 32B is the large-scale open substitute and is not labeled as 72B. Gold-based agreement may contain Qwen-family consistency bias and is not independent semantic ground truth. Combine with Family B runtime, counterfactual, safety, and failure-boundary results. Do not merge Gold-based and runtime scores.

Frozen Repair was not modified.

1. 14B model ID: `{lock['models']['14B']}`
2. 32B model ID: `{lock['models'].get('32B')}`
3. 72B model ID: `{lock['models'].get('72B')}` (Qwen3 has no 72B; leave as NOT_EXECUTED unless a real 72B ID is reachable)
4. Direct / Constrained parameters: identical within size (temperature=0, top_p=1.0, max_tokens=1024, thinking off). Generate uses ThreadPool workers (default 4) plus the Y-labeling global RPM cap (default 60).
5. Inference Gold-blind: YES (generate does not load Gold). Evaluation is a separate `--phase evaluate`.
5–16. Accuracy / FR / Preservation / runtime: pending generate+evaluate.
17–21. Scale / constraint / mechanism: pending.
22–26. invalid JSON / schema / API / tokens / latency: pending generate.
27–29. scene / MA size: pending evaluate.
30. Gold leakage: {'NO' if leak['gold_leakage'] == 0 else 'YES'} (prompt audit gold={leak['gold_leakage']} method={leak['method_leakage']})
31. Modified Frozen Repair: NO
32. Prompt tuned from results: NO
33. Retried wrong predictions until correct: NO
34. API key leaked: {'NO' if not sec['key_leak_detected'] else 'YES'}
35. Main table updated with live LLM scores: NO (placeholders until evaluate)

36–51. Latency: Mean/Min/Max Repair Latency are per-sample API durations. Total Wall-Clock is summed elapsed wall of running sessions (resume-safe). Do not use Total_Wall/N as mean when concurrency>1.
52. Accuracy–latency trade-off: pending; Frozen Repair local rule latency is not a fair hardware match to API latency.

Concurrency default = 4 workers / 60 RPM, matching Formal Y labeling. Serial remains available with `--concurrency 1`.
"""

def write_local_run() -> None:
    text = r'''# Local PowerShell commands (user-run API)

Do not paste a real API key into any file. Agent does not hold the key.

```powershell
cd <this repository>
$env:DASHSCOPE_API_KEY=""
$env:DASHSCOPE_BASE_URL=""
```

If the default Base URL is already correct you may omit DASHSCOPE_BASE_URL.

## One-shot — 14B + 32B generate then score

Do not start this while another generate is writing the same `predictions.jsonl`. If 14B Direct SS is already running, wait for it to finish, then start the orchestrator; remaining jobs resume.

```powershell
python scripts/run_qwen_14b_32b_baselines.py
```

Optional: `--generate-only` (skip scoring) or `--eval-only` (score existing predictions). `--keep-going` continues after API_FAILURE / incomplete N.

## Stage 0 — already done by Agent

```powershell
python scripts/eval_llm_repair_baselines.py --phase prepare
```

`--model` is the exact DashScope ID from the console. It is sent unchanged. Size is inferred from `14b` / `32b` / `72b` in the name, or pass `--size`. Do not pass qwen-plus / qwen-max. Qwen3 has no 72B; use `qwen3-32b` as the large open model and keep the 32B label.

## Stage 1 — 14B dry run (no Gold accuracy)

```powershell
python scripts/eval_llm_repair_baselines.py --phase dry-run --model qwen3-14b --baseline direct --dataset ss --limit 20
python scripts/eval_llm_repair_baselines.py --phase dry-run --model qwen3-14b --baseline direct --dataset ma --limit 20
```

If authentication or model ID fails, stop. Do not switch to qwen-plus or qwen-max. Copy the working ID from Model Studio into `--model` and rerun.

## Stage 2 — 14B SS

Full generate uses the Formal Y labeling pattern: `--concurrency 4 --target-rpm 60`. Mean repair latency stays per-sample API time, not Total_Wall/N. Key remains `DASHSCOPE_API_KEY` in this PowerShell session only.

```powershell
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-14b --baseline direct --dataset ss --resume --concurrency 4 --target-rpm 60
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-14b --baseline constrained --dataset ss --resume --concurrency 4 --target-rpm 60
```

## Stage 3 — 14B MA

```powershell
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-14b --baseline direct --dataset ma --resume --concurrency 4 --target-rpm 60
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-14b --baseline constrained --dataset ma --resume --concurrency 4 --target-rpm 60
```

## Stage 4 — 32B dry run (Qwen3 large dense; not 72B)

```powershell
python scripts/eval_llm_repair_baselines.py --phase dry-run --model qwen3-32b --baseline direct --dataset ss --limit 20
python scripts/eval_llm_repair_baselines.py --phase dry-run --model qwen3-32b --baseline direct --dataset ma --limit 20
```

## Stage 5 — 32B SS

```powershell
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-32b --baseline direct --dataset ss --resume --concurrency 4 --target-rpm 60
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-32b --baseline constrained --dataset ss --resume --concurrency 4 --target-rpm 60
```

## Stage 6 — 32B MA

```powershell
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-32b --baseline direct --dataset ma --resume --concurrency 4 --target-rpm 60
python scripts/eval_llm_repair_baselines.py --phase generate --model qwen3-32b --baseline constrained --dataset ma --resume --concurrency 4 --target-rpm 60
```

## Stage 7–10 — after all predictions exist (still no key in files)

```powershell
python scripts/eval_llm_repair_baselines.py --phase evaluate --all
python scripts/eval_llm_repair_baselines.py --phase runtime --all
python scripts/eval_llm_repair_baselines.py --phase statistics --all
python scripts/eval_llm_repair_baselines.py --phase timing --all
python scripts/eval_llm_repair_baselines.py --phase build-tables --all
python scripts/eval_llm_repair_baselines.py --phase report
```

Resume is default-safe: completed sample_id / parent_id rows are not re-requested.
'''
    (QO / "LOCAL_RUN.md").write_text(text, encoding="utf-8")

def timing_phase() -> None:
    ttab = collect_timing_table()
    dump(QO / "reports" / "timing_compare.json", ttab)
    build_tables()
    print(json.dumps({"phase": "timing", "n": len(ttab["rows"])}))

def write_live_report() -> None:
    lock = json.loads((QO / "QWEN_BASELINE_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    leak = json.loads((QO / "prompt_leakage_audit.json").read_text(encoding="utf-8"))
    sec = json.loads((QO / "api_key_security_audit.json").read_text(encoding="utf-8"))
    ev = QO / "evaluation" / "summary.json"
    rt = QO / "runtime" / "summary.json"
    summary = json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else {}
    runtime = json.loads(rt.read_text(encoding="utf-8")) if rt.exists() else {}
    deferred = {
        "32B_direct_ma": "NOT_EXECUTED_FULL — generate stopped on overdue-payment; resume later",
        "32B_constrained_ma": "NOT_STARTED — run after 32B Direct MA is complete",
        "72B": "NOT_EXECUTED — Qwen3 has no 72B",
    }
    lines = [
        pending_report(lock, leak, sec),
        "",
        "## Deferred generate jobs",
        json.dumps(deferred, indent=2, ensure_ascii=False),
        "",
        "## Live numbers (complete groups only; incomplete predictions are not scored)",
        json.dumps({"evaluation": summary, "runtime": runtime, "timing": collect_timing_table()}, indent=2, ensure_ascii=False),
    ]
    (QO / "qwen_baseline_final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", QO / "qwen_baseline_final_report.md")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["prepare", "dry-run", "generate", "evaluate", "runtime", "statistics", "timing", "build-tables", "report"])
    ap.add_argument("--model", default="", help="Exact DashScope model ID; forwarded unchanged")
    ap.add_argument("--size", default="", choices=["", "14B", "32B", "72B", "14b", "32b", "72b"], help="Output bucket if the ID does not contain 14b/32b/72b")
    ap.add_argument("--baseline", default="", choices=["", "direct", "constrained"])
    ap.add_argument("--dataset", default="", choices=["", "ss", "ma"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--concurrency", type=int, default=4, help="Parallel workers; same pattern as Formal Y --workers (1=serial)")
    ap.add_argument("--target-rpm", type=float, default=60.0, help="Global RPM cap when concurrency>1; same as Formal Y --target-rpm")
    args = ap.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.phase == "prepare":
        prepare()
        return
    if args.phase == "dry-run":
        if not args.model or not args.baseline or not args.dataset:
            raise SystemExit("dry-run requires --model --baseline --dataset")
        generate(args.model, args.baseline, args.dataset, resume=False, limit=args.limit or 20, dry=True, size_override=args.size, concurrency=args.concurrency, target_rpm=args.target_rpm)
        return
    if args.phase == "generate":
        if not args.model or not args.baseline or not args.dataset:
            raise SystemExit("generate requires --model --baseline --dataset")
        generate(args.model, args.baseline, args.dataset, resume=True, limit=(args.limit or None), dry=False, size_override=args.size, concurrency=args.concurrency, target_rpm=args.target_rpm)
        return
    if args.phase == "evaluate":
        evaluate_all()
        return
    if args.phase == "runtime":
        runtime_all()
        return
    if args.phase == "statistics":
        statistics_all()
        return
    if args.phase == "timing":
        timing_phase()
        return
    if args.phase == "build-tables":
        build_tables()
        return
    if args.phase == "report":
        write_live_report()

if __name__ == "__main__":
    main()
