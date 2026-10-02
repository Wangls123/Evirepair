from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from smarthome_mdf.multi_action_frozen_v1.y_llm_config import get_y_llm_config, y_llm_configured
from smarthome_mdf.multi_action_frozen_v1.y_labeling_view import (
    Y_PROMPT_VERSION,
    label_one_independent_y,
    prompt_version_hash,
)
from smarthome_mdf.multi_action_frozen_v1.y_llm_rate_limiter import (
    GlobalRateLimiter,
    LabelingRunMetrics,
    make_worker_llm_client,
)
from smarthome_mdf.v4_synthesis.llm_schema import LLMClient

CHECKPOINT_NAME = "y_checkpoint.jsonl"
FAILURES_NAME = "y_failures.jsonl"
PROGRESS_NAME = "y_progress.json"
SUMMARY_NAME = "y_summary.json"
ACTIVE_FAILURES_NAME = "active_failed_sample_ids.txt"
FAILED_IDS_NAME = "failed_sample_ids.txt"

DEFAULT_WORKERS = 4
DEFAULT_TARGET_RPM = 60.0

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_append_jsonl(path: Path, row: dict[str, Any]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())

def _safe_append_jsonl_locked(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        _safe_append_jsonl(path, row)

def load_completed_ids(checkpoint: Path, id_key: str = "sample_id") -> set[str]:

    done: set[str] = set()
    if not checkpoint.is_file():
        return done
    with checkpoint.open(encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            rid = row.get(id_key)
            if rid and row.get("label_status") == "LLM_LABEL_SUCCESS":
                done.add(str(rid))
    return done

def load_failure_history(failures: Path, id_key: str = "sample_id") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not failures.is_file():
        return rows
    with failures.open(encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                continue
    return rows

def compute_active_failures(
    failure_history: list[dict[str, Any]],
    completed_ids: set[str],
    *,
    id_key: str = "sample_id",
) -> set[str]:
    active: set[str] = set()
    for row in failure_history:
        sid = str(row.get(id_key) or "")
        if sid and sid not in completed_ids:
            active.add(sid)
    return active

def classify_failure_stage(rec: dict[str, Any]) -> str:
    status = str(rec.get("label_status") or "")
    if rec.get("LLM_CALL_FAIL") or status == "LLM_CALL_FAIL":
        return "API_CALL"
    if rec.get("LLM_PARSE_FAIL") or status == "LLM_PARSE_FAIL":
        return "PARSE"
    amb = rec.get("normalization_ambiguities") or []
    reasons = {str(a.get("reason") or "") for a in amb}
    if "ARGMAX_INCONSISTENCY" in reasons:
        return "ARGMAX"
    if any(r in reasons for r in ("SERVICE_OUT_OF_COMPONENT_DOMAIN", "TARGET_OUT_OF_ENTITY_CATALOG")):
        return "VALIDATION"
    if status == "Y_PIPELINE_FAILURE" or rec.get("normalization_status") == "Y_NORMALIZATION_FAIL":
        return "NORMALIZATION"
    return "OTHER"

def _item_label_meta(item: dict[str, Any]) -> tuple[str | None, str | None, int | None]:

    bb = item.get("blueprint_binding") or {}
    if bb.get("blueprint_id"):
        return str(bb.get("blueprint_id")), item.get("scene_type"), 1
    comps = item.get("components") or []
    if comps:
        bps = sorted({str(c.get("blueprint_id")) for c in comps if c.get("blueprint_id")})
        scenes = sorted({str(c.get("scene")) for c in comps if c.get("scene")})
        bp = bps[0] if len(bps) == 1 else "+".join(bps[:3]) + (f"+{len(bps)-3}" if len(bps) > 3 else "")
        scene = scenes[0] if len(scenes) == 1 else "+".join(scenes[:2])
        return bp, scene, item.get("component_count") or len(comps)
    return None, None, None

def build_failure_record(
    item: dict[str, Any],
    rec: dict[str, Any],
    *,
    id_key: str = "sample_id",
) -> dict[str, Any]:
    bp, scene, comp_n = _item_label_meta(item)
    bb = item.get("blueprint_binding") or {}
    row = {
        id_key: rec.get(id_key) or item.get(id_key),
        "blueprint_id": bp or bb.get("blueprint_id") or rec.get("blueprint_id"),
        "scene": scene or item.get("scene_type") or rec.get("scene_type"),
        "failure_stage": classify_failure_stage(rec),
        "error_type": str(rec.get("label_status") or "UNKNOWN"),
        "error_message": str(rec.get("error") or rec.get("normalization_status") or "")[:500],
        "retry_count": int(rec.get("retry_count") or 0) + int(rec.get("semantic_retry_count") or 0),
        "last_raw_llm_output": rec.get("raw_llm_output"),
        "normalization_ambiguities": rec.get("normalization_ambiguities") or [],
        "timestamp": _utc_now(),
    }
    if comp_n is not None:
        row["component_count"] = comp_n
    return row

def build_checkpoint_record(item: dict[str, Any], rec: dict[str, Any], *, id_key: str = "sample_id") -> dict[str, Any]:
    bp, scene, comp_n = _item_label_meta(item)
    bb = item.get("blueprint_binding") or {}
    cfg = get_y_llm_config()
    model = rec.get("model") or cfg.get("model")
    row = {
        id_key: rec.get(id_key) or item.get(id_key),
        "blueprint_id": bp or bb.get("blueprint_id") or rec.get("blueprint_id"),
        "scene": scene or item.get("scene_type") or rec.get("scene_type"),
        "label_decision": rec.get("label_decision"),
        "normalized_structured_y": rec.get("normalized_structured_y") or [],
        "label_confidence": rec.get("label_confidence"),
        "low_confidence": rec.get("low_confidence"),
        "alternative_hypotheses": rec.get("alternative_hypotheses") or [],
        "model": model,
        "prompt_version": rec.get("prompt_version") or Y_PROMPT_VERSION,
        "prompt_hash": rec.get("prompt_hash") or prompt_version_hash(),
        "retry_count": int(rec.get("retry_count") or 0) + int(rec.get("semantic_retry_count") or 0),
        "label_status": rec.get("label_status"),
        "normalization_status": rec.get("normalization_status"),
        "labeled_at": rec.get("labeled_at") or _utc_now(),
        "raw_llm_output": rec.get("raw_llm_output"),
        "parsed_llm_json": rec.get("parsed_llm_json"),
    }
    if rec.get("action_type"):
        row["action_type"] = rec.get("action_type")
    if rec.get("declared_effect_service"):
        row["declared_effect_service"] = rec.get("declared_effect_service")
    if rec.get("action_type_source"):
        row["action_type_source"] = rec.get("action_type_source")
    if comp_n is not None:
        row["component_count"] = comp_n
    return row

def write_active_failures(out_dir: Path, active: set[str]) -> None:
    path = out_dir / ACTIVE_FAILURES_NAME
    lines = sorted(active)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    legacy = out_dir / FAILED_IDS_NAME
    legacy.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

def write_progress(out_dir: Path, progress: dict[str, Any]) -> None:
    (out_dir / PROGRESS_NAME).write_text(
        json.dumps(progress, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

class IncrementalYRunner:

    def __init__(
        self,
        *,
        out_dir: Path,
        id_key: str = "sample_id",
        resume: bool = True,
        batch_size: int = 20,
        min_interval_sec: float = 2.0,
        max_consecutive_failures: int = 5,
        workers: int = 1,
        target_rpm: float = DEFAULT_TARGET_RPM,
        label_fn: Callable[..., dict[str, Any]] | None = None,
        client: Any | None = None,
    ):
        self.out_dir = out_dir
        self.id_key = id_key
        self.resume = resume
        self.batch_size = max(1, batch_size)
        self.min_interval_sec = min_interval_sec
        self.max_consecutive_failures = max_consecutive_failures
        self.workers = max(1, int(workers))
        self.target_rpm = float(target_rpm)
        self.label_fn = label_fn or label_one_independent_y
        self.client = client

        self.checkpoint_path = out_dir / CHECKPOINT_NAME
        self.failures_path = out_dir / FAILURES_NAME
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.completed_ids = load_completed_ids(self.checkpoint_path, id_key) if resume else set()
        self.failure_history = load_failure_history(self.failures_path, id_key)
        self.counts: Counter[str] = Counter()
        self.consecutive_api_failures = 0
        self.last_ts = 0.0
        self.start_time = _utc_now()
        self.processed_in_run = 0
        self._shutdown_requested = False

        self._io_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._in_flight: set[str] = set()
        self._metrics = LabelingRunMetrics()
        self._rate_limiter: GlobalRateLimiter | None = None
        if self.workers > 1:
            self._rate_limiter = GlobalRateLimiter(self.target_rpm)

    def _sleep_interval(self) -> None:
        now = time.time()
        elapsed = now - self.last_ts
        if elapsed < self.min_interval_sec:
            time.sleep(self.min_interval_sec - elapsed)
        self.last_ts = time.time()

    def _handle_signal(self, signum: int, frame: Any) -> None:
        self._shutdown_requested = True
        print("\n[Y] Interrupt received — flushing progress before exit...", flush=True)

    def install_signal_handlers(self) -> None:
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, self._handle_signal)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._handle_signal)

    def _update_progress(self, total: int, current_index: int, last_id: str | None) -> None:
        with self._state_lock:
            completed_n = len(self.completed_ids)
            failure_history = list(self.failure_history)
        active = compute_active_failures(failure_history, self.completed_ids, id_key=self.id_key)
        write_active_failures(self.out_dir, active)
        cfg = get_y_llm_config()
        progress = {
            "total_samples": total,
            "completed_success": completed_n,
            "active_failures": len(active),
            "remaining": max(0, total - completed_n),
            "current_index": current_index,
            "last_completed_sample_id": last_id,
            "start_time": self.start_time,
            "last_update_time": _utc_now(),
            "model": cfg.get("model"),
            "prompt_version": Y_PROMPT_VERSION,
            "checkpoint_sha256": _sha256_file(self.checkpoint_path),
            "workers": self.workers,
            "target_rpm": self.target_rpm if self.workers > 1 else None,
            **self._metrics.snapshot(),
        }
        write_progress(self.out_dir, progress)

    def _flush_batch(self, batch_num: int, total: int) -> None:
        with self._state_lock:
            completed_n = len(self.completed_ids)
            failure_history = list(self.failure_history)
        active = compute_active_failures(failure_history, self.completed_ids, id_key=self.id_key)
        print(
            f"[batch {batch_num}] success={completed_n} "
            f"failed_active={len(active)} remaining={max(0, total - completed_n)}",
            flush=True,
        )

    def _is_fatal_error(self, rec: dict[str, Any]) -> bool:
        err = str(rec.get("error") or "").lower()
        fatal_tokens = (
            "llm_api_not_configured",
            "not configured",
            "invalid api key",
            "authentication",
            "quota",
            "401",
            "403",
        )
        return rec.get("LLM_CALL_FAIL") and any(t in err for t in fatal_tokens)

    def _ensure_sequential_client(self) -> None:
        if self.client is None and self.label_fn is label_one_independent_y:
            if not y_llm_configured():
                raise RuntimeError(
                    "LLM not configured — edit y_llm_config.py (Y_LLM_API_KEY) or set DASHSCOPE_API_KEY"
                )
            ycfg = get_y_llm_config()
            self.client = LLMClient(
                api_key=ycfg["api_key"],
                base_url=ycfg["base_url"],
                model=ycfg["model"],
                temperature=ycfg["temperature"],
                top_p=ycfg["top_p"],
            )

    def _build_user_prompt(
        self,
        item: dict[str, Any],
        view: dict[str, Any],
        build_prompt: Callable[..., str],
    ) -> str:
        try:
            return build_prompt(view, item)
        except TypeError:
            return build_prompt(view)

    def _record_outcome(
        self,
        item: dict[str, Any],
        rec: dict[str, Any],
        *,
        last_success_holder: list[str | None],
    ) -> None:
        rid = str(item.get(self.id_key) or "")
        retries = int(rec.get("retry_count") or 0) + int(rec.get("semantic_retry_count") or 0)
        self._metrics.record_retries(retries)

        if rec.get("label_status") == "LLM_LABEL_SUCCESS":
            cp = build_checkpoint_record(item, rec, id_key=self.id_key)
            _safe_append_jsonl_locked(self.checkpoint_path, cp, self._io_lock)
            with self._state_lock:
                self.completed_ids.add(rid)
                self.consecutive_api_failures = 0
            last_success_holder[0] = rid
            self.counts["successful"] += 1
            dec = str(rec.get("label_decision") or "")
            self.counts[dec] += 1
            if rec.get("low_confidence"):
                self.counts["LOW_CONFIDENCE"] += 1
        else:
            fail = build_failure_record(item, rec, id_key=self.id_key)
            _safe_append_jsonl_locked(self.failures_path, fail, self._io_lock)
            with self._state_lock:
                self.failure_history.append(fail)
            self.counts["failed"] += 1
            stage = fail["failure_stage"]
            self.counts[f"failure_{stage}"] += 1

            if stage == "API_CALL":
                with self._state_lock:
                    self.consecutive_api_failures += 1
                    consec = self.consecutive_api_failures
            else:
                with self._state_lock:
                    self.consecutive_api_failures = 0
                    consec = self.consecutive_api_failures

            if self._is_fatal_error(rec):
                print(f"[Y] Fatal API error — stopping: {rec.get('error')}", flush=True)
                self._shutdown_requested = True
            elif consec >= self.max_consecutive_failures:
                print(
                    f"[Y] {consec} consecutive API failures — stopping.",
                    flush=True,
                )
                self._shutdown_requested = True

        status = str(rec.get("label_status") or "UNKNOWN")
        self.counts[status] += 1

    def _process_one(
        self,
        i: int,
        item: dict[str, Any],
        *,
        build_view: Callable[[dict], dict[str, Any]],
        build_prompt: Callable[..., str],
        system_prompt: str,
        client: Any | None,
        verbose: bool,
        total: int,
        last_success_holder: list[str | None],
    ) -> None:
        rid = str(item.get(self.id_key) or "")
        if not rid:
            return

        with self._state_lock:
            if self.resume and rid in self.completed_ids:
                self.counts["skipped_done"] += 1
                return
            if rid in self._in_flight:
                return
            self._in_flight.add(rid)

        try:
            view = build_view(item)
            try:
                user_prompt = self._build_user_prompt(item, view, build_prompt)
            except Exception as exc:
                rec = {
                    self.id_key: rid,
                    "label_status": "Y_PIPELINE_FAILURE",
                    "label_decision": None,
                    "error": f"prompt_build_failed: {exc}"[:300],
                    "LLM_CALL_FAIL": False,
                    "LLM_PARSE_FAIL": False,
                    "normalization_status": "PROMPT_BUILD_FAIL",
                }
                self._record_outcome(item, rec, last_success_holder=last_success_holder)
                self.processed_in_run += 1
                if verbose:
                    print(f"[{i + 1}/{total}] prompt_build_failed sample_id={rid}: {exc}", flush=True)
                return

            rec = self.label_fn(
                view,
                record_id_key=self.id_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                client=client,
                development_pilot_only=False,
            )
            rec["labeled_at"] = _utc_now()
            self._record_outcome(item, rec, last_success_holder=last_success_holder)
            self.processed_in_run += 1

            if verbose:
                with self._state_lock:
                    completed_n = len(self.completed_ids)
                    failure_history = list(self.failure_history)
                active = compute_active_failures(failure_history, self.completed_ids, id_key=self.id_key)
                status = str(rec.get("label_status") or "UNKNOWN")
                print(
                    f"[{i + 1}/{total}] success={completed_n} "
                    f"failed={len(active)} remaining={max(0, total - completed_n)} "
                    f"sample_id={rid} status={status}",
                    flush=True,
                )
        finally:
            with self._state_lock:
                self._in_flight.discard(rid)

    def _run_sequential(
        self,
        items: list[dict[str, Any]],
        *,
        build_view: Callable[[dict], dict[str, Any]],
        build_prompt: Callable[..., str],
        system_prompt: str,
        verbose: bool,
    ) -> None:

        self._ensure_sequential_client()
        total = len(items)
        batch_num = 0
        batch_in_run = 0
        last_success_holder: list[str | None] = [None]

        for i, item in enumerate(items):
            if self._shutdown_requested:
                break

            rid = str(item.get(self.id_key) or "")
            if not rid:
                continue
            if self.resume and rid in self.completed_ids:
                self.counts["skipped_done"] += 1
                continue

            self._sleep_interval()

            view = build_view(item)
            try:
                user_prompt = self._build_user_prompt(item, view, build_prompt)
            except Exception as exc:
                rec = {
                    self.id_key: rid,
                    "label_status": "Y_PIPELINE_FAILURE",
                    "label_decision": None,
                    "error": f"prompt_build_failed: {exc}"[:300],
                    "LLM_CALL_FAIL": False,
                    "LLM_PARSE_FAIL": False,
                    "normalization_status": "PROMPT_BUILD_FAIL",
                }
                fail = build_failure_record(item, rec, id_key=self.id_key)
                _safe_append_jsonl(self.failures_path, fail)
                self.failure_history.append(fail)
                self.counts["failed"] += 1
                self.counts["failure_PROMPT_BUILD"] += 1
                self.processed_in_run += 1
                if verbose:
                    print(f"[{i + 1}/{total}] prompt_build_failed sample_id={rid}: {exc}", flush=True)
                continue

            rec = self.label_fn(
                view,
                record_id_key=self.id_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                client=self.client,
                development_pilot_only=False,
            )
            rec["labeled_at"] = _utc_now()

            if rec.get("label_status") == "LLM_LABEL_SUCCESS":
                cp = build_checkpoint_record(item, rec, id_key=self.id_key)
                _safe_append_jsonl(self.checkpoint_path, cp)
                self.completed_ids.add(rid)
                self.consecutive_api_failures = 0
                last_success_holder[0] = rid
                self.counts["successful"] += 1
                dec = str(rec.get("label_decision") or "")
                self.counts[dec] += 1
                if rec.get("low_confidence"):
                    self.counts["LOW_CONFIDENCE"] += 1
            else:
                fail = build_failure_record(item, rec, id_key=self.id_key)
                _safe_append_jsonl(self.failures_path, fail)
                self.failure_history.append(fail)
                self.counts["failed"] += 1
                stage = fail["failure_stage"]
                self.counts[f"failure_{stage}"] += 1

                if stage == "API_CALL":
                    self.consecutive_api_failures += 1
                else:
                    self.consecutive_api_failures = 0

                if self._is_fatal_error(rec):
                    print(f"[Y] Fatal API error — stopping: {rec.get('error')}", flush=True)
                    self._shutdown_requested = True
                elif self.consecutive_api_failures >= self.max_consecutive_failures:
                    print(
                        f"[Y] {self.consecutive_api_failures} consecutive API failures — stopping.",
                        flush=True,
                    )
                    self._shutdown_requested = True

            self.processed_in_run += 1
            batch_in_run += 1
            status = str(rec.get("label_status") or "UNKNOWN")
            self.counts[status] += 1

            if verbose:
                active = compute_active_failures(
                    self.failure_history, self.completed_ids, id_key=self.id_key
                )
                print(
                    f"[{i + 1}/{total}] success={len(self.completed_ids)} "
                    f"failed={len(active)} remaining={max(0, total - len(self.completed_ids))} "
                    f"sample_id={rid} status={status}",
                    flush=True,
                )

            if batch_in_run >= self.batch_size:
                batch_num += 1
                self._update_progress(total, i + 1, last_success_holder[0])
                self._flush_batch(batch_num, total)
                batch_in_run = 0

            if self._shutdown_requested:
                break

    def _run_parallel(
        self,
        items: list[dict[str, Any]],
        *,
        build_view: Callable[[dict], dict[str, Any]],
        build_prompt: Callable[..., str],
        system_prompt: str,
        verbose: bool,
    ) -> None:
        if self.label_fn is label_one_independent_y and not y_llm_configured():
            raise RuntimeError(
                "LLM not configured — edit y_llm_config.py (Y_LLM_API_KEY) or set DASHSCOPE_API_KEY"
            )

        total = len(items)
        batch_num = 0
        batch_in_run = 0
        last_success_holder: list[str | None] = [None]

        pending: list[tuple[int, dict[str, Any]]] = []
        for i, item in enumerate(items):
            rid = str(item.get(self.id_key) or "")
            if not rid:
                continue
            if self.resume and rid in self.completed_ids:
                self.counts["skipped_done"] += 1
                continue
            pending.append((i, item))

        print(
            f"[Y] Parallel labeling: workers={self.workers} target_rpm={self.target_rpm} "
            f"pending={len(pending)}",
            flush=True,
        )

        def _worker_task(i: int, item: dict[str, Any]) -> None:
            if self._shutdown_requested:
                return
            if self.label_fn is label_one_independent_y:
                worker_client = make_worker_llm_client(
                    rate_limiter=self._rate_limiter,
                    metrics=self._metrics,
                )
            elif self.client is not None:
                worker_client = self.client
            else:
                worker_client = None
            self._process_one(
                i,
                item,
                build_view=build_view,
                build_prompt=build_prompt,
                system_prompt=system_prompt,
                client=worker_client,
                verbose=verbose,
                total=total,
                last_success_holder=last_success_holder,
            )

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures: dict[Future[None], tuple[int, str]] = {}
            pending_iter = iter(pending)

            def _submit_next() -> None:
                if self._shutdown_requested:
                    return
                try:
                    i, item = next(pending_iter)
                except StopIteration:
                    return
                rid = str(item.get(self.id_key) or "")
                fut = executor.submit(_worker_task, i, item)
                futures[fut] = (i, rid)

            for _ in range(min(self.workers, len(pending))):
                _submit_next()

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    i, rid = futures.pop(fut)
                    try:
                        fut.result()
                    except KeyboardInterrupt:
                        self._shutdown_requested = True
                        raise
                    except Exception as exc:
                        print(f"[Y] Worker error sample_id={rid}: {exc}", flush=True)
                        self._shutdown_requested = True

                    batch_in_run += 1
                    if batch_in_run >= self.batch_size:
                        batch_num += 1
                        self._update_progress(total, i + 1, last_success_holder[0])
                        self._flush_batch(batch_num, total)
                        batch_in_run = 0

                    if not self._shutdown_requested:
                        _submit_next()

                if self._shutdown_requested:
                    break

    def run(
        self,
        items: list[dict[str, Any]],
        *,
        build_view: Callable[[dict], dict[str, Any]],
        build_prompt: Callable[..., str],
        system_prompt: str,
        verbose: bool = True,
    ) -> dict[str, Any]:
        self.install_signal_handlers()
        total = len(items)
        last_success_id: str | None = None

        cfg = get_y_llm_config()
        print(
            f"[Y] Starting labeling: workers={self.workers} "
            f"target_rpm={self.target_rpm if self.workers > 1 else 'n/a (serial)'} "
            f"model={cfg.get('model')} total={total}",
            flush=True,
        )

        try:
            if self.workers == 1:
                self._run_sequential(
                    items,
                    build_view=build_view,
                    build_prompt=build_prompt,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )
            else:
                self._run_parallel(
                    items,
                    build_view=build_view,
                    build_prompt=build_prompt,
                    system_prompt=system_prompt,
                    verbose=verbose,
                )
        finally:
            self._update_progress(total, total, last_success_id)
            summary = self.build_summary(total)
            (self.out_dir / SUMMARY_NAME).write_text(
                json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            self._log_run_summary(summary)

        return self.build_summary(total)

    def _log_run_summary(self, summary: dict[str, Any]) -> None:
        print(
            f"[Y] Run complete: workers={summary.get('workers')} "
            f"target_rpm={summary.get('target_rpm')} "
            f"success={summary.get('successful')} failure_active={summary.get('failed_active')} "
            f"api_calls={summary.get('total_api_calls')} "
            f"429_count={summary.get('rate_limit_429_count')} "
            f"avg_latency_sec={summary.get('average_latency_sec')} "
            f"retries={summary.get('retry_count')}",
            flush=True,
        )

    def build_summary(self, total_input: int) -> dict[str, Any]:
        active = compute_active_failures(self.failure_history, self.completed_ids, id_key=self.id_key)
        cfg = get_y_llm_config()
        retries = [int(r.get("retry_count") or 0) for r in self.failure_history]
        avg_retries = sum(retries) / len(retries) if retries else 0.0
        metrics = self._metrics.snapshot()
        return {
            "total_input": total_input,
            "successful": len(self.completed_ids),
            "failed_active": len(active),
            "remaining": max(0, total_input - len(self.completed_ids)),
            "processed_this_run": self.processed_in_run,
            "skipped_done": self.counts["skipped_done"],
            "ACTIONS": self.counts.get("ACTIONS", 0),
            "NO_ACTION": self.counts.get("NO_ACTION", 0),
            "LOW_CONFIDENCE": self.counts.get("LOW_CONFIDENCE", 0),
            "api_failures": self.counts.get("failure_API_CALL", 0),
            "parse_failures": self.counts.get("failure_PARSE", 0),
            "validation_failures": self.counts.get("failure_VALIDATION", 0),
            "normalization_failures": self.counts.get("failure_NORMALIZATION", 0),
            "average_retries": round(avg_retries, 3),
            "status_counts": dict(self.counts),
            "model": cfg.get("model"),
            "prompt_version": Y_PROMPT_VERSION,
            "prompt_hash": prompt_version_hash(),
            "checkpoint": str(self.checkpoint_path),
            "failures": str(self.failures_path),
            "checkpoint_sha256": _sha256_file(self.checkpoint_path),
            "finished_at": _utc_now(),
            "workers": self.workers,
            "target_rpm": self.target_rpm if self.workers > 1 else None,
            "success_count": len(self.completed_ids),
            "failure_count": self.counts.get("failed", 0),
            **metrics,
        }

def run_incremental_y_labeling(
    *,
    items: list[dict[str, Any]],
    out_dir: Path,
    build_view: Callable[[dict], dict[str, Any]],
    build_prompt: Callable[..., str],
    system_prompt: str,
    id_key: str = "sample_id",
    resume: bool = True,
    batch_size: int = 20,
    min_interval_sec: float = 2.0,
    max_consecutive_failures: int = 5,
    workers: int = 1,
    target_rpm: float = DEFAULT_TARGET_RPM,
    label_fn: Callable[..., dict[str, Any]] | None = None,
    client: Any | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    runner = IncrementalYRunner(
        out_dir=out_dir,
        id_key=id_key,
        resume=resume,
        batch_size=batch_size,
        min_interval_sec=min_interval_sec,
        max_consecutive_failures=max_consecutive_failures,
        workers=workers,
        target_rpm=target_rpm,
        label_fn=label_fn,
        client=client,
    )
    return runner.run(
        items,
        build_view=build_view,
        build_prompt=build_prompt,
        system_prompt=system_prompt,
        verbose=verbose,
    )
