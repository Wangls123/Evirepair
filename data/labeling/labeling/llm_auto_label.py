from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from smarthome_mdf.labeling.blueprint_constraints import (
    CONTRACT_CONSTRAINED_SYSTEM_PROMPT,
    blueprint_id_from_sample,
    build_constraint_prompt_block,
    get_blueprint_constraint,
    get_label_constraint_for_sample,
    has_out_of_contract_actions,
    load_visual_constraints,
    validate_llm_annotation,
)
from smarthome_mdf.paths import HA_REGISTRY, LLM_LABEL_SCENES, QUALITY_REPORT_DIR, SAMPLES_DIR
from smarthome_mdf.synthesis.labeling_config import pending_decision_output

def resolve_llm_api_key() -> str:

    for name in ("LLM_API_KEY", "V4_LLM_API_KEY", "DASHSCOPE_API_KEY", "NVIDIA_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""

LLM_API_URL = os.environ.get("LLM_API_URL", "")
LLM_API_KEY = resolve_llm_api_key()

LLM_MODEL = "qwen/qwen3.5-397b-a17b"
LLM_MAX_TOKENS = 16384
LLM_TEMPERATURE = 0.60
LLM_TOP_P = 0.95
LLM_TOP_K = 20
LLM_PRESENCE_PENALTY = 0.0
LLM_FREQUENCY_PENALTY = 0.0
LLM_REPETITION_PENALTY = 1.0
LLM_MAX_PER_MINUTE = 40
LLM_SLEEP_SEC = 1.5
LLM_TIMEOUT_SEC = 300
LLM_RETRIES = 3
LLM_429_COOLDOWN_SEC = 90
LLM_SAMPLE_MAX_ATTEMPTS = 5
LLM_RETRY_SLEEP_SEC = 5

SCENE_FILES = {
    "climate_window": "climate_window.jsonl",
    "appliance_monitoring": "appliance_monitoring.jsonl",
    "advanced_lighting": "advanced_lighting.jsonl",
    "notification_security": "notification_security.jsonl",
    "scene_schedule_override": "scene_schedule_override.jsonl",
    "visual_fusion": "visual_fusion.jsonl",
    "periodic_task_scheduling": "periodic_task_scheduling.jsonl",
    "on_off_schedule": "on_off_schedule.jsonl",
}

MAX_YAML_CHARS = 12000

SYSTEM_PROMPT = """You are a Home Assistant Blueprint decision annotator.
Given multi-device observations and a Blueprint excerpt, output ONLY valid JSON:
{
  "expected_action": ["service.action", ...],
  "blocked_action": [],
  "conflict_type": "none",
  "sample_type": "positive|negative|conflict|boundary",
  "rationale": "short reason"
}
Use HA service names like light.turn_on, notify.mobile_app, climate.turn_off, state.snapshot, state.restore,
alarm_control_panel.trigger, script.vacuum_clean, homeassistant.turn_on, homeassistant.turn_off, schedule.activate, scene.turn_on.
If no action applies, expected_action must be [].
conflict_type must be one of: none, comfort_vs_energy, manual_override, schedule_constraint, security_alert, weather_override, energy_saving.
"""

class _RateLimiter:
    def __init__(self, max_per_minute: int = 20, min_interval_sec: float = 3.0):
        self.max_per_minute = max_per_minute
        self.min_interval_sec = min_interval_sec
        self._times: deque[float] = deque()
        self._last_request: float = 0.0
        self._cooldown_until: float = 0.0

    def penalize_429(self, cooldown_sec: float) -> None:

        self._cooldown_until = max(self._cooldown_until, time.time() + cooldown_sec)

    def wait(self) -> None:
        now = time.time()
        if now < self._cooldown_until:
            time.sleep(self._cooldown_until - now)
            now = time.time()
        while self._times and now - self._times[0] >= 60.0:
            self._times.popleft()
        if len(self._times) >= self.max_per_minute:
            sleep_for = 60.0 - (now - self._times[0]) + 0.05
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.time()
            while self._times and now - self._times[0] >= 60.0:
                self._times.popleft()
        since_last = now - self._last_request
        if since_last < self.min_interval_sec:
            time.sleep(self.min_interval_sec - since_last)
        t = time.time()
        self._times.append(t)
        self._last_request = t

def _resolve_chat_url(*, api_base: Optional[str], api_url: Optional[str]) -> str:
    if api_url:
        return api_url.strip()
    if not api_base:
        raise ValueError("Need --api-url or --api-base")
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"

class LlmClient:

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        *,
        api_base: Optional[str] = None,
        api_url: Optional[str] = None,
        max_per_minute: int = 40,
        sleep_sec: float = 1.5,
        timeout_sec: int = LLM_TIMEOUT_SEC,
        retries: int = LLM_RETRIES,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        json_mode: bool = False,
        stream: bool = False,
    ):
        self.chat_url = _resolve_chat_url(api_base=api_base, api_url=api_url)
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec
        self.retries = max(1, retries)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.json_mode = json_mode
        self.stream = stream
        self._limiter = _RateLimiter(max_per_minute, sleep_sec)

    def _build_body(
        self, messages: List[dict], *, temperature: Optional[float], minimal: bool = False
    ) -> dict:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        if minimal:
            return body
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if self.presence_penalty is not None:
            body["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            body["frequency_penalty"] = self.frequency_penalty
        if self.repetition_penalty is not None:
            body["repetition_penalty"] = self.repetition_penalty
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _normalize_content(raw: Any) -> str:
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            parts: List[str] = []
            for item in raw:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
            return "".join(parts).strip()
        return str(raw).strip()

    @classmethod
    def _extract_content(cls, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response missing choices: {json.dumps(data)[:500]}")
        ch0 = choices[0]
        msg = ch0.get("message") or {}
        for key in ("content", "reasoning_content", "reasoning", "refusal"):
            text = cls._normalize_content(msg.get(key))
            if text:
                return text
        text = cls._normalize_content(ch0.get("text"))
        if text:
            return text
        fr = ch0.get("finish_reason") or msg.get("finish_reason")
        raise RuntimeError(
            f"LLM response empty content (finish_reason={fr}): {json.dumps(ch0)[:800]}"
        )

    def chat(self, messages: List[dict], *, temperature: Optional[float] = None) -> str:
        if self.stream:
            raise NotImplementedError("Streaming not supported for batch labeling")
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):

            minimal = attempt > 1
            body = self._build_body(messages, temperature=temperature, minimal=minimal)
            self._limiter.wait()
            req = urllib.request.Request(
                self.chat_url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return self._extract_content(data)
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"LLM HTTP {e.code}: {err}")
                if e.code == 429:
                    self._limiter.penalize_429(LLM_429_COOLDOWN_SEC)
                    print(f"    [rate limit] 429 → cooldown {LLM_429_COOLDOWN_SEC}s", flush=True)
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except Exception as e:
                last_err = e
            if attempt < self.retries:
                if isinstance(last_err, RuntimeError) and "HTTP 429" in str(last_err):
                    time.sleep(min(30.0 * attempt, 120.0))
                else:
                    time.sleep(min(3.0 * attempt, 15.0))
        raise last_err if last_err else RuntimeError("LLM request failed")

def _load_registry() -> dict:
    with open(HA_REGISTRY, "r", encoding="utf-8") as f:
        return json.load(f)

def _blueprint_for_sample(sample: dict, registry: dict) -> Dict[str, Any]:

    scene = sample.get("scene_type")
    binding = sample.get("blueprint_binding") or {}
    bp: Dict[str, Any] = {"scene_type": scene}

    for k in ("blueprint_id", "blueprint_name", "blueprint_summary"):
        if binding.get(k):
            bp[k] = binding[k]

    bind_yaml = binding.get("blueprint_yaml")
    if bind_yaml:
        bp["blueprint_yaml"] = bind_yaml
    elif binding.get("blueprint_id"):
        bp["blueprint_yaml"] = None
    else:
        scene_def = (registry.get("scenes") or {}).get(scene) or {}
        bp["blueprint_yaml"] = scene_def.get("blueprint_yaml")

    if not bp.get("blueprint_summary"):
        scene_def = (registry.get("scenes") or {}).get(scene) or {}
        bp["blueprint_summary"] = scene_def.get("blueprint_logic_summary") or scene_def.get("note")

    if not bp.get("blueprint_id") and scene in (registry.get("scenes") or {}):
        scene_def = registry["scenes"][scene]
        bp.setdefault("blueprint_id", scene_def.get("blueprint_name"))
        bp.setdefault("blueprint_name", scene_def.get("blueprint_name"))

    return bp

def _read_yaml_excerpt(yaml_path: Optional[str]) -> str:
    if not yaml_path:
        return ""
    p = Path(yaml_path)
    if not p.exists():
        return f"# YAML file missing: {yaml_path}\n"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_YAML_CHARS:
        return text[:MAX_YAML_CHARS] + "\n... [truncated]"
    return text

def _blueprint_prompt_section(bp: Dict[str, Any]) -> str:
    yaml_path = bp.get("blueprint_yaml")
    if yaml_path:
        return f"## Blueprint YAML (excerpt)\n```yaml\n{_read_yaml_excerpt(yaml_path)}\n```"
    bid = bp.get("blueprint_id", "unknown")
    summary = bp.get("blueprint_summary") or "(no summary provided)"
    return (
        "## Blueprint specification (no YAML file; use summary only)\n"
        f"- blueprint_id: {bid}\n"
        f"- blueprint_name: {bp.get('blueprint_name', '')}\n"
        f"- logic_summary: {summary}\n"
        "Do NOT apply a different blueprint's rules (e.g. do not use Frigate notification logic "
        "if this id is outdoor_lighting_frigate)."
    )

def _build_user_prompt_constrained(
    sample: dict,
    bp: Dict[str, Any],
    yaml_excerpt: str,
    constraint: dict,
    manifest: dict,
) -> str:

    bid = constraint.get("blueprint_id") or bp.get("blueprint_id")
    payload = {
        "sample_id": sample.get("sample_id"),
        "scene_type": sample.get("scene_type"),
        "blueprint_id": bid,
        "entity_observations": sample.get("entity_observations"),
        "observed": sample.get("observed"),
        "system_state": sample.get("system_state"),
        "derived_observation": sample.get("derived_observation"),
    }
    if sample.get("scene_type") == "visual_fusion":
        payload["field_legend"] = {
            "synthetic_extension_fields": manifest.get("synthetic_extension_fields"),
            "note": "Fields in synthetic_extension_fields must NOT dominate unless Blueprint supports them.",
        }
    else:
        payload["blueprint"] = {
            "blueprint_id": bp.get("blueprint_id") or bp.get("blueprint_name"),
            "blueprint_name": bp.get("blueprint_name"),
            "logic_summary": bp.get("blueprint_logic_summary") or bp.get("blueprint_summary"),
        }
    bp_yaml = (
        f"## Blueprint YAML (excerpt)\n```yaml\n{yaml_excerpt}\n```"
        if yaml_excerpt
        else _blueprint_prompt_section(bp)
    )
    constraint_block = build_constraint_prompt_block(constraint, manifest)
    return (
        "Annotate under Blueprint Action Contract. Y = label truth for evaluation.\n"
        "All actions MUST be within Allowed actions (whitelist) and Action domains.\n\n"
        f"{constraint_block}\n\n"
        f"{bp_yaml}\n\n"
        f"## Sample JSON\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )

def _build_user_prompt_visual_constrained(
    sample: dict,
    bp: Dict[str, Any],
    yaml_excerpt: str,
    manifest: dict,
) -> str:
    bid = bp.get("blueprint_id") or blueprint_id_from_sample(sample)
    constraint = get_blueprint_constraint(bid, manifest)
    return _build_user_prompt_constrained(sample, bp, yaml_excerpt, constraint, manifest)

def _build_user_prompt(sample: dict, bp: Dict[str, Any], yaml_excerpt: str) -> str:
    payload = {
        "sample_id": sample.get("sample_id"),
        "scene_type": sample.get("scene_type"),
        "blueprint": {
            "blueprint_id": bp.get("blueprint_id") or bp.get("blueprint_name"),
            "blueprint_name": bp.get("blueprint_name"),
            "logic_summary": bp.get("blueprint_logic_summary") or bp.get("blueprint_summary"),
        },
        "entity_observations": sample.get("entity_observations"),
        "observed": sample.get("observed"),
        "system_state": sample.get("system_state"),
        "derived_observation": sample.get("derived_observation"),
    }
    bp_section = _blueprint_prompt_section(bp) if not yaml_excerpt else (
        f"## Blueprint YAML (excerpt)\n```yaml\n{yaml_excerpt}\n```"
    )
    return (
        "Annotate this SmartHome fusion sample per the Blueprint.\n\n"
        f"{bp_section}\n\n"
        f"## Sample JSON\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )

def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def _clear_llm_labels(sample: dict) -> None:
    sample["decision_output"] = pending_decision_output()
    meta = sample.setdefault("metadata", {})
    meta["label_status"] = "pending"
    meta.pop("labeled_by", None)

def _is_retryable_error(exc: BaseException) -> bool:

    if isinstance(exc, (TimeoutError, urllib.error.URLError)):
        return True
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg:
        return True
    if isinstance(exc, RuntimeError) and "HTTP " in str(exc):
        for code in ("429", "500", "502", "503", "504"):
            if f"HTTP {code}" in str(exc):
                return True
    return False

def _retry_sleep_seconds(exc: BaseException, attempt: int) -> float:

    if "HTTP 429" in str(exc):
        return min(30.0 * attempt, LLM_429_COOLDOWN_SEC)
    return float(LLM_RETRY_SLEEP_SEC)

def chat_with_retry(
    client: LlmClient,
    messages: List[dict],
    sample_id: str = "unknown",
) -> str:

    last_err: Optional[Exception] = None
    for attempt in range(1, LLM_SAMPLE_MAX_ATTEMPTS + 1):
        try:
            return client.chat(messages)
        except Exception as e:
            last_err = e
            if attempt >= LLM_SAMPLE_MAX_ATTEMPTS:
                break
            if not _is_retryable_error(e):
                break
            wait = _retry_sleep_seconds(e, attempt)
            print(
                f"    [{sample_id}] attempt {attempt}/{LLM_SAMPLE_MAX_ATTEMPTS} failed: {e} "
                f"→ sleep {wait:.0f}s retry",
                flush=True,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"sample {sample_id} failed after {LLM_SAMPLE_MAX_ATTEMPTS} attempts: {last_err}"
    )

def print_failed_sample_ids(
    failed_ids: List[str],
    *,
    title: str = "Failed sample_ids (re-run these)",
) -> None:
    if not failed_ids:
        return
    print(f"\n=== {title} ===", flush=True)
    for sid in failed_ids:
        print(sid, flush=True)
    print(f"=== total {len(failed_ids)} ===\n", flush=True)

def _apply_llm_annotation(
    sample: dict,
    ann: dict,
    *,
    constraint: Optional[dict] = None,
    violations: Optional[List[str]] = None,
) -> dict:
    dec = sample.setdefault("decision_output", {})
    dec["expected_action"] = ann.get("expected_action") or []
    dec["blocked_action"] = ann.get("blocked_action") or []
    dec["conflict_type"] = ann.get("conflict_type") or "none"
    dec["label_status"] = "llm"
    dec["label_confidence"] = float(ann.get("label_confidence") or 0.0)
    dec["used_blueprint_fields"] = ann.get("used_blueprint_fields") or []
    dec["ignored_non_blueprint_fields"] = ann.get("ignored_non_blueprint_fields") or []
    dec["label_rationale"] = ann.get("reason") or ann.get("rationale", "")
    dec["fused_actions"] = [{"action": a} for a in dec["expected_action"]]
    if constraint:
        dec["blueprint_id"] = constraint.get("blueprint_id")
        dec["labeling_policy"] = "blueprint_action_contract_v1"
        dec["action_domains"] = list(constraint.get("action_domains") or [])
    if violations:
        dec["constraint_violations"] = violations
    if ann.get("sample_type"):
        sample["sample_type"] = ann["sample_type"]
    meta = sample.setdefault("metadata", {})
    meta["label_status"] = "llm"
    meta["labeled_by"] = "llm_auto_label_constrained" if constraint else "llm_auto_label"
    if ann.get("notify_priority"):
        meta["notify_priority"] = ann["notify_priority"]
    return sample

def label_one_sample(
    sample: dict,
    client: LlmClient,
    registry: dict,
    *,
    visual_manifest: Optional[dict] = None,
) -> dict:

    sid = sample.get("sample_id", "unknown")
    bp = _blueprint_for_sample(sample, registry)
    yaml_excerpt = _read_yaml_excerpt(bp.get("blueprint_yaml")) if bp.get("blueprint_yaml") else ""

    constraint = get_label_constraint_for_sample(sample)
    if constraint:
        manifest = visual_manifest or load_visual_constraints()
        system_prompt = CONTRACT_CONSTRAINED_SYSTEM_PROMPT
        user_prompt = _build_user_prompt_constrained(
            sample, bp, yaml_excerpt, constraint, manifest
        )
        strict_field_usage = sample.get("scene_type") == "visual_fusion"
    else:
        strict_field_usage = False
        system_prompt = SYSTEM_PROMPT
        user_prompt = _build_user_prompt(sample, bp, yaml_excerpt)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Optional[Exception] = None
    for attempt in range(1, LLM_SAMPLE_MAX_ATTEMPTS + 1):
        try:
            raw = client.chat(messages)
            ann = _parse_llm_json(raw)
            if not ann:
                raise ValueError("LLM response is not valid JSON")
            violations: List[str] = []
            if constraint:
                violations, ann = validate_llm_annotation(
                    ann,
                    constraint,
                    sample,
                    strict_field_usage=strict_field_usage,
                )
                if has_out_of_contract_actions(violations):
                    raise ValueError(
                        f"LLM actions outside contract allowed_actions: {violations}"
                    )
            return _apply_llm_annotation(sample, ann, constraint=constraint, violations=violations)
        except Exception as e:
            last_err = e
            if attempt >= LLM_SAMPLE_MAX_ATTEMPTS:
                break
            wait = _retry_sleep_seconds(e, attempt)
            print(
                f"    [{sid}] attempt {attempt}/{LLM_SAMPLE_MAX_ATTEMPTS} failed: {e} "
                f"→ sleep {wait:.0f}s retry",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"sample {sid} failed after {LLM_SAMPLE_MAX_ATTEMPTS} attempts: {last_err}")

def _needs_label(sample: dict, force: bool) -> bool:
    if force:
        return True
    dec = sample.get("decision_output", {})
    st = dec.get("label_status") or sample.get("metadata", {}).get("label_status")
    if st == "llm" and dec.get("expected_action"):
        return False
    if st == "pending":
        return True
    return not dec.get("expected_action")

def _write_jsonl(path: Path, samples: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

def run_labeling(
    *,
    samples_dir: Path,
    scene: str,
    client: LlmClient,
    limit: Optional[int] = None,
    force: bool = False,
    clear_labels: bool = False,
    in_place: bool = True,
    sample_ids: Optional[set] = None,
) -> dict:
    fname = SCENE_FILES.get(scene)
    if not fname:
        raise ValueError(f"Unknown scene: {scene}")
    in_path = samples_dir / fname
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = in_path if in_place else samples_dir / fname.replace(".jsonl", "_labeled.jsonl")
    registry = _load_registry()
    visual_manifest = load_visual_constraints()
    samples = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    log = {
        "scene": scene,
        "total": len(samples),
        "labeled": 0,
        "skipped": 0,
        "errors": [],
        "failed_sample_ids": [],
        "checkpoint": "after_each_successful_label",
    }

    if clear_labels:
        for sample in samples:
            _clear_llm_labels(sample)
        _write_jsonl(out_path, samples)
        log["cleared"] = len(samples)

    for i, sample in enumerate(samples):
        sid = sample.get("sample_id")
        if sample_ids is not None and sid not in sample_ids:
            continue
        if limit is not None and log["labeled"] >= limit:
            continue
        if not _needs_label(sample, force):
            log["skipped"] += 1
            continue
        print(f"  [{i+1}/{len(samples)}] {sid} calling LLM...", flush=True)
        try:
            samples[i] = label_one_sample(sample, client, registry, visual_manifest=visual_manifest)
            log["labeled"] += 1
            _write_jsonl(out_path, samples)
            print(f"  [{i+1}/{len(samples)}] {sid} OK (checkpoint saved)", flush=True)
        except Exception as e:
            log["errors"].append({"sample_id": sid, "error": str(e), "attempts": LLM_SAMPLE_MAX_ATTEMPTS})
            if sid:
                log["failed_sample_ids"].append(sid)
            print(f"  [{i+1}/{len(samples)}] SKIP sample_id={sid} ERROR: {e}", flush=True)

    print_failed_sample_ids(log["failed_sample_ids"], title=f"{scene} failed sample_ids")
    log["output"] = str(out_path)
    return log

def main() -> None:
    parser = argparse.ArgumentParser(description="Blueprint LLM labeling (<=40 req/min)")
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--max-tokens", type=int, default=LLM_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=LLM_TOP_P)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--presence-penalty", type=float, default=LLM_PRESENCE_PENALTY)
    parser.add_argument("--frequency-penalty", type=float, default=LLM_FREQUENCY_PENALTY)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="请求体加 response_format=json_object（OpenAI；NVIDIA 一般不要开）",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=Path,
        default=None,
        help="Only label samples whose sample_id is listed (one per line or JSON array)",
    )
    parser.add_argument("--scene", default="all", choices=list(SCENE_FILES.keys()) + ["all"])
    parser.add_argument("--include-visual", action="store_true")
    parser.add_argument("--samples-dir", default=str(SAMPLES_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--clear-labels",
        action="store_true",
        help="打标前清空该场景全部 LLM 标签（重置为 pending）",
    )
    parser.add_argument("--no-in-place", action="store_true")
    parser.add_argument("--sleep", type=float, default=LLM_SLEEP_SEC)
    parser.add_argument("--max-per-minute", type=int, default=LLM_MAX_PER_MINUTE)
    parser.add_argument("--timeout", type=int, default=LLM_TIMEOUT_SEC, help="单次 HTTP 读超时（秒）")
    parser.add_argument("--retries", type=int, default=LLM_RETRIES, help="失败重试次数")
    args = parser.parse_args()

    if not LLM_API_URL.strip():
        parser.error("Set LLM_API_URL environment variable")
    if not LLM_API_KEY.strip():
        parser.error(
            "Set LLM_API_KEY (or V4_LLM_API_KEY / DASHSCOPE_API_KEY / NVIDIA_API_KEY) via environment"
        )

    client = LlmClient(
        LLM_API_KEY,
        model=args.model,
        api_url=LLM_API_URL,
        max_per_minute=args.max_per_minute,
        sleep_sec=args.sleep,
        timeout_sec=args.timeout,
        retries=args.retries,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        json_mode=args.json_mode,
    )
    scenes = (
        list(SCENE_FILES.keys())
        if args.scene == "all" and args.include_visual
        else (list(LLM_LABEL_SCENES) if args.scene == "all" else [args.scene])
    )

    sample_ids: Optional[set] = None
    if args.sample_ids_file:
        text = args.sample_ids_file.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            sample_ids = set(json.loads(text))
        else:
            sample_ids = {ln.strip() for ln in text.splitlines() if ln.strip()}

    all_logs = {}
    all_failed: List[str] = []
    for sc in scenes:
        print(f"\n=== {sc} ===", flush=True)
        all_logs[sc] = run_labeling(
            samples_dir=Path(args.samples_dir),
            scene=sc,
            client=client,
            limit=args.limit,
            force=args.force,
            clear_labels=args.clear_labels,
            in_place=not args.no_in_place,
            sample_ids=sample_ids,
        )
        all_failed.extend(all_logs[sc].get("failed_sample_ids", []))
        print(json.dumps(all_logs[sc], indent=2, ensure_ascii=False))

    print_failed_sample_ids(all_failed, title="ALL scenes failed sample_ids")

    QUALITY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = QUALITY_REPORT_DIR / "llm_labeling_report.json"
    report.write_text(json.dumps(all_logs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report}")

if __name__ == "__main__":
    main()
