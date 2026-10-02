from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smarthome_mdf.openhab_validation.openhab_dsl_parser import parse_openhab_directory
from smarthome_mdf.openhab_validation.types import OpenHABScenario

LABEL_VERSION = "openhab_llm_semantic_label_v1"
FORBIDDEN_INPUT = frozenset({"repair_plan", "repaired_actions", "hidden_eval", "expected_obligations"})

def _public_dir(dataset_root: Path, scenario_id: str) -> Path:
    return dataset_root / "public" / scenario_id / "public"

def build_public_prompt(scenario_id: str, pub: Path) -> str:
    items = (pub / "items" / f"{scenario_id}.items").read_text(encoding="utf-8") if (pub / "items" / f"{scenario_id}.items").exists() else ""
    rules = (pub / "rules" / f"{scenario_id}.rules").read_text(encoding="utf-8") if (pub / "rules" / f"{scenario_id}.rules").exists() else ""
    events = (pub / "events" / "event_sequence.json").read_text(encoding="utf-8") if (pub / "events" / "event_sequence.json").exists() else "[]"
    initial = (pub / "runtime" / "initial_state.json").read_text(encoding="utf-8") if (pub / "runtime" / "initial_state.json").exists() else "{}"
    return (
        "Analyze this OpenHAB automation scenario. Return JSON only with keys: "
        "expected_obligations (list of {item, command}), forbidden_actions (list), "
        "conflict_type (string), repair_required (bool), ambiguity (bool).\n\n"
        f"SCENARIO_ID: {scenario_id}\n\n"
        f"ITEMS:\n{items}\n\nRULES:\n{rules}\n\n"
        f"EVENTS:\n{events}\n\nINITIAL_STATE:\n{initial}\n"
    )

def _structure_derived_label(scenario_id: str, pub: Path) -> dict[str, Any]:

    model = parse_openhab_directory(pub)
    events = json.loads((pub / "events" / "event_sequence.json").read_text(encoding="utf-8"))
    obligations: list[dict[str, str]] = []
    forbidden: list[dict[str, str]] = []
    for _rid, rule in model["rules"].items():
        acts = rule.actions if hasattr(rule, "actions") else (rule.get("actions") if isinstance(rule, dict) else [])
        for act in acts:
            if act.get("type") == "sendCommand":
                obligations.append({"item": act["item"], "command": str(act["command"]).upper()})
    conflict_type = "unknown"
    if "missing" in scenario_id:
        conflict_type = "missing_action"
    elif "opposing" in scenario_id:
        conflict_type = "opposing_light"
    elif "remove" in scenario_id:
        conflict_type = "remove_conflict"
    elif "extra" in scenario_id:
        conflict_type = "extra_action"
    elif "sequence" in scenario_id:
        conflict_type = "sequence"
    elif "mixed" in scenario_id:
        conflict_type = "mixed_conflict"
    elif "clean" in scenario_id:
        conflict_type = "clean_no_conflict"
    elif "ambiguous" in scenario_id or "adversarial" in scenario_id:
        conflict_type = "ambiguous"
    elif "unfixable" in scenario_id:
        conflict_type = "unfixable_opposing"
    repair_required = conflict_type not in ("clean_no_conflict", "unknown")
    ambiguity = "ambiguous" in scenario_id or "adversarial" in scenario_id
    return {
        "scenario_id": scenario_id,
        "expected_obligations": obligations,
        "forbidden_actions": forbidden,
        "conflict_type": conflict_type,
        "repair_required": repair_required,
        "ambiguity": ambiguity,
        "trigger_events": events,
        "annotation_source": "public_structure_v1",
    }

def _call_llm(prompt: str) -> tuple[str, str]:
    api_key = os.environ.get("OPENHAB_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base = os.environ.get("OPENHAB_LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("OPENHAB_LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("LLM API key not configured")
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return model, content

def annotate_scenario(scenario_id: str, dataset_root: Path) -> dict[str, Any]:
    pub = _public_dir(dataset_root, scenario_id)
    prompt = build_public_prompt(scenario_id, pub)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record: dict[str, Any] = {
        "scenario_id": scenario_id,
        "label_version": LABEL_VERSION,
        "timestamp": ts,
        "prompt": prompt,
    }
    try:
        model, response = _call_llm(prompt)
        record["model"] = model
        record["response"] = response
        parsed = json.loads(response)
        record.update(parsed)
        record["annotation_source"] = "llm"
    except Exception as exc:
        derived = _structure_derived_label(scenario_id, pub)
        record["model"] = os.environ.get("OPENHAB_LLM_MODEL", "public_structure_v1")
        record["response"] = json.dumps(derived, ensure_ascii=False)
        record["llm_error"] = str(exc)[:200]
        record.update({k: derived[k] for k in ("expected_obligations", "forbidden_actions", "conflict_type", "repair_required", "ambiguity")})
        record["annotation_source"] = derived["annotation_source"]
    return record

def annotate_corpus(dataset_root: Path, out_path: Path, scenario_ids: list[str] | None = None) -> dict[str, Any]:
    manifest = json.loads((dataset_root / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))
    ids = scenario_ids or [r["scenario_id"] for r in manifest["scenarios"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = out_path.parent / "annotation_audit"
    audit_dir.mkdir(exist_ok=True)
    n_llm = n_derived = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sid in ids:
            row = annotate_scenario(sid, dataset_root)
            if row.get("annotation_source") == "llm":
                n_llm += 1
            else:
                n_derived += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            (audit_dir / f"{sid}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return {"n": len(ids), "n_llm": n_llm, "n_structure_derived": n_derived, "output": str(out_path)}
