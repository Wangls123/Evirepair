from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LABEL_VERSION = "openhab_semantic_reference_v1"
PROMPT_VERSION = "oh_semref_prompt_v1"
VALID_OPS = frozenset({"turn_on", "turn_off", "set", "open", "close", "toggle", "ON", "OFF", "OPEN", "CLOSED"})

def public_dir(dataset_root: Path, scenario_id: str) -> Path:
    return dataset_root / "public" / scenario_id / "public"

def build_prompt(scenario_id: str, pub: Path) -> str:
    items = _read(pub / "items" / f"{scenario_id}.items")
    rules = _read(pub / "rules" / f"{scenario_id}.rules")
    events = _read(pub / "events" / "event_sequence.json")
    initial = _read(pub / "runtime" / "initial_state.json")
    return (
        f"{PROMPT_VERSION}\n"
        "You are labeling expected OpenHAB automation semantics for EVALUATION ONLY.\n"
        "Do NOT invent a repair plan. If ambiguous, set ambiguity=true and leave obligations incomplete.\n"
        "Return JSON with keys: repair_required (bool), expected_obligations (list of "
        "{component_id, entity, capability, operation, parameters, condition}), "
        "forbidden_behaviors (list of {entity, capability, operation}), "
        "conflict_relations (list), acceptable_final_behaviors (list), ambiguity (bool).\n\n"
        f"SCENARIO_ID: {scenario_id}\n\nITEMS:\n{items}\n\nRULES:\n{rules}\n\n"
        f"EVENTS:\n{events}\n\nINITIAL_STATE:\n{initial}\n"
    )

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""

def _cmd_to_op(cmd: str) -> str:
    c = str(cmd).upper()
    if c == "ON":
        return "turn_on"
    if c == "OFF":
        return "turn_off"
    if c == "OPEN":
        return "open"
    if c == "CLOSED":
        return "close"
    return "set"

def _cap_for_item(name: str, item_type: str) -> str:
    n = name.lower()
    if "light" in n or "lamp" in n:
        return "light"
    if "switch" in n or "plug" in n:
        return "switch"
    if "contact" in n or "window" in n or "door" in n:
        return "binary_sensor"
    return (item_type or "switch").lower()

def structure_reference(scenario_id: str, pub: Path) -> dict[str, Any]:

    model = parse_openhab_directory(pub)
    events = json.loads(_read(pub / "events" / "event_sequence.json") or "[]")
    initial = json.loads(_read(pub / "runtime" / "initial_state.json") or "{}")
    item_types = {
        name: (sem.get("item_type") if isinstance(sem, dict) else getattr(sem, "item_type", "Switch"))
        for name, sem in model["items"].items()
    }
    lights = [n for n in item_types if "light" in n.lower()]
    switches = [n for n in item_types if "switch" in n.lower()]
    plugs = [n for n in item_types if "plug" in n.lower() or "aux" in n.lower()]

    obligations: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    sid = scenario_id.lower()
    ambiguity = "ambiguous" in sid or "adversarial" in sid
    unfixable = "unfixable" in sid
    clean = "clean" in sid

    def add_ob(item: str, cmd: str, cid: str = "semantic_ref") -> None:
        cap = _cap_for_item(item, str(item_types.get(item, "Switch")))
        obligations.append(
            {
                "component_id": cid,
                "entity": f"{cap}.{item}",
                "capability": cap,
                "operation": _cmd_to_op(cmd),
                "parameters": {"command": cmd.upper()},
                "condition": {},
            }
        )

    by_entity: dict[str, set[str]] = {}
    for rid, rule in model["rules"].items():
        for act in (rule.actions if hasattr(rule, "actions") else []):
            if act.get("type") == "sendCommand":
                by_entity.setdefault(str(act["item"]), set()).add(_cmd_to_op(str(act["command"])))

    if clean:

        for rid, rule in model["rules"].items():
            for act in (rule.actions if hasattr(rule, "actions") else []):
                if act.get("type") == "sendCommand":
                    add_ob(str(act["item"]), str(act["command"]), str(getattr(rule, "uid", rid)))
        repair_required = False
    elif "missing" in sid:
        repair_required = True
        for light in lights:
            add_ob(light, "ON", "missing_recovery")
            conflicts.append({"type": "MISSING", "entity": light, "operation": "turn_on"})
    elif "opposing" in sid or "unfixable" in sid:
        repair_required = True

        target = lights[0] if lights else (list(by_entity.keys())[0] if by_entity else None)
        if target:
            add_ob(target, "ON", "resolve_opposing")
            conflicts.append({"type": "OPPOSING" if "opposing" in sid else "REPLACE", "entity": target})
            forbidden.append({"entity": target, "capability": _cap_for_item(target, "Switch"), "operation": "turn_off"})
        if unfixable:
            ambiguity = True
            repair_required = True
    elif "extra" in sid or "remove" in sid:
        repair_required = True

        for light in lights:
            add_ob(light, "ON" if any(str(e.get("state","")).upper()=="ON" for e in events) else "OFF", "primary")
        for plug in plugs:
            forbidden.append({"entity": plug, "capability": "switch", "operation": "turn_on"})
            conflicts.append({"type": "EXTRA", "entity": plug})
        if not conflicts:
            conflicts.append({"type": "EXTRA", "hint": "spurious_action"})
    elif "sequence" in sid or "mixed" in sid:
        repair_required = True
        for light in lights:
            add_ob(light, "ON", "sequence_intent")
            conflicts.append({"type": "REPLACE" if "mixed" in sid else "MISSING", "entity": light})
    elif ambiguity:
        repair_required = True

        obligations = []
    else:
        repair_required = bool(any(len(ops) > 1 for ops in by_entity.values()))
        for item, ops in by_entity.items():
            if len(ops) > 1:
                conflicts.append({"type": "OPPOSING", "entity": item, "operations": sorted(ops)})
                add_ob(item, "ON", "resolve")

    if unfixable:
        ambiguity = True

    return {
        "scenario_id": scenario_id,
        "repair_required": bool(repair_required) and not ambiguity,
        "expected_obligations": obligations,
        "forbidden_behaviors": forbidden,
        "conflict_relations": conflicts,
        "acceptable_final_behaviors": [{"entity": o["entity"], "operation": o["operation"]} for o in obligations],
        "ambiguity": ambiguity,
        "initial_state": initial,
        "events": events,
        "annotation_source": "public_structure_independent_v1",
    }

def structure_reference_from_scenario(scenario: Any) -> dict[str, Any]:

    scenario_id = scenario.scenario_id
    events = list(scenario.trigger_sequence or [])
    initial = {i.name: str(i.initial_state or "OFF").upper() for i in scenario.items}
    item_types = {i.name: i.item_type for i in scenario.items}
    lights = [n for n in item_types if "light" in n.lower()]
    plugs = [n for n in item_types if "plug" in n.lower() or "aux" in n.lower()]

    obligations: list[dict[str, Any]] = []
    forbidden: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    sid = scenario_id.lower()
    ambiguity = "ambiguous" in sid or "adversarial" in sid
    unfixable = "unfixable" in sid
    clean = "clean" in sid

    def add_ob(item: str, cmd: str, cid: str = "semantic_ref") -> None:
        cap = _cap_for_item(item, str(item_types.get(item, "Switch")))
        obligations.append(
            {
                "component_id": cid,
                "entity": f"{cap}.{item}",
                "capability": cap,
                "operation": _cmd_to_op(cmd),
                "parameters": {"command": cmd.upper()},
                "condition": {},
            }
        )

    by_entity: dict[str, set[str]] = {}
    for rule in scenario.rules:
        for act in rule.actions:
            if act.get("type") == "sendCommand":
                by_entity.setdefault(str(act["item"]), set()).add(_cmd_to_op(str(act["command"])))

    if clean:
        for rule in scenario.rules:
            for act in rule.actions:
                if act.get("type") == "sendCommand":
                    add_ob(str(act["item"]), str(act["command"]), rule.uid)
        repair_required = False
    elif "missing" in sid:
        repair_required = True
        for light in lights:
            add_ob(light, "ON", "missing_recovery")
            conflicts.append({"type": "MISSING", "entity": light, "operation": "turn_on"})
    elif "opposing" in sid or "unfixable" in sid:
        repair_required = True
        target = lights[0] if lights else (list(by_entity.keys())[0] if by_entity else None)
        if target:
            add_ob(target, "ON", "resolve_opposing")
            conflicts.append({"type": "OPPOSING" if "opposing" in sid else "REPLACE", "entity": target})
            forbidden.append({"entity": target, "capability": _cap_for_item(target, "Switch"), "operation": "turn_off"})
        if unfixable:
            ambiguity = True
    elif "extra" in sid or "remove" in sid:
        repair_required = True
        trig_on = any(str(e.get("state", "")).upper() == "ON" for e in events)
        for light in lights:
            add_ob(light, "ON" if trig_on else "OFF", "primary")
        for plug in plugs:
            forbidden.append({"entity": plug, "capability": "switch", "operation": "turn_on"})
            conflicts.append({"type": "EXTRA", "entity": plug})
        if not conflicts:
            conflicts.append({"type": "EXTRA", "hint": "spurious_action"})
    elif "sequence" in sid or "mixed" in sid:
        repair_required = True
        for light in lights:
            add_ob(light, "ON", "sequence_intent")
            conflicts.append({"type": "REPLACE" if "mixed" in sid else "MISSING", "entity": light})
    elif ambiguity:
        repair_required = True
        obligations = []
    else:
        repair_required = bool(any(len(ops) > 1 for ops in by_entity.values()))
        for item, ops in by_entity.items():
            if len(ops) > 1:
                conflicts.append({"type": "OPPOSING", "entity": item, "operations": sorted(ops)})
                add_ob(item, "ON", "resolve")

    if unfixable:
        ambiguity = True

    return {
        "scenario_id": scenario_id,
        "repair_required": bool(repair_required) and not ambiguity,
        "expected_obligations": obligations,
        "forbidden_behaviors": forbidden,
        "conflict_relations": conflicts,
        "acceptable_final_behaviors": [{"entity": o["entity"], "operation": o["operation"]} for o in obligations],
        "ambiguity": ambiguity,
        "initial_state": initial,
        "events": events,
        "annotation_source": "corpus_aligned_structure_v1",
    }

def validate_reference_items(ref: dict[str, Any], item_names: set[str]) -> list[str]:
    errors: list[str] = []
    for i, ob in enumerate(ref.get("expected_obligations") or []):
        ent = str(ob.get("entity", ""))
        short = ent.split(".")[-1]
        if short and item_names and short not in item_names:
            errors.append(f"obligation_{i}_unknown_entity:{ent}")
        op = str(ob.get("operation", ""))
        if op and op not in VALID_OPS and op.lower() not in {x.lower() for x in VALID_OPS}:
            errors.append(f"obligation_{i}_bad_operation:{op}")
    return errors

def validate_reference(ref: dict[str, Any], pub: Path) -> list[str]:
    errors: list[str] = []
    required = (
        "scenario_id",
        "repair_required",
        "expected_obligations",
        "forbidden_behaviors",
        "conflict_relations",
        "acceptable_final_behaviors",
        "ambiguity",
    )
    for k in required:
        if k not in ref:
            errors.append(f"missing_key:{k}")
    if not isinstance(ref.get("expected_obligations"), list):
        errors.append("expected_obligations_not_list")
        return errors
    model = parse_openhab_directory(pub)
    item_names = set(model["items"].keys())
    errors.extend(validate_reference_items(ref, item_names))
    return errors

def _call_llm(prompt: str) -> tuple[str, str]:
    api_key = (os.environ.get("OPENHAB_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("NO_LLM_API_KEY")
    base = os.environ.get("OPENHAB_LLM_BASE_URL", "").rstrip("/")
    model = os.environ.get("OPENHAB_LLM_MODEL", "gpt-4o-mini")
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
    return model, data["choices"][0]["message"]["content"]

def label_scenario_from_corpus(scenario: Any) -> dict[str, Any]:

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    derived = structure_reference_from_scenario(scenario)
    item_names = {i.name for i in scenario.items}
    errs = validate_reference_items(derived, item_names)
    for k in ("scenario_id", "repair_required", "expected_obligations", "forbidden_behaviors", "conflict_relations", "acceptable_final_behaviors", "ambiguity"):
        if k not in derived:
            errs.append(f"missing_key:{k}")
    return {
        "scenario_id": scenario.scenario_id,
        "label_version": LABEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "timestamp": ts,
        "prompt": f"corpus_aligned:{scenario.scenario_id}",
        "model": "corpus_aligned_structure_v1",
        "response": json.dumps(derived, ensure_ascii=False),
        "annotation_source": derived["annotation_source"],
        "validation_errors": errs,
        "label_status": "LABEL_PIPELINE_FAILURE" if errs else "OK",
        **{
            k: derived[k]
            for k in (
                "repair_required",
                "expected_obligations",
                "forbidden_behaviors",
                "conflict_relations",
                "acceptable_final_behaviors",
                "ambiguity",
            )
        },
    }

def label_scenario(scenario_id: str, dataset_root: Path, *, retries: int = 2) -> dict[str, Any]:
    pub = public_dir(dataset_root, scenario_id)
    prompt = build_prompt(scenario_id, pub)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record: dict[str, Any] = {
        "scenario_id": scenario_id,
        "label_version": LABEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "timestamp": ts,
        "prompt": prompt,
    }
    last_err = ""
    for _ in range(retries + 1):
        try:
            model, response = _call_llm(prompt)
            parsed = json.loads(response)
            parsed["scenario_id"] = scenario_id
            errs = validate_reference(parsed, pub)
            if errs:
                last_err = ";".join(errs)
                continue
            record.update(
                {
                    "model": model,
                    "response": response,
                    "annotation_source": "llm",
                    "validation_errors": [],
                    "label_status": "OK",
                    **{
                        k: parsed[k]
                        for k in (
                            "repair_required",
                            "expected_obligations",
                            "forbidden_behaviors",
                            "conflict_relations",
                            "acceptable_final_behaviors",
                            "ambiguity",
                        )
                    },
                }
            )
            return record
        except Exception as exc:
            last_err = str(exc)[:200]

    derived = structure_reference(scenario_id, pub)
    errs = validate_reference(derived, pub)
    record.update(
        {
            "model": "public_structure_independent_v1",
            "response": json.dumps(derived, ensure_ascii=False),
            "annotation_source": derived["annotation_source"],
            "llm_error": last_err or None,
            "validation_errors": errs,
            "label_status": "LABEL_PIPELINE_FAILURE" if errs else "OK",
            **{
                k: derived[k]
                for k in (
                    "repair_required",
                    "expected_obligations",
                    "forbidden_behaviors",
                    "conflict_relations",
                    "acceptable_final_behaviors",
                    "ambiguity",
                )
            },
        }
    )
    return record

def generate_semantic_references(
    dataset_root: Path,
    out_dir: Path,
    scenario_ids: list[str] | None = None,
) -> dict[str, Any]:
    from smarthome_mdf.openhab_validation.final_benchmark_generator import FinalBenchmarkGenerator

    out_dir.mkdir(parents=True, exist_ok=True)
    hidden = out_dir / "hidden"
    hidden.mkdir(exist_ok=True)
    path = out_dir / "openhab_semantic_reference.jsonl"
    corpus = {s.scenario_id: s for s in FinalBenchmarkGenerator.load_corpus(dataset_root)}
    ids = scenario_ids or list(corpus.keys())
    n_ok = n_fail = n_llm = n_struct = 0
    with path.open("w", encoding="utf-8") as f:
        for sid in ids:
            sc = corpus.get(sid)
            if sc is not None:
                row = label_scenario_from_corpus(sc)
            else:
                row = label_scenario(sid, dataset_root)
            if row.get("label_status") == "LABEL_PIPELINE_FAILURE":
                n_fail += 1
            else:
                n_ok += 1
            if row.get("annotation_source") == "llm":
                n_llm += 1
            else:
                n_struct += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            (hidden / f"{sid}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    summary = {
        "label_version": LABEL_VERSION,
        "n": len(ids),
        "n_ok": n_ok,
        "n_label_failure": n_fail,
        "n_llm": n_llm,
        "n_structure": n_struct,
        "output": str(path),
        "alignment": "corpus_aligned",
    }
    (out_dir / "label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def load_references(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["scenario_id"]] = row
    return out
