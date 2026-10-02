from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smarthome_mdf.paths import CONFIGS_DIR
from smarthome_mdf.rules.blueprint_action_contract import (
    contract_for_sample,
    get_contract,
    validate_y_annotation,
)

VISUAL_BLUEPRINT_CONSTRAINTS = CONFIGS_DIR / "visual_blueprint_constraints.json"

OBSERVED_FIELD_ALIASES: Dict[str, str] = {
    "person_detected": "person_detected",
    "detection_confidence": "detection_confidence",
    "illuminance": "illuminance",
    "indoor_temperature": "indoor_temperature",
    "power_mean": "power_mean",
    "home_mode": "home_mode",
    "camera": "camera_event",
    "zone": "zone",
}

FRIGATE_NOTIFICATION_TYPES = frozenset({"frigate_notification"})

def load_visual_constraints(path: Optional[Path] = None) -> dict:
    p = path or VISUAL_BLUEPRINT_CONSTRAINTS
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def get_blueprint_constraint(
    blueprint_id: str,
    manifest: Optional[dict] = None,
) -> dict:
    contract = get_contract(blueprint_id)
    if contract:
        return _contract_to_label_constraint(contract)
    manifest = manifest or load_visual_constraints()
    bp = (manifest.get("blueprints") or {}).get(blueprint_id)
    if not bp:
        raise KeyError(f"Unknown visual blueprint_id: {blueprint_id}")
    return bp

def get_label_constraint_for_sample(sample: dict) -> Optional[dict]:

    contract = contract_for_sample(sample)
    if contract:
        return _contract_to_label_constraint(contract)
    return None

def _contract_to_label_constraint(contract: dict) -> dict:

    return {
        "blueprint_id": contract.get("blueprint_id"),
        "blueprint_name": contract.get("blueprint_id"),
        "blueprint_type": contract.get("blueprint_type"),
        "action_domains": list(contract.get("action_domains") or []),
        "has_yaml": contract.get("rule_source") == "yaml",
        "summary_only": contract.get("rule_source") != "yaml",
        "label_confidence_cap": 0.75 if contract.get("rule_source") != "yaml" else 1.0,
        "supported_fields": list((contract.get("gates") or {}).keys()),
        "allowed_actions": list(contract.get("allowed_actions") or []),
        "unsupported_actions": list(contract.get("unsupported_actions") or []),
        "field_policy": {},
        "notes": "Validated via blueprint_action_contracts.json (allowed_actions whitelist)",
        "_action_contract": contract,
    }

def has_out_of_contract_actions(violations: List[str]) -> bool:

    return any(
        v.startswith("not_in_allowed_actions:") or v == "actions_sanitized"
        for v in (violations or [])
    )

def audit_sample_decision_output(sample: dict) -> dict:

    sid = sample.get("sample_id", "")
    scene = sample.get("scene_type", "")
    constraint = get_label_constraint_for_sample(sample)
    if not constraint:
        return {
            "sample_id": sid,
            "scene_type": scene,
            "status": "no_contract",
            "violations": [],
        }

    dec = sample.get("decision_output") or {}
    label_status = dec.get("label_status") or sample.get("metadata", {}).get("label_status") or ""
    exp = list(dec.get("expected_action") or [])
    blk = list(dec.get("blocked_action") or [])

    if label_status not in ("llm", "llm_checked"):
        return {
            "sample_id": sid,
            "scene_type": scene,
            "blueprint_id": constraint.get("blueprint_id"),
            "status": "not_llm",
            "label_status": label_status,
            "expected_action": exp,
            "blocked_action": blk,
            "violations": [],
        }

    if not exp and not blk:
        return {
            "sample_id": sid,
            "scene_type": scene,
            "blueprint_id": constraint.get("blueprint_id"),
            "status": "empty_label",
            "label_status": label_status,
            "violations": [],
        }

    ann = {
        "expected_action": exp,
        "blocked_action": blk,
        "conflict_type": dec.get("conflict_type") or "none",
        "label_confidence": dec.get("label_confidence"),
        "used_blueprint_fields": dec.get("used_blueprint_fields") or [],
        "ignored_non_blueprint_fields": dec.get("ignored_non_blueprint_fields") or [],
        "reason": dec.get("label_rationale") or dec.get("reason") or "",
    }
    contract = constraint["_action_contract"]
    violations, _ = validate_y_annotation(ann, contract, sanitize=False)
    status = "out_of_contract" if has_out_of_contract_actions(violations) else "ok"
    return {
        "sample_id": sid,
        "scene_type": scene,
        "blueprint_id": constraint.get("blueprint_id"),
        "action_domains": list(constraint.get("action_domains") or []),
        "allowed_actions": list(constraint.get("allowed_actions") or []),
        "status": status,
        "label_status": label_status,
        "expected_action": exp,
        "blocked_action": blk,
        "violations": violations,
    }

def visual_y_training_readiness(
    *,
    audit_path: Optional[Path] = None,
    audit_report: Optional[dict] = None,
) -> dict:

    from smarthome_mdf.paths import QUALITY_REPORT_DIR

    report = audit_report
    if report is None:
        path = audit_path or (QUALITY_REPORT_DIR / "llm_contract_audit.json")
        if not path.exists():
            return {
                "ready": False,
                "reason": "audit_file_missing",
                "audit_path": str(path),
            }
        report = json.loads(path.read_text(encoding="utf-8"))
        audit_path = path

    visual = (report.get("scenes") or report.get("by_scene") or {}).get("visual_fusion") or {}
    by_status = visual.get("by_status") or {}
    ooc = int(visual.get("out_of_contract_count") or visual.get("out_of_contract") or 0)
    not_llm = int(by_status.get("not_llm") or 0)
    empty_label = int(by_status.get("empty_label") or 0)
    ok = int(by_status.get("ok") or 0)
    no_contract = int(by_status.get("no_contract") or 0)

    blockers: List[str] = []
    if ooc > 0:
        blockers.append(f"out_of_contract={ooc}")
    if not_llm > 0:
        blockers.append(f"not_llm={not_llm}")

    ready = not blockers
    return {
        "ready": ready,
        "reason": "ok" if ready else "; ".join(blockers),
        "out_of_contract": ooc,
        "not_llm": not_llm,
        "empty_label": empty_label,
        "empty_label_note": (
            "LLM-valid intentional no-action labels; does not block training"
            if empty_label
            else ""
        ),
        "ok": ok,
        "no_contract": no_contract,
        "total": int(visual.get("total") or 0),
        "audit_path": str(audit_path) if audit_path else None,
        "generated_at": report.get("generated_at"),
    }

def blueprint_id_from_sample(sample: dict) -> str:
    binding = sample.get("blueprint_binding") or {}
    bid = binding.get("blueprint_id")
    if not bid:
        raise ValueError(f"sample {sample.get('sample_id')} missing blueprint_binding.blueprint_id")
    return bid

def list_sample_field_keys(sample: dict) -> List[str]:
    keys: List[str] = []
    obs = {**(sample.get("observed") or {}), **(sample.get("derived_observation") or {})}
    for k in obs:
        keys.append(k)
    ss = sample.get("system_state") or {}
    for k in ("home_mode", "manual_override", "schedule_active"):
        if k in ss:
            keys.append(k)
    return sorted(set(keys))

def build_constraint_prompt_block(constraint: dict, manifest: dict) -> str:
    synth = manifest.get("synthetic_extension_fields") or []
    lines = [
        "## Blueprint constraints (MUST obey)",
        f"- blueprint_id: {constraint['blueprint_id']}",
        f"- blueprint_type: {constraint['blueprint_type']}",
        f"- has_yaml: {constraint.get('has_yaml')}",
        f"- summary_only: {constraint.get('summary_only')}",
        "",
        "### Action domains (each action must map to one of these service domains)",
        json.dumps(constraint.get("action_domains") or [], ensure_ascii=False),
        "",
        "### Allowed actions (WHITELIST — only these may appear in expected_action / blocked_action)",
        json.dumps(constraint.get("allowed_actions") or [], ensure_ascii=False),
        "",
        "### Unsupported actions (documentation only; validity is determined solely by Allowed actions above)",
        json.dumps(constraint.get("unsupported_actions") or [], ensure_ascii=False),
        "",
        "### Blueprint-supported fields (may use as primary evidence)",
        json.dumps(constraint.get("supported_fields") or [], ensure_ascii=False),
        "",
        "### Field policy for synthetic extension fields",
        json.dumps(constraint.get("field_policy") or {}, ensure_ascii=False, indent=2),
        "",
        "### Global synthetic extension fields (default: do NOT use as primary decision basis)",
        json.dumps(synth, ensure_ascii=False),
        "",
        "### Rules",
        "- Action validity = action MUST be in Allowed actions (whitelist). Do NOT infer from Unsupported list.",
        "- Do NOT invent HA service names or thresholds not in Blueprint.",
        "- Frigate critical / alarm_stream ≠ alarm_control_panel.trigger; use notify.mobile_app only.",
        "- If Blueprint has no home_mode condition, home_mode must NOT dominate notify/light/climate.",
        "- If Blueprint has no confidence threshold, detection_confidence must NOT trigger alarm tiers (0.8/0.9).",
        "- If Blueprint is not lighting-type, illuminance must NOT justify light.turn_on.",
        "- If insufficient evidence, output expected_action=[] and conflict_type=uncertain, label_confidence≤0.5.",
        f"- Max label_confidence when summary_only: {constraint.get('label_confidence_cap', 1.0)}",
        "",
        f"Notes: {constraint.get('notes', '')}",
    ]
    return "\n".join(lines)

CONTRACT_CONSTRAINED_SYSTEM_PROMPT = """You are a Home Assistant Blueprint Action Contract annotator.

Output ONLY valid JSON with this exact schema:
{
  "expected_action": ["service.action", ...],
  "blocked_action": ["service.action", ...],
  "conflict_type": "none",
  "label_confidence": 0.0,
  "used_blueprint_fields": ["field_name", ...],
  "ignored_non_blueprint_fields": ["field_name", ...],
  "reason": "short explanation"
}

Rules:
- expected_action / blocked_action MUST use ONLY actions from the user message "Allowed actions" whitelist.
- Each action must belong to one of the listed "Action domains" via its service prefix.
- If an action is not in Allowed actions, do NOT output it (use [] instead).
- "Unsupported actions" is reference only; whitelist Allowed actions is the sole validity rule.
- NEVER output alarm_control_panel.trigger unless it appears in Allowed actions.
- security_alert conflict_type means security priority notification, NOT alarm_control_panel.trigger.
- partial_constraint_conflict is for multi-action partial blocks only (usually not single-action visual).
- valid conflict_type: none, manual_override, schedule_constraint, energy_saving, comfort_vs_energy, security_alert, partial_constraint_conflict, uncertain.
- List which Blueprint-supported fields you actually used vs ignored synthetic fields.
"""

VISUAL_CONSTRAINED_SYSTEM_PROMPT = CONTRACT_CONSTRAINED_SYSTEM_PROMPT

def validate_llm_annotation(
    ann: dict,
    constraint: dict,
    sample: Optional[dict] = None,
    *,
    strict_field_usage: bool = False,
) -> Tuple[List[str], dict]:

    contract = constraint.get("_action_contract") or get_contract(constraint.get("blueprint_id", ""))
    if contract:
        violations, out = validate_y_annotation(ann, contract, sanitize=True)

        if strict_field_usage and sample:
            used = set(out.get("used_blueprint_fields") or [])
            field_policy = constraint.get("field_policy") or {}
            forbidden_dominant = {
                k for k, v in field_policy.items() if "forbidden_dominant" in str(v)
            }
            if forbidden_dominant & used:
                violations.append(f"used_forbidden_dominant_fields:{sorted(forbidden_dominant & used)}")
        valid_ct = set(load_visual_constraints().get("valid_conflict_types") or [])
        ct = out.get("conflict_type") or "none"
        if ct not in valid_ct:
            violations.append(f"invalid_conflict_type:{ct}")
            out["conflict_type"] = "uncertain"
        cap = float(constraint.get("label_confidence_cap") or 1.0)
        if "label_confidence" in out and out["label_confidence"] is not None:
            conf = float(out.get("label_confidence") or 0.0)
            if conf > cap:
                violations.append(f"label_confidence_exceeds_cap:{conf}>{cap}")
                out["label_confidence"] = cap
        out.setdefault("used_blueprint_fields", [])
        out.setdefault("ignored_non_blueprint_fields", [])
        out.setdefault("reason", out.pop("rationale", "") or "")
        return violations, out

    violations: List[str] = []
    out = dict(ann)
    allowed = set(constraint.get("allowed_actions") or [])
    valid_ct = set(load_visual_constraints().get("valid_conflict_types") or [])

    exp = list(out.get("expected_action") or [])
    blk = list(out.get("blocked_action") or [])
    all_actions = set(exp) | set(blk)

    for a in all_actions:
        if not allowed or a not in allowed:
            violations.append(f"not_in_allowed_actions:{a}")

    ct = out.get("conflict_type") or "none"
    if ct not in valid_ct:
        violations.append(f"invalid_conflict_type:{ct}")
        out["conflict_type"] = "uncertain"

    cap = float(constraint.get("label_confidence_cap") or 1.0)
    if "label_confidence" in out and out["label_confidence"] is not None:
        conf = float(out.get("label_confidence") or 0.0)
        if conf > cap:
            violations.append(f"label_confidence_exceeds_cap:{conf}>{cap}")
            out["label_confidence"] = cap

    used = set(out.get("used_blueprint_fields") or [])
    supported = set(constraint.get("supported_fields") or [])
    field_policy = constraint.get("field_policy") or {}
    forbidden_dominant = {
        k for k, v in field_policy.items()
        if "forbidden_dominant" in str(v)
    }

    if strict_field_usage and sample and forbidden_dominant & used:
        violations.append(f"used_forbidden_dominant_fields:{sorted(forbidden_dominant & used)}")

    if strict_field_usage and supported and used and not (used & supported):
        violations.append("used_blueprint_fields_not_in_supported_set")

    exp_clean = [a for a in exp if allowed and a in allowed]
    blk_clean = [a for a in blk if allowed and a in allowed]
    if exp_clean != exp or blk_clean != blk:
        violations.append("actions_sanitized")
    out["expected_action"] = exp_clean
    out["blocked_action"] = blk_clean

    out.setdefault("used_blueprint_fields", [])
    out.setdefault("ignored_non_blueprint_fields", [])
    out.setdefault("reason", out.pop("rationale", "") or "")

    return violations, out

def audit_existing_label(sample: dict, manifest: Optional[dict] = None) -> dict:

    manifest = manifest or load_visual_constraints()
    bid = blueprint_id_from_sample(sample)
    constraint = get_blueprint_constraint(bid, manifest)
    dec = sample.get("decision_output") or {}
    ann = {
        "expected_action": dec.get("expected_action") or [],
        "blocked_action": dec.get("blocked_action") or [],
        "conflict_type": dec.get("conflict_type") or "none",
        "label_confidence": dec.get("label_confidence"),
        "used_blueprint_fields": dec.get("used_blueprint_fields") or [],
        "ignored_non_blueprint_fields": dec.get("ignored_non_blueprint_fields") or [],
        "reason": dec.get("label_rationale") or dec.get("reason") or "",
    }
    violations, _ = validate_llm_annotation(ann, constraint, sample)
    return {
        "sample_id": sample.get("sample_id"),
        "blueprint_id": bid,
        "blueprint_type": constraint.get("blueprint_type"),
        "violations": violations,
        "expected_action": ann["expected_action"],
    }
