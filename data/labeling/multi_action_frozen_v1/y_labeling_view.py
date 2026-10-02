from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.y_llm_config import (
    Y_LLM_SAMPLE_MAX_ATTEMPTS,
    Y_LLM_TEMPERATURE,
    get_y_llm_config,
    y_llm_configured,
)
from smarthome_mdf.v4_synthesis.llm_schema import LLMClient
from smarthome_mdf.multi_action_frozen_v1.config import (
    ACTION_SCHEMA_VERSION,
    EXPECTED_MULTI_ACTION_FROZEN_SHA256,
    Y_ADAPTER_VERSION,
    Y_PROMPT_VERSION,
)
from smarthome_mdf.multi_action_vnext.action_schema import ActionVNext, from_raw_action
from smarthome_mdf.multi_action_frozen_v1.y_canonical_semantics import build_canonical_blueprint_semantics
from smarthome_mdf.multi_action_frozen_v1.y_compact_prompt import (
    FROZEN_INDEPENDENT_Y_V5_SYSTEM_PROMPT,
    build_compact_ma_user_prompt,
)
from smarthome_mdf.multi_action_frozen_v1.y_service_target_contract import (
    canonicalize_service,
    classify_action_role,
    service_requires_target,
)
from smarthome_mdf.multi_action_frozen_v1.y_yaml_services import static_yaml_service_list
from smarthome_mdf.rules.blueprint_action_contract import get_contract
from smarthome_mdf.semantic_alignment.action_parser import parse_raw_action
from smarthome_mdf.synthesis.multi_action.vocab import load_vocab

Y_FORBIDDEN_PROMPT_TOKENS = (
    "parent_b0",
    "execute_parent_b0",
    "compute_parent_b0",
    "decision_output",
    "current_expected_action_ha",
    "current_blocked_action_ha",
    "component_oracle_traces",
    "local_expected_actions",
    "conflict_label",
    "repair_actions",
    "repair_hints",
    "ars_evaluator",
    "service_level_view",
    "get_v3_b0_actions",
    "branch_id",
    "behavior_target",
    "expected_service",
    "expected_action",
    "component_action_schema",
    "action_path_signature",
    "strict_path_key",
    "multi_action_gt",
    "formal_b0",
    "semantic_b0_actions",
    "selected_runtime_path",
    "conflict",
    "repair",
)

Y_FORBIDDEN_VIEW_KEYS = frozenset(
    {
        "component_action_schema",
        "behavior_target",
        "expected_action",
        "expected_service",
        "action_path_signature",
        "strict_path_key",
        "multi_action_gt",
        "branch_id",
        "parent_b0",
        "semantic_b0_actions",
        "selected_runtime_path",
        "formal_b0",
        "b0",
        "b0_actions",
        "decision_output",
        "component_oracle_traces",
        "local_expected_actions",
        "conflict_label",
        "repair_actions",
        "repair_hints",
    }
)

from smarthome_mdf.multi_action_frozen_v1.y_forced_decision import (
    FORCED_DECISION_RETRY_INSTRUCTION,
    FORMAL_Y_DECISIONS,
    INVALID_FORMAL_Y_DECISIONS,
    FormalYNormalizationResult,
    Y_SEMANTIC_RETRYABLE_REASONS,
    audit_confidence_metadata,
)

Y_FORBIDDEN_IMPORTS = (
    "smarthome_mdf.multi_action_vnext.parent_executor",
    "smarthome_mdf.multi_action_vnext.parent_b0",
    "smarthome_mdf.multi_action_vnext.gt_builder",
    "smarthome_mdf.multi_action_frozen_v1.b0_adapter",
)

FROZEN_INDEPENDENT_Y_SYSTEM_PROMPT = FROZEN_INDEPENDENT_Y_V5_SYSTEM_PROMPT

@dataclass
class StructuredYAction:
    service: str
    target_entity: str | None
    parameters: dict[str, Any]
    component_origin: str | None
    execution_order: int
    action_role: str = "business_device_action"
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def prompt_version_hash() -> str:
    payload = FROZEN_INDEPENDENT_Y_SYSTEM_PROMPT + Y_PROMPT_VERSION
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

def extract_static_yaml_services(blueprint_id: str) -> list[str]:

    return static_yaml_service_list(blueprint_id)

def _static_blueprint_allowed_actions(blueprint_id: str) -> list[str]:
    contract = get_contract(blueprint_id) or {}
    contract_allowed = set(contract.get("allowed_actions") or [])
    yaml_services = set(static_yaml_service_list(blueprint_id))
    return sorted(contract_allowed | yaml_services)

def _entity_catalog_for_component(comp: dict[str, Any]) -> list[dict[str, str]]:
    catalog: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    cid = str(comp.get("component_id") or "")

    def _add(role: str, ent: str) -> None:
        key = (cid, ent)
        if ent and key not in seen:
            seen.add(key)
            catalog.append({"role": str(role), "entity_id": str(ent), "component_origin": cid})

    eb = comp.get("entity_binding") or {}
    for rec in eb.get("records") or []:
        ent = rec.get("component_local_entity") or rec.get("parent_entity") or rec.get("original_entity")
        role = rec.get("role") or rec.get("entity_role") or "entity"
        _add(str(role), str(ent) if ent else "")
    for role, ent in (eb.get("component_entity_map") or {}).items():
        _add(str(role), str(ent))
    return catalog

def _entity_candidates_for_service(service: str, comp: dict[str, Any]) -> list[str]:
    catalog = _entity_catalog_for_component(comp)
    eb = comp.get("entity_binding") or {}
    entity_map = eb.get("component_entity_map") or {}

    role_hints = {
        "light.turn_on": ("light_target", "target_light", "light", "lights"),
        "light.turn_off": ("light_target", "target_light", "light", "lights"),
        "scene.turn_on": ("scene", "scene_entity", "target_scene"),
        "climate.set_hvac_mode": ("climate_entity", "climate", "hvac"),
        "switch.turn_on": ("switch", "entity", "target"),
        "switch.turn_off": ("switch", "entity", "target"),
    }
    for role in role_hints.get(service, ()):
        ent = entity_map.get(role)
        if isinstance(ent, str) and ent:
            return [ent]

    if not service or "." not in service:
        return [e["entity_id"] for e in catalog]
    domain = service.split(".", 1)[0]
    matched = [e["entity_id"] for e in catalog if e["entity_id"].startswith(f"{domain}.")]
    if matched:
        return sorted(set(matched))
    action_domains = {
        "light.turn_on": "light.",
        "light.turn_off": "light.",
        "climate.set_hvac_mode": "climate.",
        "switch.turn_on": "switch.",
        "switch.turn_off": "switch.",
    }
    prefix = action_domains.get(service)
    if prefix:
        matched = [e["entity_id"] for e in catalog if e["entity_id"].startswith(prefix)]
        if matched:
            return sorted(set(matched))
    return sorted({e["entity_id"] for e in catalog})

def _static_blueprint_action_domain(blueprint_id: str) -> list[dict[str, Any]]:
    contract = get_contract(blueprint_id) or {}
    contract_services = set(contract.get("allowed_actions") or [])
    yaml_services = set(static_yaml_service_list(blueprint_id))
    services = sorted(contract_services | yaml_services)
    domains = list(contract.get("action_domains") or [])
    return [
        {
            "service": svc,
            "action_domain": domains[i % len(domains)] if domains else svc.split(".")[0] if "." in svc else "",
            "parameter_schema": contract.get("parameter_schema") or {},
            "source": (
                "contract+yaml"
                if svc in contract_services and svc in yaml_services
                else "contract_only"
                if svc in contract_services
                else "yaml_only"
            ),
        }
        for i, svc in enumerate(services)
    ]

def build_candidate_actions_ha(components: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:

    per_component: dict[str, list[str]] = {}
    per_component_domain: dict[str, list[dict]] = {}
    union: set[str] = set()
    sources: dict[str, str] = {}

    for comp in components:
        cid = str(comp.get("component_id") or "")
        bid = str(comp.get("blueprint_id") or "")
        domain_entries = _static_blueprint_action_domain(bid)
        static = {e["service"] for e in domain_entries}
        per_component[cid] = sorted(static)
        per_component_domain[cid] = domain_entries
        union |= static
        sources[cid] = "blueprint_contract_union_yaml_static"

    composite = get_contract("multi_action_composite") or {}
    composite_union = sorted(set(composite.get("allowed_actions") or []) | union)

    audit = {
        "construction_rule": (
            "Per-component allowed_actions = contract.allowed_actions UNION static YAML services. "
            "multi_action_composite provides audit union only — does NOT filter components. "
            "Forbidden inputs: formal executor output, oracle traces, prior labels, construction hints."
        ),
        "per_component": per_component,
        "per_component_domain": per_component_domain,
        "composite_audit_union": composite_union,
        "sources": sources,
        "uses_b0": False,
        "uses_oracle": False,
        "composite_hard_filter": False,
    }
    return sorted(union), audit

def _ha_device_group_map(candidate_ha: list[str]) -> dict[str, str]:
    vocab, _, _ = load_vocab()
    ha_to_group: dict[str, str] = {}
    for _abs, meta in vocab.items():
        ha = meta.get("ha_action")
        if ha and ha in candidate_ha:
            ha_to_group[ha] = meta.get("device_group", "")
    return ha_to_group

def _sanitize_evidence_dict(obj: Any) -> Any:

    if isinstance(obj, dict):
        return {
            k: _sanitize_evidence_dict(v)
            for k, v in obj.items()
            if str(k) not in Y_FORBIDDEN_VIEW_KEYS
        }
    if isinstance(obj, list):
        return [_sanitize_evidence_dict(x) for x in obj]
    return obj

def build_evidence_from_components(components: list[dict[str, Any]]) -> dict[str, Any]:
    per_component: dict[str, Any] = {}
    for comp in components:
        cid = str(comp.get("component_id") or "")
        rs = comp.get("runtime_state") or {}
        per_component[cid] = _sanitize_evidence_dict(
            {
                "observed": rs.get("observed"),
                "system_state": rs.get("system_state"),
                "entity_observations": rs.get("entity_observations"),
            }
        )
    return {"per_component": per_component}

def _runtime_for_y(comp: dict[str, Any], ss: dict[str, Any] | None = None) -> dict[str, Any]:

    if ss:
        return _sanitize_evidence_dict(
            {
                "observed": ss.get("observed") or {},
                "system_state": ss.get("system_state") or {},
                "entity_observations": ss.get("entity_observations") or [],
            }
        )
    rs = comp.get("runtime_state") or {}
    return _sanitize_evidence_dict(
        {
            "observed": rs.get("observed"),
            "system_state": rs.get("system_state"),
            "entity_observations": rs.get("entity_observations"),
        }
    )

def _component_blueprint_inputs(comp: dict[str, Any], ss: dict[str, Any]) -> dict[str, Any]:
    eb = comp.get("entity_binding") or {}
    prov = comp.get("source_provenance") or {}
    gi = (ss.get("blueprint_binding") or {}).get("grounded_instance") or {}
    for src in (comp.get("blueprint_inputs"), eb.get("blueprint_inputs"), gi.get("blueprint_inputs"), prov.get("blueprint_inputs")):
        if isinstance(src, dict) and src:
            return dict(src)
    return {}

def build_y_labeling_view(ma: dict[str, Any], frozen_index: dict[str, dict[str, Any]]) -> dict[str, Any]:

    components: list[dict[str, Any]] = []
    canonical_semantics: dict[str, Any] = {}
    for comp in ma.get("components") or []:
        ss = frozen_index.get(str(comp.get("single_scene_sample_id") or ""), {})
        bid = str(comp.get("blueprint_id") or "")
        bp_inputs = _component_blueprint_inputs(comp, ss)
        if bid and bid not in canonical_semantics:
            canonical_semantics[bid] = build_canonical_blueprint_semantics(bid, bp_inputs)
        components.append(
            {
                "component_id": comp.get("component_id"),
                "blueprint_id": bid,
                "scene": comp.get("scene"),
                "blueprint_inputs": bp_inputs,
                "entity_binding": comp.get("entity_binding"),
                "temporal_alignment": comp.get("temporal_alignment"),
                "runtime_state": _runtime_for_y(comp, ss if ss else None),
                "single_scene_sample_id": comp.get("single_scene_sample_id"),
            }
        )

    candidate_ha, candidate_audit = build_candidate_actions_ha(components)
    entity_catalog = []
    for comp in components:
        entity_catalog.extend(_entity_catalog_for_component(comp))

    return {
        "multi_action_id": ma.get("multi_action_id"),
        "component_count": ma.get("component_count"),
        "composition_type": ma.get("composition_type"),
        "source_mode": ma.get("source_mode"),
        "composition_timeline": ma.get("composition_timeline"),
        "shared_entities": ma.get("shared_entities"),
        "components": components,
        "canonical_blueprint_semantics": canonical_semantics,
        "candidate_actions_ha": candidate_ha,
        "candidate_action_domain": candidate_audit.get("per_component_domain") or {},
        "entity_catalog": entity_catalog,
        "candidate_actions_audit": candidate_audit,
        "evidence": build_evidence_from_components(components),
        "multi_action_frozen_corpus_sha256": EXPECTED_MULTI_ACTION_FROZEN_SHA256,
        "labeling_mode": "independent_frozen_y_v5_forced_decision",
    }

def build_frozen_y_user_prompt(view: dict[str, Any], sample: dict[str, Any] | None = None) -> str:

    samples_by_comp: dict[str, dict] = {}
    if sample:
        samples_by_comp["c0"] = sample
    else:
        for comp in view.get("components") or []:
            cid = str(comp.get("component_id") or "")
            rs = comp.get("runtime_state") or {}
            if rs:
                samples_by_comp[cid] = {
                    "observed": rs.get("observed") or {},
                    "entity_observations": rs.get("entity_observations") or [],
                }
    return build_compact_ma_user_prompt(view, samples_by_comp or None)

def _audit_view_object(obj: Any, path: str = "") -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            key_path = f"{path}.{key}" if path else str(key)
            if str(key) in Y_FORBIDDEN_VIEW_KEYS:
                violations.append({"path": key_path, "key": str(key), "kind": "view_key"})
            lower_key = str(key).lower()
            for tok in Y_FORBIDDEN_PROMPT_TOKENS:
                if tok.lower() == lower_key:
                    violations.append({"path": key_path, "key": str(key), "kind": "forbidden_key_name"})
            violations.extend(_audit_view_object(val, key_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(_audit_view_object(item, f"{path}[{i}]"))
    elif isinstance(obj, str):
        lower = obj.lower()
        for tok in Y_FORBIDDEN_PROMPT_TOKENS:
            if tok.lower() in lower:
                violations.append({"path": path, "token": tok, "kind": "view_string"})
    return violations

def audit_y_view(view: dict[str, Any]) -> dict[str, Any]:
    violations = _audit_view_object(view)
    return {
        "Y_VIEW_INFORMATION_FLOW_VIOLATION": len(violations),
        "violations": violations,
    }

def audit_y_prompt(prompt_text: str) -> dict[str, Any]:
    lower = prompt_text.lower()
    violations = [tok for tok in Y_FORBIDDEN_PROMPT_TOKENS if tok.lower() in lower]
    return {
        "Y_PROMPT_INFORMATION_FLOW_VIOLATION": len(violations),
        "violations": violations,
    }

def audit_candidate_actions_independence(candidate_audit: dict[str, Any]) -> dict[str, Any]:
    leakage = 0
    if candidate_audit.get("uses_b0"):
        leakage += 1
    if candidate_audit.get("uses_oracle"):
        leakage += 1
    return {"Y_CANDIDATE_ACTION_LEAKAGE": leakage, "audit": candidate_audit}

def _entity_candidates_for_component(comp: dict[str, Any]) -> list[str]:

    catalog = _entity_catalog_for_component(comp)
    return sorted({e["entity_id"] for e in catalog if e.get("entity_id")})

def _component_allowed_services(view: dict[str, Any], component_origin: str) -> set[str]:
    domain = (view.get("candidate_action_domain") or {}).get(component_origin) or []
    if isinstance(domain, list) and domain and isinstance(domain[0], dict):
        return {str(e.get("service")) for e in domain if e.get("service")}
    per = (view.get("candidate_actions_audit") or {}).get("per_component") or {}
    return set(per.get(component_origin) or [])

def _catalog_entities_for_component(view: dict[str, Any], component_origin: str) -> set[str]:
    return {
        str(e.get("entity_id"))
        for e in (view.get("entity_catalog") or [])
        if str(e.get("component_origin") or "") == str(component_origin) and e.get("entity_id")
    }

def _match_components_for_service(service: str, components: list[dict[str, Any]]) -> list[str]:
    matched: list[str] = []
    for comp in components:
        bid = str(comp.get("blueprint_id") or "")
        allowed = set(_static_blueprint_allowed_actions(bid))
        if service in allowed:
            matched.append(str(comp.get("component_id") or ""))
    return sorted(set(matched))

def normalize_formal_y(parsed: dict[str, Any], view: dict[str, Any]) -> FormalYNormalizationResult:

    components = view.get("components") or []
    decision = str(parsed.get("decision") or "").upper().strip()
    ambiguities: list[dict[str, str]] = []

    if not decision:
        return FormalYNormalizationResult(
            [], "Y_NORMALIZATION_FAIL", [{"reason": "missing_decision"}], None, formal_y_valid=False
        )

    if decision in INVALID_FORMAL_Y_DECISIONS or parsed.get("valid_multi_action") is False:
        return FormalYNormalizationResult(
            [],
            "Y_NORMALIZATION_FAIL",
            [{"reason": "INVALID_FORMAL_Y_DECISION", "decision": decision}],
            None,
            formal_y_valid=False,
        )

    if decision not in FORMAL_Y_DECISIONS:
        return FormalYNormalizationResult(
            [],
            "Y_NORMALIZATION_FAIL",
            [{"reason": "INVALID_FORMAL_Y_DECISION", "decision": decision}],
            None,
            formal_y_valid=False,
        )

    confidence, low_confidence, alternatives, uncertainty_reason, conf_violations = audit_confidence_metadata(
        parsed, decision
    )
    ambiguities.extend(conf_violations)

    if decision == "NO_ACTION":
        if list(parsed.get("actions") or []):
            return FormalYNormalizationResult(
                [],
                "Y_NORMALIZATION_FAIL",
                ambiguities + [{"reason": "NONEMPTY_ACTIONS_FOR_NO_ACTION"}],
                None,
                label_confidence=confidence,
                low_confidence=low_confidence,
                alternative_hypotheses=alternatives,
                uncertainty_reason=uncertainty_reason,
                formal_y_valid=False,
            )
        if any(v.get("reason") == "ARGMAX_INCONSISTENCY" for v in conf_violations):
            return FormalYNormalizationResult(
                [],
                "Y_NORMALIZATION_FAIL",
                ambiguities,
                None,
                label_confidence=confidence,
                low_confidence=low_confidence,
                alternative_hypotheses=alternatives,
                uncertainty_reason=uncertainty_reason,
                formal_y_valid=False,
            )
        return FormalYNormalizationResult(
            [],
            "Y_NORMALIZATION_SUCCESS",
            [],
            "NO_ACTION",
            label_confidence=confidence,
            low_confidence=low_confidence,
            alternative_hypotheses=alternatives,
            uncertainty_reason=uncertainty_reason,
            formal_y_valid=True,
        )

    actions_in = list(parsed.get("actions") or [])
    if not actions_in:
        return FormalYNormalizationResult(
            [],
            "Y_NORMALIZATION_FAIL",
            ambiguities + [{"reason": "EMPTY_ACTIONS_FOR_ACTIONS"}],
            None,
            label_confidence=confidence,
            low_confidence=low_confidence,
            alternative_hypotheses=alternatives,
            uncertainty_reason=uncertainty_reason,
            formal_y_valid=False,
        )

    if any(v.get("reason") == "ARGMAX_INCONSISTENCY" for v in conf_violations):
        return FormalYNormalizationResult(
            [],
            "Y_NORMALIZATION_FAIL",
            ambiguities,
            None,
            label_confidence=confidence,
            low_confidence=low_confidence,
            alternative_hypotheses=alternatives,
            uncertainty_reason=uncertainty_reason,
            formal_y_valid=False,
        )

    structured: list[StructuredYAction] = []
    hard_fail = False

    for raw in actions_in:
        if isinstance(raw, str):
            pa = parse_raw_action(raw)
            raw = {
                "service": pa.get("service"),
                "target_entity": pa.get("entity_id"),
                "parameters": pa.get("parameters") or {},
            }
        service = canonicalize_service(str(raw.get("service") or ""))
        target = raw.get("target_entity") or raw.get("target") or raw.get("entity_id")
        while isinstance(target, dict):
            target = target.get("entity_id") or target.get("entity") or target.get("id")
            break
        if isinstance(target, list) and target:
            first = target[0]
            target = first.get("entity_id") if isinstance(first, dict) else first
        params = dict(raw.get("parameters") or raw.get("data") or {})
        component_origin = raw.get("component_origin")
        order = int(raw.get("execution_order") or len(structured) + 1)
        target_resolution = ""
        if service and not service_requires_target(service):
            target = None

        if not service:
            return FormalYNormalizationResult(
                [s.to_dict() for s in structured],
                "Y_NORMALIZATION_FAIL",
                ambiguities + [{"reason": "missing_service"}],
                None,
                label_confidence=confidence,
                formal_y_valid=False,
            )

        if not component_origin:
            comp_origins = _match_components_for_service(service, components)
            if len(comp_origins) == 1:
                component_origin = comp_origins[0]
            elif len(comp_origins) > 1:
                hard_fail = True
                ambiguities.append(
                    {"service": service, "reason": "ambiguous_component_origin", "components": ",".join(comp_origins)}
                )
            else:
                hard_fail = True
                ambiguities.append({"service": service, "reason": "missing_component_origin"})

        if component_origin:
            allowed = _component_allowed_services(view, str(component_origin))
            if allowed and service not in allowed:
                hard_fail = True
                ambiguities.append(
                    {
                        "service": service,
                        "component_origin": str(component_origin),
                        "reason": "SERVICE_OUT_OF_COMPONENT_DOMAIN",
                    }
                )

        if component_origin and target:
            catalog_ents = _catalog_entities_for_component(view, str(component_origin))
            if catalog_ents and str(target) not in catalog_ents:
                hard_fail = True
                ambiguities.append(
                    {
                        "service": service,
                        "target_entity": str(target),
                        "reason": "TARGET_OUT_OF_ENTITY_CATALOG",
                    }
                )

        if not target and component_origin:
            comp = next((c for c in components if str(c.get("component_id")) == str(component_origin)), {})
            ents = _entity_candidates_for_service(service, comp)
            if len(ents) == 1:
                target = ents[0]
                target_resolution = "UNIQUE_ENTITY_INFERENCE"
            elif len(ents) > 1 and service_requires_target(service):
                hard_fail = True
                ambiguities.append(
                    {"service": service, "reason": "multiple_entity_candidates", "candidates": ",".join(ents)}
                )

        if not target and service_requires_target(service):
            hard_fail = True
            ambiguities.append({"service": service, "reason": "missing_target_entity"})

        role = classify_action_role(service)
        av = from_raw_action(
            {"service": service, "target_entity": target, "parameters": params},
            component_origin=str(component_origin or ""),
            execution_order=order,
        )
        provenance = {
            "raw_llm_action": json.dumps(raw, ensure_ascii=False),
            "normalization": "formal_y2",
        }
        if target_resolution:
            provenance["target_resolution"] = target_resolution
        structured.append(
            StructuredYAction(
                service=av.service,
                target_entity=av.target_entity,
                parameters=av.parameters,
                component_origin=str(component_origin) if component_origin else None,
                execution_order=order,
                action_role=role,
                provenance=provenance,
            )
        )

    if hard_fail:
        return FormalYNormalizationResult(
            [s.to_dict() for s in structured],
            "Y_NORMALIZATION_FAIL",
            ambiguities,
            None,
            label_confidence=confidence,
            low_confidence=low_confidence,
            alternative_hypotheses=alternatives,
            uncertainty_reason=uncertainty_reason,
            formal_y_valid=False,
        )

    return FormalYNormalizationResult(
        [s.to_dict() for s in structured],
        "Y_NORMALIZATION_SUCCESS",
        [],
        "ACTIONS",
        label_confidence=confidence,
        low_confidence=low_confidence,
        alternative_hypotheses=alternatives,
        uncertainty_reason=uncertainty_reason,
        formal_y_valid=True,
    )

def normalize_structured_y(
    parsed: dict[str, Any],
    view: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, list[dict[str, str]], str]:

    result = normalize_formal_y(parsed, view)
    return result.to_legacy_tuple()

def _parse_llm_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None

def _retry_sleep_seconds(exc: BaseException, attempt: int) -> float:
    cfg = get_y_llm_config()
    if "HTTP 429" in str(exc) or "429" in str(exc):
        return min(30.0 * attempt, 90.0)
    return float(cfg.get("retry_sleep_sec") or 5.0)

def _llm_credential_available() -> bool:
    return y_llm_configured()

def _make_llm_client() -> tuple[LLMClient, str]:
    cfg = get_y_llm_config()
    client = LLMClient(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
    )
    return client, str(cfg["model"])

def _semantic_retryable(result: FormalYNormalizationResult, parsed: dict | None) -> bool:
    if parsed is None:
        return True
    if not result.formal_y_valid:
        for row in result.ambiguities:
            reason = str(row.get("reason") or "")
            if reason in Y_SEMANTIC_RETRYABLE_REASONS:
                return True
        if result.normalization_status == "Y_NORMALIZATION_FAIL":
            return True
    return False

def _build_semantic_retry_user_prompt(
    base_prompt: str,
    result: FormalYNormalizationResult,
    *,
    forced_decision: bool = False,
) -> str:
    reasons = sorted({str(a.get("reason")) for a in result.ambiguities if a.get("reason")})
    if not reasons and result.normalization_status:
        reasons = [result.normalization_status]
    payload = {"normalization_status": result.normalization_status, "reasons": reasons}
    suffix = "\n\n## Validation failure (fix your JSON; do not guess answers)\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    if forced_decision or any(
        r in ("INVALID_FORMAL_Y_DECISION", "ARGMAX_INCONSISTENCY") for r in reasons
    ):
        suffix += "\n\n" + FORCED_DECISION_RETRY_INSTRUCTION
    return base_prompt + suffix

def _formal_y_record_fields(result: FormalYNormalizationResult) -> dict[str, Any]:
    return {
        "label_confidence": result.label_confidence,
        "low_confidence": result.low_confidence,
        "alternative_hypotheses": result.alternative_hypotheses,
        "uncertainty_reason": result.uncertainty_reason,
        "formal_y_valid": result.formal_y_valid,
    }

def label_one_independent_y(
    view: dict[str, Any],
    *,
    record_id_key: str = "multi_action_id",
    system_prompt: str = FROZEN_INDEPENDENT_Y_SYSTEM_PROMPT,
    user_prompt: str | None = None,
    client: Any | None = None,
    max_attempts: int = Y_LLM_SAMPLE_MAX_ATTEMPTS,
    semantic_max_retries: int = 2,
    development_pilot_only: bool = False,
) -> dict[str, Any]:

    record_id = view.get(record_id_key)
    user_prompt = user_prompt or build_frozen_y_user_prompt(view)
    view_audit = audit_y_view(view)
    prompt_audit = audit_y_prompt(user_prompt + system_prompt)
    prompt_audit["Y_VIEW_INFORMATION_FLOW_VIOLATION"] = view_audit["Y_VIEW_INFORMATION_FLOW_VIOLATION"]
    prompt_audit["view_violations"] = view_audit.get("violations") or []

    if not _llm_credential_available():
        return {
            record_id_key: record_id,
            "development_pilot_only": development_pilot_only,
            "LLM_CALL_SUCCESS": False,
            "LLM_CALL_FAIL": True,
            "LLM_PARSE_SUCCESS": False,
            "LLM_PARSE_FAIL": False,
            "label_status": "LLM_CALL_FAIL",
            "label_decision": None,
            "parser_status": "skipped",
            "normalization_status": "skipped",
            "raw_llm_output": None,
            "normalized_structured_y": [],
            "error": "LLM credential not configured — edit src/smarthome_mdf/multi_action_frozen_v1/y_llm_config.py (Y_LLM_API_KEY)",
            "model": get_y_llm_config()["model"],
            "prompt_version": Y_PROMPT_VERSION,
            "prompt_hash": prompt_version_hash(),
            "temperature": Y_LLM_TEMPERATURE,
            "retry_count": 0,
            "semantic_retry_count": 0,
            "prompt_audit": prompt_audit,
            "view_audit": view_audit,
        }

    model_name = get_y_llm_config()["model"]
    if client is None:
        client, model_name = _make_llm_client()
    elif hasattr(client, "model") and getattr(client, "model", None):
        model_name = str(getattr(client, "model"))

    base_user_prompt = user_prompt
    current_user_prompt = user_prompt
    raw: str | None = None
    parsed_direct: dict[str, Any] | None = None
    retry_count = 0
    semantic_retry_count = 0
    last_err: str | None = None

    for attempt in range(1, max_attempts + 1):
        retry_count = attempt - 1
        try:
            resp = client.chat(system=system_prompt, user=current_user_prompt)
            if isinstance(resp, dict):
                raw = resp.get("raw_response") or resp.get("content")
                if resp.get("parsed_response") and isinstance(resp["parsed_response"], dict):
                    parsed_direct = resp["parsed_response"]
            else:
                raw = str(resp)
            last_err = None
        except Exception as exc:
            last_err = str(exc)[:300]
            parsed_direct = None
            raw = None
            if attempt >= max_attempts:
                break
            _retry_sleep_seconds(attempt, exc)
            continue

        parsed = parsed_direct or _parse_llm_json(raw or "")
        if parsed is None:
            if semantic_retry_count < semantic_max_retries:
                semantic_retry_count += 1
                fail_result = FormalYNormalizationResult(
                    [], "LLM_PARSE_FAIL", [{"reason": "invalid_json"}], None, formal_y_valid=False
                )
                current_user_prompt = _build_semantic_retry_user_prompt(base_user_prompt, fail_result)
                parsed_direct = None
                continue
            break

        result = normalize_formal_y(parsed, view)
        if _semantic_retryable(result, parsed) and semantic_retry_count < semantic_max_retries:
            semantic_retry_count += 1
            forced = any(
                str(a.get("reason")) in ("INVALID_FORMAL_Y_DECISION", "ARGMAX_INCONSISTENCY")
                for a in result.ambiguities
            )
            current_user_prompt = _build_semantic_retry_user_prompt(
                base_user_prompt, result, forced_decision=forced
            )
            parsed_direct = None
            continue

        formal_success = result.formal_y_valid and result.label_decision in FORMAL_Y_DECISIONS
        label_status = "LLM_LABEL_SUCCESS" if formal_success else "Y_PIPELINE_FAILURE"
        return {
            record_id_key: record_id,
            "development_pilot_only": development_pilot_only,
            "LLM_CALL_SUCCESS": True,
            "LLM_CALL_FAIL": False,
            "LLM_PARSE_SUCCESS": True,
            "LLM_PARSE_FAIL": False,
            "LLM_LABEL_SUCCESS": formal_success,
            "label_status": label_status,
            "label_decision": result.label_decision if formal_success else None,
            "parser_status": "LLM_PARSE_SUCCESS",
            "normalization_status": result.normalization_status,
            "raw_llm_output": raw,
            "parsed_llm_json": parsed,
            "normalized_structured_y": result.structured,
            "normalization_ambiguities": result.ambiguities,
            "action_schema_version": ACTION_SCHEMA_VERSION,
            "candidate_actions_audit": view.get("candidate_actions_audit"),
            "model": model_name,
            "prompt_version": Y_PROMPT_VERSION,
            "prompt_hash": prompt_version_hash(),
            "temperature": Y_LLM_TEMPERATURE,
            "retry_count": retry_count,
            "semantic_retry_count": semantic_retry_count,
            "prompt_audit": prompt_audit,
            "view_audit": view_audit,
            **_formal_y_record_fields(result),
        }

    fail_base = {
        record_id_key: record_id,
        "development_pilot_only": development_pilot_only,
        "model": model_name,
        "prompt_version": Y_PROMPT_VERSION,
        "prompt_hash": prompt_version_hash(),
        "temperature": Y_LLM_TEMPERATURE,
        "retry_count": retry_count,
        "semantic_retry_count": semantic_retry_count,
        "prompt_audit": prompt_audit,
        "view_audit": view_audit,
    }
    if raw is None and parsed_direct is None:
        return {
            **fail_base,
            "LLM_CALL_SUCCESS": False,
            "LLM_CALL_FAIL": True,
            "LLM_PARSE_SUCCESS": False,
            "LLM_PARSE_FAIL": False,
            "label_status": "Y_PIPELINE_FAILURE",
            "label_decision": None,
            "parser_status": "skipped",
            "normalization_status": "skipped",
            "raw_llm_output": None,
            "normalized_structured_y": [],
            "error": last_err,
            "formal_y_valid": False,
        }

    parsed = parsed_direct or _parse_llm_json(raw or "")
    if parsed is None:
        return {
            **fail_base,
            "LLM_CALL_SUCCESS": True,
            "LLM_CALL_FAIL": False,
            "LLM_PARSE_SUCCESS": False,
            "LLM_PARSE_FAIL": True,
            "label_status": "Y_PIPELINE_FAILURE",
            "label_decision": None,
            "parser_status": "LLM_PARSE_FAIL",
            "normalization_status": "skipped",
            "raw_llm_output": raw,
            "normalized_structured_y": [],
            "formal_y_valid": False,
        }

    result = normalize_formal_y(parsed, view)
    formal_success = result.formal_y_valid and result.label_decision in FORMAL_Y_DECISIONS
    return {
        **fail_base,
        "LLM_CALL_SUCCESS": True,
        "LLM_CALL_FAIL": False,
        "LLM_PARSE_SUCCESS": True,
        "LLM_PARSE_FAIL": False,
        "LLM_LABEL_SUCCESS": formal_success,
        "label_status": "LLM_LABEL_SUCCESS" if formal_success else "Y_PIPELINE_FAILURE",
        "label_decision": result.label_decision if formal_success else None,
        "parser_status": "LLM_PARSE_SUCCESS",
        "normalization_status": result.normalization_status,
        "raw_llm_output": raw,
        "parsed_llm_json": parsed,
        "normalized_structured_y": result.structured,
        "normalization_ambiguities": result.ambiguities,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "candidate_actions_audit": view.get("candidate_actions_audit"),
        "formal_y_valid": result.formal_y_valid,
        **_formal_y_record_fields(result),
    }

def label_one_frozen_sample(
    view: dict[str, Any],
    *,
    client: Any | None = None,
    max_attempts: int = Y_LLM_SAMPLE_MAX_ATTEMPTS,
) -> dict[str, Any]:

    return label_one_independent_y(
        view,
        client=client,
        max_attempts=max_attempts,
        development_pilot_only=True,
    )

def _scan_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports

def audit_y_information_flow(view: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(__file__)
    imports = _scan_imports(path)
    violations: list[dict[str, str]] = []
    for tok in Y_FORBIDDEN_IMPORTS:
        if any(i.startswith(tok) for i in imports):
            violations.append({"token": tok, "kind": "import"})
    if view is not None:
        for v in audit_y_view(view).get("violations") or []:
            violations.append(v)
    return {"Y_INFORMATION_FLOW_VIOLATION": len(violations), "violations": violations}

def audit_b0_y_coupling_from_y_side() -> dict[str, Any]:
    imports = _scan_imports(Path(__file__))
    b0_imports = [i for i in imports if "b0_adapter" in i or "parent_executor" in i or "parent_b0" in i]
    return {"B0_Y_COUPLING_VIOLATION": len(b0_imports), "imports": b0_imports}
