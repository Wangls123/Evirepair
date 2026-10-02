from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import ExecutableActionPath, _template_key
from smarthome_mdf.single_scene_blueprint_complete.generation_path_request import GenerationPathRequest

_MAESTRO_VAR_INPUT: dict[str, str] = {
    "main_lights_var": "main_lights",
    "day_lights_var": "day_lights",
    "night_lights_var": "night_lights",
    "scene_day_var": "scene_day",
    "scene_night_var": "scene_night",
}

@dataclass(frozen=True)
class StrictPathIdentity:
    branch_id: str
    action_path_signature: str
    template_key: str
    strict_path_key: str
    template_provenance: str
    resolved_target_entity: str | None
    parameter_keys: tuple[str, ...] = ()

    def canonical_template_key(self) -> str:
        svc = self.action_path_signature.split("|")[0] if "|" in self.action_path_signature else self.action_path_signature
        resolved = self.resolved_target_entity
        if resolved is None and self.template_provenance:
            resolved = self.template_provenance
        params = list(self.parameter_keys)
        return f"{svc}|{resolved}|{params}"

def _normalize_entity_ref(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        flat = [str(v).strip() for v in value if str(v).strip()]
        if not flat:
            return None
        return "+".join(sorted(flat)) if len(flat) > 1 else flat[0]
    if isinstance(value, dict):
        eid = value.get("entity_id")
        if isinstance(eid, str):
            return eid
        if isinstance(eid, list):
            return _normalize_entity_ref(eid)
    return str(value)

def _input_value(inputs: dict[str, Any], key: str) -> Any:
    if key in inputs:
        return inputs[key]
    for part in key.split("."):
        if part in inputs:
            return inputs[part]
    return None

def resolve_input_selector(raw: str, *, instance: dict[str, Any]) -> str | None:

    if not raw.startswith("!input "):
        return None
    key = raw.replace("!input ", "").strip()
    inputs = dict(instance.get("blueprint_inputs") or {})
    val = _input_value(inputs, key)
    resolved = _normalize_entity_ref(val)
    if resolved:
        return resolved
    bound = instance.get("bound_entities") or []
    for ent in bound:
        ent_s = str(ent)
        if key.replace("_helper", "") in ent_s or key in ent_s:
            return ent_s
    return None

def resolve_jinja_variable(expr: str, *, instance: dict[str, Any]) -> str | None:

    text = expr.strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[2:-2].strip()
    parts = re.split(r"\s*\+\s*", text)
    resolved_parts: list[str] = []
    inputs = dict(instance.get("blueprint_inputs") or {})
    for part in parts:
        part = part.strip()
        input_key = _MAESTRO_VAR_INPUT.get(part)
        if input_key is None and part.endswith("_var"):
            input_key = part[:-4]
        if input_key and input_key in inputs:
            norm = _normalize_entity_ref(inputs[input_key])
            if norm:
                resolved_parts.append(norm)
                continue
        if part in inputs:
            norm = _normalize_entity_ref(inputs[part])
            if norm:
                resolved_parts.append(norm)
                continue
        legacy_entity = inputs.get("entity")
        if legacy_entity and part.endswith("_var"):
            resolved_parts.append(str(legacy_entity))
            continue
    if not resolved_parts:
        return None
    if len(resolved_parts) == 1:
        return resolved_parts[0]
    return "+".join(resolved_parts)

def resolve_target_entity(raw: str | None, *, instance: dict[str, Any]) -> tuple[str | None, str]:

    if not raw:
        return None, ""
    raw = str(raw)
    if raw.startswith("!input "):
        return resolve_input_selector(raw, instance=instance), raw
    if raw.startswith("{{") or "_var" in raw:
        expr = raw if raw.startswith("{{") else f"{{{{ {raw} }}}}"
        return resolve_jinja_variable(expr, instance=instance), raw
    return raw, raw

def build_strict_path_identity(
    *,
    branch_id: str,
    action_path_signature: str,
    template_key: str,
    target_entity_raw: str | None,
    instance: dict[str, Any],
    parameter_keys: tuple[str, ...] = (),
) -> StrictPathIdentity:
    provenance = str(target_entity_raw or "")
    if not provenance and "|" in template_key:
        parts = template_key.split("|", 2)
        if len(parts) >= 2:
            provenance = parts[1] if parts[1] not in ("None", "") else ""
    resolved, prov = resolve_target_entity(provenance or None, instance=instance)
    if prov:
        provenance = prov
    strict = f"{branch_id}::{action_path_signature}::{template_key}"
    return StrictPathIdentity(
        branch_id=branch_id,
        action_path_signature=action_path_signature,
        template_key=template_key,
        strict_path_key=strict,
        template_provenance=provenance,
        resolved_target_entity=resolved,
        parameter_keys=parameter_keys,
    )

def identity_from_path(path: ExecutableActionPath, instance: dict[str, Any]) -> StrictPathIdentity:
    return build_strict_path_identity(
        branch_id=path.branch_id,
        action_path_signature=path.action_path_signature,
        template_key=path.action_template_key,
        target_entity_raw=path.target_entity,
        instance=instance,
        parameter_keys=tuple(path.parameter_keys or ()),
    )

def identity_from_request(request: GenerationPathRequest, instance: dict[str, Any]) -> StrictPathIdentity:
    return build_strict_path_identity(
        branch_id=request.target_branch,
        action_path_signature=request.target_signature,
        template_key=request.target_template_key,
        target_entity_raw=request.target_entity_raw or request.template_provenance,
        instance=instance,
        parameter_keys=request.parameter_keys,
    )

def identity_from_sample(sample: dict, *, instance: dict[str, Any] | None = None) -> StrictPathIdentity | None:
    meta = sample.get("synthesis_metadata") or {}
    branch = str(meta.get("branch_id") or "")
    sig = str(meta.get("action_path_signature") or "")
    if not branch or not sig:
        return None
    tmpl_key = str(meta.get("target_template_key") or "")
    provenance = str(meta.get("template_provenance") or "")
    resolved = meta.get("resolved_target_entity")
    if not tmpl_key:
        svc = sig.split("|")[0] if "|" in sig else sig
        target = meta.get("target_entity") or resolved or provenance
        params = meta.get("template_parameter_keys") or []
        if isinstance(params, str):
            params = json.loads(params) if params.startswith("[") else [params]
        tmpl_key = f"{svc}|{target}|{sorted(params)}"
    if not provenance and "|" in tmpl_key:
        provenance = tmpl_key.split("|", 1)[1] if "|" in tmpl_key else ""
    inst = instance or {"blueprint_inputs": (sample.get("blueprint_binding") or {}).get("grounded_instance", {}).get("blueprint_inputs", {})}
    if resolved is None and instance is not None:
        _, provenance2 = resolve_target_entity(provenance or None, instance=inst)
        resolved = resolve_target_entity(provenance or None, instance=inst)[0]
        if provenance2:
            provenance = provenance2
    strict = str(meta.get("strict_path_key") or f"{branch}::{sig}::{tmpl_key}")
    return StrictPathIdentity(
        branch_id=branch,
        action_path_signature=sig,
        template_key=tmpl_key,
        strict_path_key=strict,
        template_provenance=provenance,
        resolved_target_entity=str(resolved) if resolved else None,
        parameter_keys=tuple(meta.get("template_parameter_keys") or ()),
    )

def canonical_template_keys_equivalent(a: StrictPathIdentity, b: StrictPathIdentity) -> bool:
    if a.branch_id != b.branch_id or a.action_path_signature != b.action_path_signature:
        return False
    if a.template_key == b.template_key:
        return True
    if a.template_provenance and a.template_provenance == b.template_provenance:
        return True
    ca = a.canonical_template_key()
    cb = b.canonical_template_key()
    if ca == cb:
        return True
    if a.resolved_target_entity and b.resolved_target_entity and a.resolved_target_entity == b.resolved_target_entity:
        return a.parameter_keys == b.parameter_keys
    return False

def strict_paths_equivalent(
    target: StrictPathIdentity | GenerationPathRequest | ExecutableActionPath,
    sample: dict,
    *,
    instance: dict[str, Any],
) -> bool:
    if isinstance(target, GenerationPathRequest):
        target_id = identity_from_request(target, instance)
    elif isinstance(target, ExecutableActionPath):
        target_id = identity_from_path(target, instance)
    else:
        target_id = target
    generated = identity_from_sample(sample, instance=instance)
    if generated is None:
        return False
    return canonical_template_keys_equivalent(target_id, generated)

def apply_template_path_metadata(
    sample: dict,
    request: GenerationPathRequest,
    instance: dict[str, Any],
) -> dict:

    identity = identity_from_request(request, instance)
    meta = dict(sample.get("synthesis_metadata") or {})
    meta["branch_id"] = request.target_branch
    meta["action_path_signature"] = request.target_signature
    meta["target_template_key"] = request.target_template_key
    meta["template_provenance"] = identity.template_provenance or request.template_provenance
    meta["resolved_target_entity"] = identity.resolved_target_entity
    meta["strict_path_key"] = request.strict_path_key
    meta["template_parameter_keys"] = list(request.parameter_keys)
    meta["generation_path_request"] = request.to_dict()
    meta["action_path_scheduling"] = "template_aware_v3"
    sample["synthesis_metadata"] = meta
    prov = dict(sample.get("provenance") or {})
    prov["template_provenance"] = meta["template_provenance"]
    prov["resolved_target_entity"] = identity.resolved_target_entity
    sample["provenance"] = prov
    return sample

def validate_generated_matches_target(
    sample: dict,
    request: GenerationPathRequest,
    instance: dict[str, Any],
) -> tuple[bool, str]:
    meta = sample.get("synthesis_metadata") or {}
    if meta.get("branch_id") != request.target_branch:
        return False, f"branch mismatch {meta.get('branch_id')} != {request.target_branch}"
    if meta.get("action_path_signature") != request.target_signature:
        return False, f"signature mismatch {meta.get('action_path_signature')} != {request.target_signature}"
    target_id = identity_from_request(request, instance)
    if target_id.template_provenance.startswith("!input") and not target_id.resolved_target_entity:
        return False, f"unresolved !input binding for {target_id.template_provenance}"
    if target_id.template_provenance.startswith("{{") and not target_id.resolved_target_entity:
        return False, f"unresolved template variable {target_id.template_provenance}"
    apply_template_path_metadata(sample, request, instance)
    generated = identity_from_sample(sample, instance=instance)
    if generated is None:
        return False, "missing generated identity"
    if generated.strict_path_key != request.strict_path_key:
        return False, f"strict_path_key mismatch {generated.strict_path_key} != {request.strict_path_key}"
    if not canonical_template_keys_equivalent(target_id, generated):
        return False, (
            f"template identity mismatch target={target_id.canonical_template_key()} "
            f"generated={generated.canonical_template_key()}"
        )
    return True, "ok"
