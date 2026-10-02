from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class EntityBindingRecord:
    original_entity: str
    component_local_entity: str
    parent_entity: str
    binding_provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _domain(entity_id: str) -> str:
    return entity_id.split(".")[0] if "." in entity_id else entity_id

def bind_entities_independent(
    component_id: str,
    entities: list[str],
) -> tuple[list[EntityBindingRecord], dict[str, Any]]:

    records: list[EntityBindingRecord] = []
    prefix = f"comp_{component_id}_"
    mapping: dict[str, str] = {}
    for ent in entities:
        local = f"{prefix}{ent.replace('.', '_')}"
        parent = local
        records.append(
            EntityBindingRecord(
                original_entity=ent,
                component_local_entity=local,
                parent_entity=parent,
                binding_provenance="INDEPENDENT_NAMESPACE",
            )
        )
        mapping[ent] = parent
    return records, {
        "shared_entity": False,
        "shared_entity_mapping": {},
        "entity_binding_reason": "INDEPENDENT_ENTITIES_PRESERVED",
        "component_entity_map": mapping,
    }

def bind_entities_shared(
    component_id: str,
    entities: list[str],
    shared_targets: dict[str, str],
    *,
    reason: str,
) -> tuple[list[EntityBindingRecord], dict[str, Any]]:

    records: list[EntityBindingRecord] = []
    comp_map: dict[str, str] = {}
    for ent in entities:
        parent = shared_targets.get(ent)
        if not parent:
            prefix = f"comp_{component_id}_"
            parent = f"{prefix}{ent.replace('.', '_')}"
            prov = "INDEPENDENT_FALLBACK"
        else:
            prov = "INTENTIONAL_SHARED_ENTITY"
        local = f"comp_{component_id}_{ent.replace('.', '_')}"
        records.append(
            EntityBindingRecord(
                original_entity=ent,
                component_local_entity=local,
                parent_entity=parent,
                binding_provenance=prov,
            )
        )
        comp_map[ent] = parent
    return records, {
        "shared_entity": bool(shared_targets),
        "shared_entity_mapping": dict(shared_targets),
        "entity_binding_reason": reason,
        "component_entity_map": comp_map,
    }

def propose_shared_mapping(
    entities_a: list[str],
    entities_b: list[str],
    *,
    role_a: list[str],
    role_b: list[str],
) -> dict[str, str] | None:

    shared: dict[str, str] = {}
    for ea, ra in zip(entities_a, role_a[: len(entities_a)]):
        for eb, rb in zip(entities_b, role_b[: len(entities_b)]):
            if ra != rb:
                continue
            if ra not in {"light", "climate", "motion", "lux", "contact", "window"}:
                continue
            if _domain(ea) != _domain(eb):
                continue
            parent = f"shared.{ra}_{ea.split('.')[-1][:24]}"
            shared[ea] = parent
            shared[eb] = parent
    return shared if shared else None

def validate_entity_binding(binding_meta: dict[str, Any], all_records: list[EntityBindingRecord]) -> tuple[bool, str]:
    seen_parent: dict[str, set[str]] = {}
    for rec in all_records:
        dom = _domain(rec.parent_entity)
        seen_parent.setdefault(rec.parent_entity, set()).add(dom)
    for parent, doms in seen_parent.items():
        if len(doms) > 1 and not parent.startswith("shared."):
            return False, f"ENTITY_DOMAIN_COLLISION:{parent}"
    if binding_meta.get("shared_entity") and not binding_meta.get("shared_entity_mapping"):
        return False, "SHARED_ENTITY_MAPPING_EMPTY"
    return True, "OK"
