from __future__ import annotations

import hashlib
import json
from typing import Any

def parent_signature(
    *,
    blueprint_ids: list[str],
    behavior_signatures: list[str],
    source_identities: list[str],
    shared_entity: bool,
    shared_entity_mapping: dict[str, str],
    temporal_alignment_class: str,
    component_count: int,
) -> str:

    payload = {
        "parent_signature_version": "parent_signature_v1",
        "component_count": component_count,
        "blueprint_multiset": sorted(blueprint_ids),
        "component_identities": sorted(
            [
                {
                    "blueprint": bp,
                    "behavior": beh,
                    "source": src,
                }
                for bp, beh, src in zip(blueprint_ids, behavior_signatures, source_identities)
            ],
            key=lambda x: (x["blueprint"], x["behavior"], x["source"]),
        ),
        "entity_sharing": {
            "shared_entity": shared_entity,
            "shared_entity_mapping": dict(sorted(shared_entity_mapping.items())),
        },
        "temporal_alignment_class": temporal_alignment_class,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
