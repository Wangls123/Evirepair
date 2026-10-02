from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.blueprint_profile import BlueprintProfile, _entity_role
from smarthome_mdf.multi_action_frozen_v1.composability import ComposabilityClass, classify_pair, classify_triple
from smarthome_mdf.multi_action_frozen_v1.entity_policy import propose_shared_mapping

def canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def canonical_triple(a: str, b: str, c: str) -> tuple[str, str, str]:
    return tuple(sorted((a, b, c)))

@dataclass
class SampleIndex:
    by_id: dict[str, dict]
    by_bp: dict[str, list[dict]]
    by_source: dict[str, list[dict]]
    by_source_bp: dict[tuple[str, str], list[dict]]
    legal_pairs: list[tuple[str, str]]
    legal_triples: list[tuple[str, str, str]]
    shared_entity_pairs: set[tuple[str, str]]
    same_source_pair_slots: list[tuple[str, str, str]]
    same_source_triple_slots: list[tuple[str, str, str, str]]
    profiles: dict[str, BlueprintProfile]

    def pick_from_bp(self, bp: str, rng, *, exclude: set[str] | None = None) -> dict | None:
        pool = [s for s in self.by_bp.get(bp, []) if not exclude or s["sample_id"] not in exclude]
        if not pool:
            return None
        return pool[rng.randrange(len(pool))]

    def pick_same_source(self, bp_a: str, bp_b: str, source_record_id: str, rng) -> tuple[dict, dict] | None:
        pool_a = self.by_source_bp.get((source_record_id, bp_a), [])
        pool_b = self.by_source_bp.get((source_record_id, bp_b), [])
        if not pool_a or not pool_b:
            return None
        return pool_a[rng.randrange(len(pool_a))], pool_b[rng.randrange(len(pool_b))]

def build_sample_index(samples: list[dict], profiles: dict[str, BlueprintProfile]) -> SampleIndex:
    by_id = {s["sample_id"]: s for s in samples}
    by_bp: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    by_source_bp: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for s in samples:
        bp = s["blueprint_binding"]["blueprint_id"]
        by_bp[bp].append(s)
        rid = (s.get("provenance") or {}).get("source_record_id")
        if rid:
            by_source[rid].append(s)
            by_source_bp[(rid, bp)].append(s)

    grounded = sorted(k for k, v in profiles.items() if v.sample_count > 0)
    legal_pairs: list[tuple[str, str]] = []
    shared_entity_pairs: set[tuple[str, str]] = set()
    for i in range(len(grounded)):
        for j in range(i + 1, len(grounded)):
            a, b = grounded[i], grounded[j]
            pc = classify_pair(profiles[a], profiles[b])
            if pc.classification != ComposabilityClass.INCOMPATIBLE.value:
                legal_pairs.append(canonical_pair(a, b))
                if pc.shared_entity_opportunity:
                    shared_entity_pairs.add(canonical_pair(a, b))

    legal_triples: list[tuple[str, str, str]] = []
    for combo in combinations(grounded, 3):
        t = classify_triple(profiles[combo[0]], profiles[combo[1]], profiles[combo[2]])
        if t["classification"] != ComposabilityClass.INCOMPATIBLE.value:
            legal_triples.append(canonical_triple(*combo))

    same_source_pair_slots: list[tuple[str, str, str]] = []
    same_source_triple_slots: list[tuple[str, str, str, str]] = []
    for rid, rows in by_source.items():
        bps = sorted({s["blueprint_binding"]["blueprint_id"] for s in rows})
        if len(bps) >= 2:
            for a, b in combinations(bps, 2):
                if canonical_pair(a, b) in set(legal_pairs):
                    same_source_pair_slots.append((a, b, rid))
        if len(bps) >= 3:
            for combo in combinations(bps, 3):
                if canonical_triple(*combo) in set(legal_triples):
                    same_source_triple_slots.append((*combo, rid))

    return SampleIndex(
        by_id=by_id,
        by_bp=dict(by_bp),
        by_source=dict(by_source),
        by_source_bp=dict(by_source_bp),
        legal_pairs=legal_pairs,
        legal_triples=legal_triples,
        shared_entity_pairs=shared_entity_pairs,
        same_source_pair_slots=same_source_pair_slots,
        same_source_triple_slots=same_source_triple_slots,
        profiles=profiles,
    )

def estimate_capacities(index: SampleIndex) -> dict[str, Any]:
    shared_grounded = 0
    for bp_a, bp_b in index.shared_entity_pairs:
        for sa in index.by_bp.get(bp_a, [])[:50]:
            ents_a = sa["blueprint_binding"].get("entities") or []
            roles_a = [_entity_role(e) for e in ents_a]
            for sb in index.by_bp.get(bp_b, [])[:50]:
                ents_b = sb["blueprint_binding"].get("entities") or []
                roles_b = [_entity_role(e) for e in ents_b]
                if propose_shared_mapping(ents_a, ents_b, role_a=roles_a, role_b=roles_b):
                    shared_grounded += 1
                    break
            else:
                continue
            break

    return {
        "eligible_pair_count": len(index.legal_pairs),
        "eligible_triple_count": len(index.legal_triples),
        "same_source_capacity_estimate": len(index.same_source_pair_slots),
        "same_source_triple_capacity_estimate": len(index.same_source_triple_slots),
        "shared_entity_pair_opportunities": len(index.shared_entity_pairs),
        "shared_entity_capacity_estimate": max(shared_grounded, len(index.shared_entity_pairs)),
    }

@dataclass
class CoverageState:
    bp_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pair_usage: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    triple_usage: dict[tuple[str, str, str], int] = field(default_factory=lambda: defaultdict(int))
    scene_pair_usage: dict[tuple[str, ...], int] = field(default_factory=lambda: defaultdict(int))
    source_mode_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    temporal_mode_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    same_scene_count: int = 0
    cross_scene_count: int = 0
    shared_entity_count: int = 0
    independent_entity_count: int = 0
    same_source_count: int = 0
    component_tuple_seen: set[frozenset[str]] = field(default_factory=set)
    parent_signatures_seen: set[str] = field(default_factory=set)

    def record_accept(self, record: dict[str, Any]) -> None:
        comps = record.get("components") or []
        bps = [c.get("blueprint_id") for c in comps]
        for bp in bps:
            self.bp_usage[bp] += 1
        if len(bps) == 2:
            self.pair_usage[canonical_pair(bps[0], bps[1])] += 1
        elif len(bps) == 3:
            self.triple_usage[canonical_triple(bps[0], bps[1], bps[2])] += 1
        scenes = tuple(sorted({c.get("scene") for c in comps}))
        self.scene_pair_usage[scenes] += 1
        if record.get("composition_type") == "SAME_SCENE":
            self.same_scene_count += 1
        else:
            self.cross_scene_count += 1
        sm = record.get("source_mode") or "UNKNOWN"
        self.source_mode_usage[sm] += 1
        tm = (record.get("composition_timeline") or {}).get("alignment_mode") or "UNKNOWN"
        self.temporal_mode_usage[tm] += 1
        if (record.get("shared_entities") or {}).get("shared_entity"):
            self.shared_entity_count += 1
        else:
            self.independent_entity_count += 1
        if sm == "SAME_SOURCE":
            self.same_source_count += 1
        sids = frozenset(c.get("single_scene_sample_id") for c in comps)
        self.component_tuple_seen.add(sids)
        sig = record.get("canonical_parent_signature")
        if sig:
            self.parent_signatures_seen.add(sig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bp_usage": dict(self.bp_usage),
            "pair_usage": {f"{a}::{b}": v for (a, b), v in self.pair_usage.items()},
            "triple_usage": {f"{a}::{b}::{c}": v for (a, b, c), v in self.triple_usage.items()},
            "source_mode_usage": dict(self.source_mode_usage),
            "temporal_mode_usage": dict(self.temporal_mode_usage),
            "same_scene_count": self.same_scene_count,
            "cross_scene_count": self.cross_scene_count,
            "shared_entity_count": self.shared_entity_count,
            "independent_entity_count": self.independent_entity_count,
            "same_source_count": self.same_source_count,
            "parent_signatures_seen": sorted(self.parent_signatures_seen),
            "component_tuple_seen": [sorted(x) for x in self.component_tuple_seen],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CoverageState:
        st = cls()
        st.bp_usage = defaultdict(int, data.get("bp_usage") or {})
        for k, v in (data.get("pair_usage") or {}).items():
            a, b = k.split("::")
            st.pair_usage[(a, b)] = v
        for k, v in (data.get("triple_usage") or {}).items():
            parts = k.split("::")
            st.triple_usage[(parts[0], parts[1], parts[2])] = v
        st.source_mode_usage = defaultdict(int, data.get("source_mode_usage") or {})
        st.temporal_mode_usage = defaultdict(int, data.get("temporal_mode_usage") or {})
        st.same_scene_count = data.get("same_scene_count", 0)
        st.cross_scene_count = data.get("cross_scene_count", 0)
        st.shared_entity_count = data.get("shared_entity_count", 0)
        st.independent_entity_count = data.get("independent_entity_count", 0)
        st.same_source_count = data.get("same_source_count", 0)
        st.parent_signatures_seen = set(data.get("parent_signatures_seen") or [])
        st.component_tuple_seen = {frozenset(x) for x in (data.get("component_tuple_seen") or [])}
        return st
