from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.sample_index import CoverageState, SampleIndex, canonical_pair, canonical_triple

@dataclass
class ScheduleRequest:
    component_count: int
    blueprint_combo: tuple[str, ...]
    use_shared_entity: bool
    same_source: bool
    same_source_record_id: str | None
    force_synthetic: bool

class CoverageScheduler:
    def __init__(
        self,
        index: SampleIndex,
        *,
        target_total: int,
        two_component_target: int,
        rng: random.Random,
        same_source_target: int | None = None,
        shared_entity_target: int | None = None,
    ) -> None:
        self.index = index
        self.target_total = target_total
        self.two_component_target = two_component_target
        self.three_component_target = target_total - two_component_target
        self.rng = rng
        caps = max(1, target_total // 20)
        self.same_source_target = same_source_target if same_source_target is not None else min(caps * 3, max(40, target_total // 20))
        self.shared_entity_target = shared_entity_target if shared_entity_target is not None else min(caps * 2, max(60, target_total // 15))
        self.soft_bp_cap = max(180, int(target_total * 0.12))
        self.soft_pair_cap = max(25, int(target_total / len(index.legal_pairs) * 3))
        self.soft_triple_cap = max(15, int(target_total / max(1, len(index.legal_triples)) * 4))

    def _needs_two_component(self, state: CoverageState) -> bool:
        two = sum(1 for _ in [])
        two = state.bp_usage and sum(
            1 for _ in range(state.same_scene_count + state.cross_scene_count)
        )

        return True

    def next_request(self, state: CoverageState, accepted_count: int) -> ScheduleRequest:
        two_count = sum(state.pair_usage.values()) + 0

        want_three = False
        if hasattr(self, "_two_accepted"):
            want_three = self._three_accepted < self.three_component_target and (
                self._two_accepted >= self.two_component_target
                or self.rng.random() < (1 - self.two_component_target / self.target_total)
            )
        else:
            want_three = self.rng.random() < (1 - self.two_component_target / self.target_total)

        want_same_source = state.same_source_count < self.same_source_target and self.index.same_source_pair_slots
        want_shared = state.shared_entity_count < self.shared_entity_target and self.index.shared_entity_pairs

        if want_three and self.index.legal_triples:
            combo = self._pick_triple(state)
            same_source = False
            rid = None
            if want_same_source and self.index.same_source_triple_slots:
                slot = self._pick_same_source_triple_slot(state)
                if slot:
                    combo = (slot[0], slot[1], slot[2])
                    rid = slot[3]
                    same_source = True
            use_shared = want_shared and canonical_pair(combo[0], combo[1]) in self.index.shared_entity_pairs and self.rng.random() < 0.5
            force_synthetic = not same_source and self.rng.random() < 0.12
            return ScheduleRequest(3, combo, use_shared, same_source, rid, force_synthetic)

        combo = self._pick_pair(state)
        same_source = False
        rid = None
        if want_same_source:
            slot = self._pick_same_source_pair_slot(state)
            if slot:
                combo = (slot[0], slot[1])
                rid = slot[2]
                same_source = True
        use_shared = want_shared and combo in self.index.shared_entity_pairs
        if use_shared and not same_source:
            use_shared = self.rng.random() < 0.7
        force_synthetic = not same_source and self.rng.random() < 0.12
        return ScheduleRequest(2, combo, use_shared, same_source, rid, force_synthetic)

    def _pick_pair(self, state: CoverageState) -> tuple[str, str]:
        scored: list[tuple[float, tuple[str, str]]] = []
        for pair in self.index.legal_pairs:
            usage = state.pair_usage.get(pair, 0)
            bp_a, bp_b = pair
            bp_pen = (state.bp_usage.get(bp_a, 0) + state.bp_usage.get(bp_b, 0)) / max(1, self.soft_bp_cap)
            if usage >= self.soft_pair_cap and bp_pen > 1.5:
                continue
            score = usage * 10 + bp_pen
            if usage == 0:
                score -= 50
            scored.append((score, pair))
        scored.sort(key=lambda x: (x[0], x[1]))
        top = scored[: max(20, len(scored) // 5)] or [(0, self.index.legal_pairs[0])]
        return self.rng.choice(top)[1]

    def _pick_triple(self, state: CoverageState) -> tuple[str, str, str]:
        scored: list[tuple[float, tuple[str, str, str]]] = []
        for triple in self.index.legal_triples:
            usage = state.triple_usage.get(triple, 0)
            bp_pen = sum(state.bp_usage.get(bp, 0) for bp in triple) / max(1, self.soft_bp_cap * 3)
            if usage >= self.soft_triple_cap and bp_pen > 2:
                continue
            score = usage * 10 + bp_pen
            if usage == 0:
                score -= 30
            scored.append((score, triple))
        scored.sort(key=lambda x: (x[0], x[1]))
        top = scored[: max(30, len(scored) // 4)] or [(0, self.index.legal_triples[0])]
        return self.rng.choice(top)[1]

    def _pick_same_source_pair_slot(self, state: CoverageState) -> tuple[str, str, str] | None:
        slots = self.index.same_source_pair_slots
        if not slots:
            return None
        scored = sorted(slots, key=lambda s: state.pair_usage.get(canonical_pair(s[0], s[1]), 0))
        return self.rng.choice(scored[: min(40, len(scored))])

    def _pick_same_source_triple_slot(self, state: CoverageState) -> tuple[str, str, str, str] | None:
        slots = self.index.same_source_triple_slots
        if not slots:
            return None
        scored = sorted(slots, key=lambda s: state.triple_usage.get(canonical_triple(s[0], s[1], s[2]), 0))
        return self.rng.choice(scored[: min(30, len(scored))])

    def materialize_samples(self, req: ScheduleRequest) -> list[dict] | None:
        idx = self.index
        if req.component_count == 2:
            a, b = req.blueprint_combo[0], req.blueprint_combo[1]
            if req.same_source and req.same_source_record_id:
                picked = idx.pick_same_source(a, b, req.same_source_record_id, self.rng)
                if picked:
                    return list(picked)
            sa = idx.pick_from_bp(a, self.rng)
            sb = idx.pick_from_bp(b, self.rng)
            if sa and sb:
                return [sa, sb]
            return None
        a, b, c = req.blueprint_combo
        if req.same_source and req.same_source_record_id:
            pa = idx.by_source_bp.get((req.same_source_record_id, a), [])
            pb = idx.by_source_bp.get((req.same_source_record_id, b), [])
            pc = idx.by_source_bp.get((req.same_source_record_id, c), [])
            if pa and pb and pc:
                return [self.rng.choice(pa), self.rng.choice(pb), self.rng.choice(pc)]
        sa = idx.pick_from_bp(a, self.rng)
        sb = idx.pick_from_bp(b, self.rng)
        sc = idx.pick_from_bp(c, self.rng)
        if sa and sb and sc:
            return [sa, sb, sc]
        return None

    def set_component_counts(self, two: int, three: int) -> None:
        self._two_accepted = two
        self._three_accepted = three

    def next_request_balanced(self, state: CoverageState, two_accepted: int, three_accepted: int) -> ScheduleRequest:
        self.set_component_counts(two_accepted, three_accepted)
        want_three = three_accepted < self.three_component_target and (
            two_accepted >= self.two_component_target or self.rng.random() < 0.4
        )
        want_same_source = state.same_source_count < self.same_source_target
        want_shared = state.shared_entity_count < self.shared_entity_target

        if want_three and three_accepted < self.three_component_target and self.index.legal_triples:
            combo = self._pick_triple(state)
            same_source = False
            rid = None
            if want_same_source and self.index.same_source_triple_slots:
                slot = self._pick_same_source_triple_slot(state)
                if slot:
                    combo = (slot[0], slot[1], slot[2])
                    rid = slot[3]
                    same_source = True
            use_shared = want_shared and self.rng.random() < 0.4
            if use_shared:
                for p in [canonical_pair(combo[0], combo[1]), canonical_pair(combo[0], combo[2]), canonical_pair(combo[1], combo[2])]:
                    if p in self.index.shared_entity_pairs:
                        use_shared = True
                        break
                else:
                    use_shared = False
            force_synthetic = not same_source and len({tuple(self.index.profiles[c].source_dataset_capabilities) for c in combo}) > 1
            return ScheduleRequest(3, combo, use_shared, same_source, rid, force_synthetic)

        combo = self._pick_pair(state)
        same_source = False
        rid = None
        if want_same_source:
            slot = self._pick_same_source_pair_slot(state)
            if slot:
                combo = (slot[0], slot[1])
                rid = slot[2]
                same_source = True
        use_shared = want_shared and combo in self.index.shared_entity_pairs and self.rng.random() < 0.65
        force_synthetic = not same_source and (
            self.index.profiles[combo[0]].source_dataset_capabilities
            != self.index.profiles[combo[1]].source_dataset_capabilities
        )
        return ScheduleRequest(2, combo, use_shared, same_source, rid, force_synthetic)
