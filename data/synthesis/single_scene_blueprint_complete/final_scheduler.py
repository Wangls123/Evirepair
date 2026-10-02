from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import _template_key
from smarthome_mdf.single_scene_blueprint_complete.final_regeneration_config import (
    BASE_MIN_PER_BLUEPRINT,
    MAX_BLUEPRINT_SHARE_WITHIN_SCENE,
    QUOTA_BOOST_WHEN_ZERO_PATHS,
    QUOTA_REDUCTION_WHEN_PATHS_COVERED,
    SCENE_BUDGETS,
    TRUE_EVIDENCE_LIMITED_BLUEPRINTS,
    generated_blueprints_for_scene,
)
from smarthome_mdf.single_scene_blueprint_complete.path_coverage_state import PathCoverageState

@dataclass
class BehaviorTarget:
    scene: str
    blueprint_id: str
    automation_instance_id: str
    behavior_target: str
    branch_id: str
    action_path_signature: str
    action_template_key: str
    strict_path_key: str = ""
    grounded_reachable: bool = True
    observed_count: int = 0
    static_only: bool = False
    source_saturated: bool = False

@dataclass
class BlueprintGenerationState:
    blueprint_id: str
    scene: str
    allocated_target: int
    attempt_budget: int
    accepted_count: int = 0
    attempt_count: int = 0
    consecutive_fail: int = 0
    stop_reason: str = "PENDING"
    initial_allocated_target: int = 0
    effective_target: int = 0
    generation_phase: int = 1
    cached_remaining_pairs: int = -1
    grounded_reachable_targets: int = 0
    observed_targets: set[str] = field(default_factory=set)
    source_obs_ids: dict[str, int] = field(default_factory=dict)
    reachable_instance_ids: set[str] = field(default_factory=set)
    observed_instance_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.initial_allocated_target:
            self.initial_allocated_target = self.allocated_target
        if not self.effective_target:
            self.effective_target = self.allocated_target

@dataclass
class SceneGenerationState:
    scene: str
    accepted_count: int = 0
    stop_reason: str = "PENDING"

class FinalGenerationScheduler:
    def __init__(
        self,
        behavior_specs: list[dict],
        blueprint_budgets: list[dict],
        *,
        instance_templates: dict[str, list[dict]] | None = None,
        diversity_selector: Any | None = None,
        path_coverage: PathCoverageState | None = None,
    ) -> None:
        self.behavior_targets: list[BehaviorTarget] = []
        self.bp_state: dict[str, BlueprintGenerationState] = {}
        self.scene_state: dict[str, SceneGenerationState] = {}
        self.signatures: set[str] = set()
        self.instance_templates = instance_templates or {}
        self.diversity_selector = diversity_selector
        self.path_coverage = path_coverage or PathCoverageState.from_behavior_specs(behavior_specs)
        self.generation_phase: int = 1
        self._build_states(behavior_specs, blueprint_budgets)
        self._build_targets(behavior_specs)

    def _build_states(self, specs: list[dict], budgets: list[dict]) -> None:
        budget_by_bp = {b["blueprint_id"]: b for b in budgets}
        seen_scene: set[str] = set()
        for spec in specs:
            bp = spec["blueprint_id"]
            scene = spec["scene"]
            if bp in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
                continue
            if bp not in self.bp_state:
                b = budget_by_bp.get(bp, {})
                st = BlueprintGenerationState(
                    blueprint_id=bp,
                    scene=scene,
                    allocated_target=int(b.get("allocated_target") or BASE_MIN_PER_BLUEPRINT),
                    attempt_budget=int(b.get("attempt_budget") or 800),
                )
                insts = self.instance_templates.get(bp) or []
                st.reachable_instance_ids = {i["automation_instance_id"] for i in insts}
                self.bp_state[bp] = st
            if scene not in seen_scene:
                seen_scene.add(scene)
                self.scene_state[scene] = SceneGenerationState(scene=scene)

    def _build_targets(self, specs: list[dict]) -> None:
        seen: set[str] = set()
        for spec in specs:
            bp = spec["blueprint_id"]
            if bp in TRUE_EVIDENCE_LIMITED_BLUEPRINTS:
                continue
            scene = spec["scene"]
            inst_list = self.instance_templates.get(bp) or [{"automation_instance_id": spec.get("automation_instance_id") or f"{bp}__inst_default"}]
            for inst in inst_list:
                iid = inst.get("automation_instance_id") or f"{bp}__inst_default"
                branches = spec.get("branches") or []
                if not branches:
                    branches = [{"branch_id": "default", "semantic_action_templates": spec.get("semantic_action_templates") or []}]
                for br in branches:
                    bid = br.get("branch_id") if isinstance(br, dict) else "default"
                    templates = br.get("semantic_action_templates") if isinstance(br, dict) else []
                    if not templates:
                        templates = [{"service": "valid.no_action", "branch_id": bid}]
                    for tmpl in templates:
                        svc = tmpl.get("service") or "valid.no_action"
                        sig = "|".join(sorted({svc}))
                        tmpl_key = _template_key(tmpl)
                        strict_key = f"{bid}::{sig}::{tmpl_key}"
                        bt_key = f"{bp}::{iid}::{bid}::{sig}::{tmpl_key}"
                        if bt_key in seen:
                            continue
                        seen.add(bt_key)
                        self.behavior_targets.append(
                            BehaviorTarget(
                                scene=scene,
                                blueprint_id=bp,
                                automation_instance_id=iid,
                                behavior_target=f"{bp}::{bid}::{sig}",
                                branch_id=bid,
                                action_path_signature=sig,
                                action_template_key=tmpl_key,
                                strict_path_key=strict_key,
                            )
                        )

    def mark_grounded_reachable(self, bp_id: str, target_key: str) -> None:
        st = self.bp_state.get(bp_id)
        if st:
            st.grounded_reachable_targets += 1
        for t in self.behavior_targets:
            if t.blueprint_id == bp_id and t.behavior_target == target_key:
                t.grounded_reachable = True
                return

    def mark_static_only(self, bp_id: str, target_key: str) -> None:
        for t in self.behavior_targets:
            if t.blueprint_id == bp_id and t.behavior_target.startswith(target_key.split("::")[0]):
                if t.behavior_target == target_key or target_key in t.behavior_target:
                    t.static_only = True
                    t.grounded_reachable = False

    def scene_count(self, scene: str) -> int:
        return self.scene_state.get(scene, SceneGenerationState(scene)).accepted_count

    def blueprint_share_in_scene(self, bp_id: str) -> float:
        st = self.bp_state.get(bp_id)
        if not st:
            return 0.0
        sc = self.scene_count(st.scene)
        if sc == 0:
            return 0.0
        gen_bps = generated_blueprints_for_scene(st.scene)
        if len(gen_bps) <= 1:
            return 0.0
        return st.accepted_count / sc

    def _scene_can_accept(self, scene: str) -> bool:
        budget = SCENE_BUDGETS.get(scene, {})
        st = self.scene_state.get(scene)
        if not st:
            return True
        if st.stop_reason != "PENDING":
            return False
        if st.accepted_count >= budget.get("recommended_max", 999999):
            return False
        return True

    def _scene_below_min(self, scene: str) -> bool:
        budget = SCENE_BUDGETS.get(scene, {})
        st = self.scene_state.get(scene)
        if not st:
            return True
        return st.accepted_count < budget.get("min", 0)

    def _instance_gap(self, st: BlueprintGenerationState) -> bool:
        return bool(st.reachable_instance_ids - st.observed_instance_ids)

    def _bp_can_accept(self, bp_id: str) -> bool:
        st = self.bp_state.get(bp_id)
        if not st or st.stop_reason != "PENDING":
            return False
        if st.attempt_count >= st.attempt_budget:
            if self._instance_gap(st) and st.accepted_count < st.allocated_target:
                st.attempt_budget = min(st.attempt_budget + 400, st.attempt_budget * 2)
                st.consecutive_fail = 0
                return True
            if st.accepted_count == 0:
                st.stop_reason = "EVIDENCE_CAPACITY_EXHAUSTED"
            elif st.accepted_count >= st.allocated_target and not self._instance_gap(st):
                st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
            elif st.accepted_count >= BASE_MIN_PER_BLUEPRINT and not self._scene_below_min(st.scene) and not self._instance_gap(st):
                st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
            else:
                st.stop_reason = "GROUNDED_BEHAVIOR_SPACE_SMALL"
            return False
        target_limit = st.effective_target if st.generation_phase >= 2 else st.initial_allocated_target
        if st.accepted_count >= target_limit:
            missing_paths = self.path_coverage.missing_paths(bp_id)
            uncovered = [
                t
                for t in self.behavior_targets
                if t.blueprint_id == bp_id and t.observed_count == 0 and t.grounded_reachable
            ]
            if missing_paths or uncovered:
                return True
            st.stop_reason = "ALLOCATED_TARGET_REACHED"
            return False
        share = self.blueprint_share_in_scene(bp_id)
        gen_bps = generated_blueprints_for_scene(st.scene)
        if (
            len(gen_bps) > 1
            and not self._scene_below_min(st.scene)
            and st.accepted_count >= BASE_MIN_PER_BLUEPRINT
            and share >= MAX_BLUEPRINT_SHARE_WITHIN_SCENE
        ):
            others_saturated = all(
                self.bp_state[ob].stop_reason != "PENDING"
                for ob in gen_bps
                if ob != bp_id and ob in self.bp_state
            )
            if not others_saturated:
                return False
        return True

    def mark_blueprint_saturated(self, bp_id: str, reason: str = "GROUNDED_SPACE_SATURATED") -> None:
        st = self.bp_state.get(bp_id)
        if st and st.stop_reason == "PENDING":
            remaining = st.cached_remaining_pairs
            missing_paths = self.path_coverage.missing_paths(bp_id)
            if remaining > 0 or missing_paths:
                if st.accepted_count >= st.effective_target:
                    st.stop_reason = "ALLOCATED_TARGET_REACHED"
                return
            if remaining >= 0 and remaining > 0:
                st.stop_reason = "ALLOCATED_TARGET_REACHED"
            else:
                st.stop_reason = reason if reason != "BLUEPRINT_GROUNDED_SPACE_SATURATED" else "GROUNDED_SPACE_SATURATED"

    def adjust_dynamic_quotas(self) -> dict[str, dict[str, Any]]:

        changes: dict[str, dict[str, Any]] = {}
        for bp_id, st in self.bp_state.items():
            if st.stop_reason != "PENDING":
                continue
            missing = self.path_coverage.missing_paths(bp_id)
            ratio = self.path_coverage.blueprint_path_coverage_ratio(bp_id)
            prev = st.effective_target or st.allocated_target
            if missing:
                boosted = int(max(prev, st.accepted_count + len(missing)) * QUOTA_BOOST_WHEN_ZERO_PATHS)
                st.effective_target = max(st.accepted_count + 1, boosted)
                changes[bp_id] = {"action": "boost_zero_paths", "missing_paths": len(missing), "effective_target": st.effective_target}
            elif ratio >= 0.99 and st.accepted_count >= BASE_MIN_PER_BLUEPRINT:
                reduced = max(BASE_MIN_PER_BLUEPRINT, int(prev * QUOTA_REDUCTION_WHEN_PATHS_COVERED))
                st.effective_target = min(prev, max(st.accepted_count, reduced))
                changes[bp_id] = {"action": "reduce_covered", "coverage_ratio": ratio, "effective_target": st.effective_target}
        return changes

    def saturated_blueprint_ids(self) -> list[str]:
        return [
            st.blueprint_id
            for st in self.bp_state.values()
            if st.stop_reason in ("GROUNDED_SPACE_SATURATED", "BLUEPRINT_GROUNDED_SPACE_SATURATED", "GROUNDED_BEHAVIOR_SPACE_SMALL")
            and (st.cached_remaining_pairs < 0 or st.cached_remaining_pairs == 0)
        ]

    def quota_limited_blueprint_ids(self) -> list[str]:
        return [
            st.blueprint_id
            for st in self.bp_state.values()
            if st.stop_reason in ("ALLOCATED_TARGET_REACHED", "SAMPLE_TARGET_AND_COVERAGE_REACHED")
        ]

    def pending_blueprint_ids(self) -> list[str]:
        return [st.blueprint_id for st in self.bp_state.values() if st.stop_reason == "PENDING"]

    def all_scheduling_inactive(self) -> bool:

        if not self.bp_state:
            return True
        return all(st.stop_reason != "PENDING" for st in self.bp_state.values())

    def all_grounded_saturated(self) -> bool:

        return self.all_scheduling_inactive()

    def all_grounded_capacity_exhausted(self, remaining_by_bp: dict[str, int]) -> bool:
        if not self.bp_state:
            return True
        for bp_id, st in self.bp_state.items():
            remaining = remaining_by_bp.get(bp_id, st.cached_remaining_pairs)
            if remaining > 0:
                return False
        return True

    def total_remaining_grounded_pairs(self, remaining_by_bp: dict[str, int]) -> int:
        return sum(max(0, v) for v in remaining_by_bp.values())

    def _behavior_source_novelty(self, target: BehaviorTarget) -> int:
        if not self.diversity_selector:
            return 0
        from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import behavior_temporal_key_from_target

        key = behavior_temporal_key_from_target(
            blueprint_id=target.blueprint_id,
            automation_instance_id=target.automation_instance_id,
            branch_id=target.branch_id,
            action_path_signature=target.action_path_signature,
            entity_binding_signature="",
        )
        saturated = self.diversity_selector.saturated_behavior_source_pairs
        prefix = f"{key}|"
        if any(p.startswith(prefix) for p in saturated):
            return 1
        return 0

    def mark_target_source_saturated(self, target: BehaviorTarget) -> None:
        target.source_saturated = True
        st = self.bp_state.get(target.blueprint_id)
        if st and st.cached_remaining_pairs > 0:
            return
        pending = [
            t
            for t in self.behavior_targets
            if t.blueprint_id == target.blueprint_id and t.grounded_reachable and not t.source_saturated
        ]
        if not pending:
            self.mark_blueprint_saturated(target.blueprint_id)

    def _path_score_for_target(self, target: BehaviorTarget) -> tuple:
        key = target.strict_path_key or f"{target.branch_id}::{target.action_path_signature}::{target.action_template_key}"
        return self.path_coverage.path_score_tuple(target.blueprint_id, key)

    def next_target(self) -> BehaviorTarget | None:
        candidates: list[tuple[tuple, BehaviorTarget]] = []
        for t in self.behavior_targets:
            if t.source_saturated:
                continue
            if not self._scene_can_accept(t.scene):
                continue
            if not self._bp_can_accept(t.blueprint_id):
                continue
            st = self.bp_state[t.blueprint_id]
            sc = self.scene_state[t.scene]
            scene_budget = SCENE_BUDGETS.get(t.scene, {})
            scene_under_min = 0 if sc.accepted_count < scene_budget.get("min", 0) else 1
            bp_under_base = 0 if st.accepted_count < BASE_MIN_PER_BLUEPRINT else 1
            instance_uncovered = 0 if t.automation_instance_id not in st.observed_instance_ids else 1
            path_tier, neg_path_weight, path_count = self._path_score_for_target(t)
            target_uncovered = 0 if path_count == 0 and t.grounded_reachable else 1
            branch_uncovered = 0 if path_count == 0 else 1
            path_uncovered = 0 if path_count == 0 else 1
            template_uncovered = 0 if t.observed_count == 0 else 1
            behavior_source_saturated = self._behavior_source_novelty(t)
            min_source_use = min(st.source_obs_ids.values()) if st.source_obs_ids else 0
            phase2_underrep = 0 if self.generation_phase >= 2 and st.generation_phase >= 2 else 1
            score = (
                scene_under_min,
                instance_uncovered,
                path_tier,
                neg_path_weight,
                branch_uncovered if self.generation_phase >= 2 else target_uncovered,
                path_uncovered if self.generation_phase >= 2 else behavior_source_saturated,
                template_uncovered if self.generation_phase >= 2 else target_uncovered,
                behavior_source_saturated if self.generation_phase >= 2 else target_uncovered,
                min_source_use if self.generation_phase >= 2 else sum(st.source_obs_ids.values()),
                phase2_underrep,
                bp_under_base,
                target_uncovered,
                path_count,
                -t.observed_count,
                st.accepted_count,
                t.blueprint_id,
                t.branch_id,
                t.strict_path_key,
            )
            candidates.append((score, t))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _targets_for_sample(self, bp: str, br: str, ap: str, sample: dict) -> list[BehaviorTarget]:
        from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import strict_path_key_from_sample

        iid = (sample.get("blueprint_binding") or {}).get("automation_instance_id")
        strict = strict_path_key_from_sample(sample)
        exact: list[BehaviorTarget] = []
        if strict:
            exact = [t for t in self.behavior_targets if t.blueprint_id == bp and t.strict_path_key == strict]
            if iid:
                inst_exact = [t for t in exact if t.automation_instance_id == iid]
                if inst_exact:
                    return inst_exact[:1]
            if len(exact) == 1:
                return exact
        loose = [t for t in self.behavior_targets if t.blueprint_id == bp and t.branch_id == br and t.action_path_signature == ap]
        if iid:
            inst_loose = [t for t in loose if t.automation_instance_id == iid]
            if len(inst_loose) == 1:
                return inst_loose
        if len(loose) == 1:
            return loose
        if exact:
            return exact[:1]
        return loose[:1] if loose else []

    def register_accept(self, sample: dict, *, behavior_target: str) -> None:
        bb = sample.get("blueprint_binding") or {}
        meta = sample.get("synthesis_metadata") or {}
        bp = bb.get("blueprint_id", "")
        scene = sample.get("scene_type", "")
        br = meta.get("branch_id", "")
        ap = meta.get("action_path_signature", "")
        sig = sample.get("diversity_signature", "")
        prov = sample.get("provenance") or {}
        src_id = str(prov.get("source_record_id") or "")

        if bp in self.bp_state:
            st = self.bp_state[bp]
            st.accepted_count += 1
            st.consecutive_fail = 0
            st.observed_targets.add(f"{br}::{ap}")
            iid = (sample.get("blueprint_binding") or {}).get("automation_instance_id")
            if iid:
                st.observed_instance_ids.add(iid)
            if src_id:
                st.source_obs_ids[src_id] = st.source_obs_ids.get(src_id, 0) + 1
        if scene in self.scene_state:
            self.scene_state[scene].accepted_count += 1
        matched = self._targets_for_sample(bp, br, ap, sample)
        for t in matched:
            t.observed_count += 1
        self.path_coverage.register_sample(sample)
        if sig:
            self.signatures.add(sig)

    def register_reject(self, bp_id: str) -> None:
        st = self.bp_state.get(bp_id)
        if not st:
            return
        st.attempt_count += 1
        st.consecutive_fail += 1
        if st.consecutive_fail >= 40 and st.accepted_count >= st.allocated_target and not self._instance_gap(st):
            st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
        elif st.consecutive_fail >= 40 and st.accepted_count >= BASE_MIN_PER_BLUEPRINT and not self._scene_below_min(st.scene) and not self._instance_gap(st):
            st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"
        elif st.consecutive_fail >= 80 and st.accepted_count == 0:
            st.stop_reason = "EVIDENCE_CAPACITY_EXHAUSTED"

    def register_attempt(self, bp_id: str) -> None:
        st = self.bp_state.get(bp_id)
        if st:
            st.attempt_count += 1

    def mark_target_saturated(self, target: BehaviorTarget) -> None:
        target.observed_count = max(target.observed_count, 1)
        for t in self.behavior_targets:
            if t.blueprint_id == target.blueprint_id and t.strict_path_key == target.strict_path_key:
                t.observed_count = max(t.observed_count, 1)

    def consecutive_numeric_dup_limit(self, bp_id: str, limit: int = 30) -> bool:
        st = self.bp_state.get(bp_id)
        return bool(st and st.consecutive_fail >= limit)

    def all_complete(self) -> bool:
        if not self.bp_state:
            return True
        for st in self.bp_state.values():
            if st.stop_reason == "PENDING":
                return False
            if self._instance_gap(st):
                return False
        return True

    def instance_coverage_gaps(self) -> dict[str, dict[str, Any]]:
        gaps: dict[str, dict[str, Any]] = {}
        for bp_id, st in self.bp_state.items():
            missing = sorted(st.reachable_instance_ids - st.observed_instance_ids)
            if missing:
                gaps[bp_id] = {
                    "reachable_instance_count": len(st.reachable_instance_ids),
                    "observed_instance_count": len(st.observed_instance_ids),
                    "missing_instance_ids": missing,
                }
        return gaps

    def total_accepted(self) -> int:
        return sum(st.accepted_count for st in self.bp_state.values())

    def finalize_scene_stops(self) -> None:
        for scene, st in self.scene_state.items():
            if st.stop_reason != "PENDING":
                continue
            budget = SCENE_BUDGETS.get(scene, {})
            if st.accepted_count >= budget.get("target", 0):
                st.stop_reason = "ALLOCATED_TARGET_REACHED"
            elif st.accepted_count >= budget.get("min", 0):
                st.stop_reason = "ALL_GROUNDED_BEHAVIORS_COVERED"

    def numeric_only_near_duplicate(self, sample: dict, prior: dict) -> bool:
        meta_a = sample.get("synthesis_metadata") or {}
        meta_b = prior.get("synthesis_metadata") or {}
        bb_a = sample.get("blueprint_binding") or {}
        bb_b = prior.get("blueprint_binding") or {}
        keys = ("branch_id", "action_path_signature", "entity_binding_signature", "temporal_pattern", "behavior_target")
        if any(meta_a.get(k) != meta_b.get(k) for k in keys):
            return False
        if bb_a.get("blueprint_id") != bb_b.get("blueprint_id"):
            return False
        if bb_a.get("automation_instance_id") != bb_b.get("automation_instance_id"):
            return False
        return True

    def temporal_shift_near_duplicate(self, sample: dict, prior: dict) -> bool:
        if not self.numeric_only_near_duplicate(sample, prior):
            return False
        prov_a = sample.get("provenance") or {}
        prov_b = prior.get("provenance") or {}
        ts_a = prov_a.get("source_record_id")
        ts_b = prov_b.get("source_record_id")
        if ts_a != ts_b:
            meta_a = sample.get("synthesis_metadata") or {}
            meta_b = prior.get("synthesis_metadata") or {}
            if meta_a.get("temporal_pattern") == meta_b.get("temporal_pattern"):
                return True
        return False

    def to_generation_state_dict(self) -> dict[str, Any]:
        return {
            "blueprints": [
                {
                    "blueprint_id": st.blueprint_id,
                    "scene": st.scene,
                    "allocated_target": st.allocated_target,
                    "initial_allocated_target": st.initial_allocated_target,
                    "effective_target": st.effective_target,
                    "generation_phase": st.generation_phase,
                    "attempt_budget": st.attempt_budget,
                    "accepted_count": st.accepted_count,
                    "attempt_count": st.attempt_count,
                    "stop_reason": st.stop_reason,
                    "cached_remaining_pairs": st.cached_remaining_pairs,
                    "observed_behavior_targets": len(st.observed_targets),
                    "unique_source_observations": len(st.source_obs_ids),
                }
                for st in self.bp_state.values()
            ],
            "scenes": [
                {
                    "scene": st.scene,
                    "accepted_count": st.accepted_count,
                    "stop_reason": st.stop_reason,
                    **SCENE_BUDGETS.get(st.scene, {}),
                }
                for st in self.scene_state.values()
            ],
        }
