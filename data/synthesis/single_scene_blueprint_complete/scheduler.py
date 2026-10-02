from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class CoverageState:
    blueprint_counts: dict[str, int] = field(default_factory=dict)
    instance_counts: dict[str, int] = field(default_factory=dict)
    branch_counts: dict[str, int] = field(default_factory=dict)
    action_path_counts: dict[str, int] = field(default_factory=dict)
    signatures: set[str] = field(default_factory=set)

    def blueprint_total(self, bp_id: str) -> int:
        return self.blueprint_counts.get(bp_id, 0)

@dataclass
class GenerationTarget:
    scene: str
    blueprint_id: str
    automation_instance_id: str
    behavior_target: str
    branch_id: str
    action_path_signature: str
    priority: tuple

class BlueprintScheduler:
    def __init__(self, behavior_specs: list[dict], min_per_blueprint: int = 20) -> None:
        self.min_per_blueprint = min_per_blueprint
        self.coverage = CoverageState()
        self.work_queue: list[GenerationTarget] = []
        self._build_queue(behavior_specs)

    def _build_queue(self, specs: list[dict]) -> None:
        seen_bp: set[str] = set()
        for spec in specs:
            bp = spec["blueprint_id"]
            if bp in seen_bp:
                continue
            seen_bp.add(bp)
            scene = spec["scene"]
            iid = spec.get("automation_instance_id") or f"{bp}__inst_default"
            branches = spec.get("branches") or []
            if not branches:
                branches = [{"branch_id": "default", "semantic_action_templates": spec.get("semantic_action_templates") or []}]
            for br in branches:
                bid = br.get("branch_id") if isinstance(br, dict) else getattr(br, "branch_id", "default")
                templates = br.get("semantic_action_templates") if isinstance(br, dict) else []
                sig = "|".join(sorted(t.get("service", "") for t in templates)) or "valid_no_action"
                bt = f"{bp}::{bid}::{sig}"
                self.work_queue.append(
                    GenerationTarget(
                        scene=scene,
                        blueprint_id=bp,
                        automation_instance_id=iid,
                        behavior_target=bt,
                        branch_id=bid,
                        action_path_signature=sig,
                        priority=(0, bp, bid),
                    )
                )

    def next_target(self, skipped: set[str] | None = None) -> GenerationTarget | None:
        if not self.work_queue:
            return None
        skip = skipped or set()

        def score(t: GenerationTarget) -> tuple:
            bp_c = self.coverage.blueprint_total(t.blueprint_id)
            inst_c = self.coverage.instance_counts.get(t.automation_instance_id, 0)
            br_c = self.coverage.branch_counts.get(f"{t.blueprint_id}::{t.branch_id}", 0)
            ap_c = self.coverage.action_path_counts.get(t.action_path_signature, 0)
            under_bp = 0 if bp_c < self.min_per_blueprint else 1
            skipped_penalty = 1 if t.blueprint_id in skip else 0
            return (skipped_penalty, under_bp, bp_c, inst_c, br_c, ap_c, t.blueprint_id, t.branch_id)

        self.work_queue.sort(key=score)
        for t in self.work_queue:
            if t.blueprint_id not in skip:
                return t
        return None

    def register_accept(self, sample: dict) -> None:
        bb = sample.get("blueprint_binding") or {}
        meta = sample.get("synthesis_metadata") or {}
        bp = bb.get("blueprint_id", "")
        iid = bb.get("automation_instance_id", "")
        br = meta.get("branch_id", "")
        ap = meta.get("action_path_signature", "")
        sig = sample.get("diversity_signature", "")
        self.coverage.blueprint_counts[bp] = self.coverage.blueprint_counts.get(bp, 0) + 1
        self.coverage.instance_counts[iid] = self.coverage.instance_counts.get(iid, 0) + 1
        self.coverage.branch_counts[f"{bp}::{br}"] = self.coverage.branch_counts.get(f"{bp}::{br}", 0) + 1
        self.coverage.action_path_counts[ap] = self.coverage.action_path_counts.get(ap, 0) + 1
        if sig:
            self.coverage.signatures.add(sig)

    def all_blueprints_satisfied(self, blueprint_ids: list[str]) -> bool:
        return all(self.coverage.blueprint_total(bp) >= self.min_per_blueprint for bp in blueprint_ids)

    def numeric_only_near_duplicate(self, sample: dict, prior: dict) -> bool:
        meta_a = sample.get("synthesis_metadata") or {}
        meta_b = prior.get("synthesis_metadata") or {}
        bb_a = sample.get("blueprint_binding") or {}
        bb_b = prior.get("blueprint_binding") or {}
        keys = ("branch_id", "action_path_signature", "entity_binding_signature", "temporal_pattern")
        if any(meta_a.get(k) != meta_b.get(k) for k in keys):
            return False
        if bb_a.get("blueprint_id") != bb_b.get("blueprint_id"):
            return False
        if bb_a.get("automation_instance_id") != bb_b.get("automation_instance_id"):
            return False
        return True
