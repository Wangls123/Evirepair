from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import (
    ExecutableActionPath,
    audit_action_path_coverage,
    build_blueprint_action_inventory,
    enumerate_paths_from_spec,
    match_sample_to_path,
    path_balance_skew,
    strict_path_key_from_sample,
)
from smarthome_mdf.single_scene_blueprint_complete.behavior_specs import build_all_behavior_specs

DEFAULT_PATH_WEIGHT_ALPHA = 1.0
LOW_FREQUENCY_PATH_THRESHOLD = 3

def path_sampling_weight(count: int, *, alpha: float = DEFAULT_PATH_WEIGHT_ALPHA) -> float:

    return 1.0 / (max(0, count) + alpha)

def path_priority_tier(count: int, *, low_threshold: int = LOW_FREQUENCY_PATH_THRESHOLD) -> int:

    if count <= 0:
        return 0
    if count < low_threshold:
        return 1
    return 2

@dataclass
class PathCoverageState:

    alpha: float = DEFAULT_PATH_WEIGHT_ALPHA
    low_frequency_threshold: int = LOW_FREQUENCY_PATH_THRESHOLD
    paths_by_bp: dict[str, list[ExecutableActionPath]] = field(default_factory=dict)
    counts_by_bp: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    branch_counts_by_bp: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    action_counts_by_bp: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    @classmethod
    def from_behavior_specs(cls, specs: list[dict] | None = None, *, alpha: float = DEFAULT_PATH_WEIGHT_ALPHA) -> PathCoverageState:
        specs = specs or build_all_behavior_specs()
        state = cls(alpha=alpha)
        seen_bp: set[str] = set()
        for spec in specs:
            bp = spec["blueprint_id"]
            if bp in seen_bp:
                continue
            seen_bp.add(bp)
            state.paths_by_bp[bp] = enumerate_paths_from_spec(spec)
        return state

    def seed_from_samples(self, samples: list[dict]) -> None:
        for sample in samples:
            self.register_sample(sample)

    def register_sample(self, sample: dict) -> str | None:
        bb = sample.get("blueprint_binding") or {}
        bp = str(bb.get("blueprint_id") or "")
        if not bp:
            return None
        meta = sample.get("synthesis_metadata") or {}
        explicit_key = meta.get("strict_path_key")
        paths = self.paths_by_bp.get(bp) or []
        if explicit_key:
            key = str(explicit_key)
            branch_id = key.split("::", 1)[0]
            sig_part = key.split("::")[1] if "::" in key else ""
            svc = sig_part.split("|")[0] if sig_part else ""
        else:
            matched = match_sample_to_path(sample, paths)
            if matched is not None:
                key = matched.path_key
                branch_id = matched.branch_id
                svc = matched.service
            else:
                key = strict_path_key_from_sample(sample)
                if not key:
                    return None
                branch_id = key.split("::", 1)[0]
                sig_part = key.split("::")[1] if "::" in key else ""
                svc = sig_part.split("|")[0] if sig_part else ""
        self.counts_by_bp[bp][key] += 1
        self.branch_counts_by_bp[bp][branch_id] += 1
        if svc:
            self.action_counts_by_bp[bp][svc] += 1
        return key

    def path_count(self, blueprint_id: str, strict_path_key: str) -> int:
        return self.counts_by_bp.get(blueprint_id, Counter()).get(strict_path_key, 0)

    def missing_paths(self, blueprint_id: str) -> list[ExecutableActionPath]:
        paths = self.paths_by_bp.get(blueprint_id, [])
        counts = self.counts_by_bp.get(blueprint_id, Counter())
        return [p for p in paths if counts.get(p.path_key, 0) <= 0]

    def path_score_tuple(self, blueprint_id: str, strict_path_key: str) -> tuple:
        count = self.path_count(blueprint_id, strict_path_key)
        tier = path_priority_tier(count, low_threshold=self.low_frequency_threshold)
        weight = path_sampling_weight(count, alpha=self.alpha)
        return (tier, -weight, count)

    def blueprint_path_coverage_ratio(self, blueprint_id: str) -> float:
        paths = self.paths_by_bp.get(blueprint_id, [])
        if not paths:
            return 1.0
        counts = self.counts_by_bp.get(blueprint_id, Counter())
        covered = sum(1 for p in paths if counts.get(p.path_key, 0) > 0)
        return covered / len(paths)

    def blueprint_path_skew(self, blueprint_id: str) -> float:
        counts = self.counts_by_bp.get(blueprint_id, Counter())
        if not counts:
            return 0.0
        return path_balance_skew(list(counts.values()))

    def global_metrics(self) -> dict[str, Any]:
        total_paths = 0
        covered_paths = 0
        zero_paths = 0
        skew_by_bp: dict[str, float] = {}
        for bp, paths in self.paths_by_bp.items():
            counts = self.counts_by_bp.get(bp, Counter())
            bp_counts = [counts.get(p.path_key, 0) for p in paths]
            total_paths += len(paths)
            covered_paths += sum(1 for c in bp_counts if c > 0)
            zero_paths += sum(1 for c in bp_counts if c <= 0)
            if bp_counts:
                skew_by_bp[bp] = path_balance_skew(bp_counts)
        ratio = covered_paths / total_paths if total_paths else 1.0
        max_skew = max(skew_by_bp.values()) if skew_by_bp else 0.0
        return {
            "total_strict_paths": total_paths,
            "covered_strict_paths": covered_paths,
            "zero_strict_paths": zero_paths,
            "strict_path_coverage_ratio": round(ratio, 4),
            "max_path_skew": round(max_skew, 4) if max_skew != float("inf") else None,
            "blueprint_path_skew": {k: (round(v, 4) if v != float("inf") else None) for k, v in skew_by_bp.items()},
        }

    def audit_report(self, samples: list[dict] | None = None) -> dict[str, Any]:
        inventory = build_blueprint_action_inventory()
        if samples is None:
            samples = []
        report = audit_action_path_coverage(samples, inventory)
        report["path_coverage_state"] = self.global_metrics()
        return report

    def compare_metrics(self, other: PathCoverageState) -> dict[str, Any]:
        before = self.global_metrics()
        after = other.global_metrics()
        return {
            "before": before,
            "after": after,
            "delta_covered_paths": after["covered_strict_paths"] - before["covered_strict_paths"],
            "delta_zero_paths": after["zero_strict_paths"] - before["zero_strict_paths"],
            "delta_coverage_ratio": round(after["strict_path_coverage_ratio"] - before["strict_path_coverage_ratio"], 4),
            "delta_max_skew": None
            if before.get("max_path_skew") is None or after.get("max_path_skew") is None
            else round((after.get("max_path_skew") or 0) - (before.get("max_path_skew") or 0), 4),
        }
