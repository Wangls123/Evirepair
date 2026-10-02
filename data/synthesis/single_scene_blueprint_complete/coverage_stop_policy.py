from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.path_coverage_state import PathCoverageState

GLOBAL_STOP_SAMPLE_TARGET_AND_COVERAGE_REACHED = "SAMPLE_TARGET_AND_COVERAGE_REACHED"
GLOBAL_STOP_COVERAGE_INCOMPLETE = "COVERAGE_INCOMPLETE_AT_TARGET"

@dataclass
class CoverageStopPolicy:

    min_strict_path_coverage_ratio: float = 0.95
    max_zero_strict_paths: int = 10
    max_path_skew: float = 50.0
    target_total: int = 12_000

    def evaluate(
        self,
        *,
        sample_count: int,
        coverage: PathCoverageState,
        force_at_target: bool = False,
    ) -> dict[str, Any]:
        metrics = coverage.global_metrics()
        ratio = metrics["strict_path_coverage_ratio"]
        zero_paths = metrics["zero_strict_paths"]
        max_skew = metrics.get("max_path_skew")
        skew_ok = max_skew is None or max_skew <= self.max_path_skew
        coverage_ok = ratio >= self.min_strict_path_coverage_ratio and zero_paths <= self.max_zero_strict_paths and skew_ok
        at_target = sample_count >= self.target_total

        if at_target and coverage_ok:
            return {
                "should_stop": True,
                "stop_reason": GLOBAL_STOP_SAMPLE_TARGET_AND_COVERAGE_REACHED,
                "coverage_ok": True,
                "at_target": True,
                "metrics": metrics,
            }
        if at_target and not coverage_ok and not force_at_target:
            return {
                "should_stop": False,
                "stop_reason": GLOBAL_STOP_COVERAGE_INCOMPLETE,
                "coverage_ok": False,
                "at_target": True,
                "metrics": metrics,
                "gaps": {
                    "needs_coverage_ratio": self.min_strict_path_coverage_ratio,
                    "needs_zero_paths_max": self.max_zero_strict_paths,
                    "needs_max_skew": self.max_path_skew,
                },
            }
        if force_at_target and at_target:
            return {
                "should_stop": True,
                "stop_reason": GLOBAL_STOP_COVERAGE_INCOMPLETE,
                "coverage_ok": coverage_ok,
                "at_target": True,
                "metrics": metrics,
            }
        return {
            "should_stop": False,
            "stop_reason": "PENDING",
            "coverage_ok": coverage_ok,
            "at_target": at_target,
            "metrics": metrics,
        }

    def augment_complete(
        self,
        *,
        coverage: PathCoverageState,
        blueprint_ids: list[str],
        remaining_by_bp: dict[str, int] | None = None,
    ) -> dict[str, Any]:

        remaining_by_bp = remaining_by_bp or {}
        bp_reports: list[dict[str, Any]] = []
        all_done = True
        for bp in blueprint_ids:
            missing = coverage.missing_paths(bp)
            remaining = remaining_by_bp.get(bp, 0)
            reachable_missing = missing if remaining > 0 else []
            bp_reports.append(
                {
                    "blueprint_id": bp,
                    "missing_strict_paths": [p.path_key for p in missing],
                    "reachable_missing_paths": [p.path_key for p in reachable_missing],
                    "remaining_grounded_pairs": remaining,
                    "complete": len(reachable_missing) == 0,
                }
            )
            if reachable_missing:
                all_done = False
        return {"complete": all_done, "blueprints": bp_reports}
