from __future__ import annotations

from smarthome_mdf.single_scene_blueprint_complete.config import ALL_BLUEPRINT_IDS, SCENE_BLUEPRINT_MAP

TRUE_EVIDENCE_LIMITED_BLUEPRINTS = frozenset({"appliance_vibration_sensor"})

GROUNDED_CAPABLE_BLUEPRINTS = [bp for bp in ALL_BLUEPRINT_IDS if bp not in TRUE_EVIDENCE_LIMITED_BLUEPRINTS]

TARGET_TOTAL = 12_000
ACCEPTABLE_MIN = 10_500
HARD_MAX = 13_000
BASE_MIN_PER_BLUEPRINT = 250
MAX_BLUEPRINT_SHARE_WITHIN_SCENE = 0.40

SCENE_BUDGETS: dict[str, dict[str, int]] = {
    "climate_window": {"min": 2200, "target": 2600, "recommended_max": 2900},
    "advanced_lighting": {"min": 2000, "target": 2400, "recommended_max": 2700},
    "notification_security": {"min": 1800, "target": 2200, "recommended_max": 2500},
    "visual_fusion": {"min": 1300, "target": 1600, "recommended_max": 1800},
    "appliance_monitoring": {"min": 950, "target": 1200, "recommended_max": 1400},
    "scene_schedule_override": {"min": 800, "target": 1000, "recommended_max": 1200},
    "periodic_task_scheduling": {"min": 400, "target": 550, "recommended_max": 700},
    "on_off_schedule": {"min": 350, "target": 450, "recommended_max": 600},
}

FINAL_OUTPUT_DIR_NAME = "final_regeneration"
V3_CLEAN_OUTPUT_DIR_NAME = "final_regeneration_v3_clean"

BLUEPRINT_STATE_ACTIVE = "ACTIVE"
BLUEPRINT_STATE_ALLOCATED_TARGET_REACHED = "ALLOCATED_TARGET_REACHED"
BLUEPRINT_STATE_GROUNDED_SPACE_SATURATED = "GROUNDED_SPACE_SATURATED"
BLUEPRINT_STATE_TRUE_EVIDENCE_LIMITED = "TRUE_EVIDENCE_LIMITED"

GLOBAL_STOP_TARGET_REACHED = "GLOBAL_TARGET_REACHED"
GLOBAL_STOP_ALL_GROUNDED_SATURATED = "ALL_GROUNDED_BEHAVIOR_SPACE_SATURATED"
GLOBAL_STOP_NO_MEANINGFUL_CAPACITY = "NO_MEANINGFUL_GROUNDED_CAPACITY_REMAINS"

PHASE2_INCREMENT_BASE = 150
PHASE2_MAX_BP_ABSOLUTE = 2500
PHASE2_MAX_SHARE_DELTA = 0.12

VALID_STOP_REASONS = frozenset(
    {
        "SAMPLE_TARGET_AND_COVERAGE_REACHED",
        "COVERAGE_INCOMPLETE_AT_TARGET",
        "ALLOCATED_TARGET_REACHED",
        "BLUEPRINT_GROUNDED_SPACE_SATURATED",
        "GROUNDED_SPACE_SATURATED",
        "ALL_GROUNDED_BEHAVIORS_COVERED",
        "EVIDENCE_CAPACITY_EXHAUSTED",
        "GROUNDED_BEHAVIOR_SPACE_SMALL",
        "TRUE_EVIDENCE_LIMITED",
        "PENDING",
    }
)

PATH_WEIGHT_ALPHA = 1.0
LOW_FREQUENCY_PATH_THRESHOLD = 3
MIN_STRICT_PATH_COVERAGE_RATIO = 0.95
MAX_ZERO_STRICT_PATHS = 10
MAX_PATH_SKEW = 50.0
QUOTA_REDUCTION_WHEN_PATHS_COVERED = 0.75
QUOTA_BOOST_WHEN_ZERO_PATHS = 1.25

def generated_blueprints_for_scene(scene: str) -> list[str]:
    return [bp for bp in SCENE_BLUEPRINT_MAP.get(scene, []) if bp not in TRUE_EVIDENCE_LIMITED_BLUEPRINTS]
