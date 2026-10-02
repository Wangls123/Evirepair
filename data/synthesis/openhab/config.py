from __future__ import annotations

import os
from pathlib import Path

from smarthome_mdf.paths import MDF_ROOT

OPENHAB_VALIDATION_DIR = MDF_ROOT / "runs" / "openhab_validation"
OPENHAB_SCENARIOS_DIR = OPENHAB_VALIDATION_DIR / "scenarios"
OPENHAB_SCALE_SCENARIOS_DIR = OPENHAB_VALIDATION_DIR / "scenarios_v3_500"
OPENHAB_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results"
OPENHAB_SCALE_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "scale_v3"
OPENHAB_V4_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "v4_paper_audit"
OPENHAB_V5_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "v5_final_validation"
OPENHAB_V6_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "v6_ambiguity_safety"
OPENHAB_V8_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "v8_adapter"
OPENHAB_PLATFORM_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "platform_aware"
OPENHAB_V9_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "v9"
OPENHAB_V9_LIVE_DIR = OPENHAB_V9_RESULTS_DIR / "live_v9"
OPENHAB_V9_SCENARIOS_DIR = OPENHAB_V9_RESULTS_DIR / "scenarios"
OPENHAB_FINAL_RESULTS_DIR = OPENHAB_VALIDATION_DIR / "results" / "final_paper"
OPENHAB_BLIND_SCENARIOS_DIR = OPENHAB_VALIDATION_DIR / "scenarios_v5_blind"
OPENHAB_CLEAN_SCENARIOS_DIR = OPENHAB_VALIDATION_DIR / "scenarios_v5_clean_100"
OPENHAB_AUDIT_SCENARIOS_DIR = OPENHAB_VALIDATION_DIR / "scenarios_v4_audit"
OPENHAB_DOCKER_DIR = MDF_ROOT / "docker" / "openhab_validation"

RUNTIME_SIMULATOR = "simulator"
RUNTIME_OPENHAB_DOCKER_LIVE = "openhab_docker_live"
RUNTIME_OPENHAB_REST_LIVE = "openhab_rest_live"
RUNTIME_SIMULATOR_FALLBACK = "simulator_fallback_not_live"

DEFAULT_OPENHAB_URL = os.environ.get("OPENHAB_URL", "http://localhost:8080").rstrip("/")
DEFAULT_OPENHAB_TOKEN = os.environ.get("OPENHAB_TOKEN", "").strip()

MIDDLEWARE_VERSION = "openhab_repair_middleware_v2_ambiguity_gate"

FORBIDDEN_REPAIR_INPUT_FIELDS = frozenset(
    {
        "expected_action",
        "expected_actions",
        "ground_truth",
        "y_label",
        "y_output",
        "repair_answer",
        "oracle_repair_answer",
        "decision_output",
        "multi_action_gt",
    }
)
