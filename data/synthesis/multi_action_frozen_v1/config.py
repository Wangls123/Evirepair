from __future__ import annotations

from pathlib import Path

from smarthome_mdf.paths import DATA_DIR

FROZEN_V1_DIR = DATA_DIR / "single_scene_blueprint_complete" / "frozen_v1"
FROZEN_SAMPLES = FROZEN_V1_DIR / "single_scene_samples.jsonl"
FROZEN_MANIFEST = FROZEN_V1_DIR / "FREEZE_SHA256_MANIFEST.json"
FROZEN_INVENTORY = FROZEN_V1_DIR / "blueprint_inventory.json"

OUTPUT_DIR = DATA_DIR / "multi_action_frozen_v1"

SCHEMA_VERSION = "multi_action_frozen_v1_1"
COMPOSABILITY_POLICY_VERSION = "composability_v1"
ENTITY_BINDING_VERSION = "entity_binding_v1"
TEMPORAL_ALIGNMENT_VERSION = "temporal_alignment_v1"
PARENT_SIGNATURE_VERSION = "parent_signature_v1"
CONSTRUCTION_VALIDATOR_VERSION = "construction_validator_v1"

DEFAULT_PILOT_SEED = 20260903
PILOT_TARGET = 250
PILOT_TWO_COMPONENT_RATIO = 0.6

FULL_GENERATION_SEED = 202609032400
FORMAL_TARGET_TOTAL = 2400
FORMAL_TWO_COMPONENT_TARGET = 1440
FORMAL_THREE_COMPONENT_TARGET = 960
FORMAL_TWO_COMPONENT_RATIO = 0.6
SMOKE_TARGET = 350
SCHEDULER_VERSION = "coverage_scheduler_v1"

EXPECTED_FROZEN_SHA256 = "ff00370ae4b5d8d9933e3282353369e38faa702ba49425d46c30f98fad47a897"

MULTI_ACTION_FROZEN_V1_DIR = OUTPUT_DIR / "frozen_v1"
MULTI_ACTION_FROZEN_SAMPLES = MULTI_ACTION_FROZEN_V1_DIR / "multi_action_samples.jsonl"
MULTI_ACTION_FROZEN_SS_REPAIR_DIR = OUTPUT_DIR / "frozen_v1_ss_repair"
MULTI_ACTION_FROZEN_SS_REPAIR_SAMPLES = MULTI_ACTION_FROZEN_SS_REPAIR_DIR / "multi_action_samples.jsonl"

SS_REPAIR_DIR = DATA_DIR / "single_scene_blueprint_complete" / "frozen_v1_ss_repair"
SS_REPAIR_SAMPLES = SS_REPAIR_DIR / "single_scene_samples.jsonl"

FORMAL_B0_SS_REPAIR_DIR = OUTPUT_DIR / "formal_b0_v2_ss_repair"
FORMAL_B0_SS_REPAIR_FROZEN_DIR = FORMAL_B0_SS_REPAIR_DIR / "frozen_v1"
Y_LABELING_SS_REPAIR_DIR = OUTPUT_DIR / "y_labeling_ss_repair"
EXPECTED_MULTI_ACTION_FROZEN_SHA256 = (
    "891a4c39b4a7cf67fb964947199bcfeb1e5fdea22859e14a8c47a1d15c87d883"
)

DOWNSTREAM_PILOT_DIR = OUTPUT_DIR / "downstream_pilot"
DOWNSTREAM_PILOT_B0_DIR = DOWNSTREAM_PILOT_DIR / "b0"
DOWNSTREAM_PILOT_Y_DIR = DOWNSTREAM_PILOT_DIR / "y"
DOWNSTREAM_ADAPTER_PILOT_DIR = OUTPUT_DIR / "downstream_adapter_pilot"

B0_ADAPTER_VERSION = "b0_adapter_v1"
FORMAL_B0_PIPELINE_VERSION = "formal_b0_v1"
FORMAL_B0_V2_PIPELINE_VERSION = "formal_b0_v2"
FORMAL_B0_V4_PIPELINE_VERSION = "formal_b0_v4"
FORMAL_B0_V4_CANDIDATE_VERSION = "formal_b0_yaml_runtime_v4_candidate"
FORMAL_B0_REGISTRY_VERSION = "frozen_blueprint_executor_registry_v2"
FORMAL_B0_GENERIC_EXECUTOR_VERSION = "generic_blueprint_executor_v1"
FORMAL_B0_RUNTIME_PROJECTOR_VERSION = "frozen_runtime_projector_v1"
FORMAL_B0_INSTANCE_RESOLVER_VERSION = "frozen_instance_resolver_v1"
FORMAL_B0_DIR = OUTPUT_DIR / "formal_b0_v1"
FORMAL_B0_FROZEN_DIR = FORMAL_B0_DIR / "frozen_v1"
FORMAL_B0_V2_DIR = OUTPUT_DIR / "formal_b0_v2"
FORMAL_B0_V2_FROZEN_DIR = FORMAL_B0_V2_DIR / "frozen_v1"
FORMAL_B0_V1_FROZEN_SHA256 = (
    "a0382efdd35023aec90a62a55a0b4711fb7f295bef65eebcf45bcee5cc32a472"
)
Y_ADAPTER_VERSION = "y_labeling_view_v1"
Y_PROMPT_VERSION = "frozen_independent_y_v5"
Y_OUTPUT_SCHEMA_VERSION = "formal_y_minimal_v1"
Y_RUNTIME_SEMANTIC_BUILDER_VERSION = "y_runtime_semantic_v1"
ACTION_SCHEMA_VERSION = "action_vnext_v1"
DOWNSTREAM_PILOT_SEED = 20260903
DOWNSTREAM_PILOT_TARGET = 50
DOWNSTREAM_B0_READINESS_TARGET = 80

FULL_GENERATION_DIR = OUTPUT_DIR / "full_generation"
SMOKE_DIR = OUTPUT_DIR / "smoke"
GENERATION_1200_DIR = OUTPUT_DIR / "generation_1200_3scene"

GENERATION_1200_TARGET = 1200
GENERATION_1200_TWO_COMPONENT_TARGET = 720
GENERATION_1200_THREE_COMPONENT_TARGET = 480
GENERATION_1200_SEED = 202609051200
GENERATION_1200_UPSTREAM = DATA_DIR / "single_scene_blueprint_complete" / "single_scene_frozen_v3" / "single_scene_samples.jsonl"
GENERATION_1200_SCENES = frozenset(
    {
        "climate_window",
        "advanced_lighting",
        "notification_security",
    }
)

FORBIDDEN_FIELDS = frozenset(
    {
        "b0",
        "b0_actions",
        "parent_b0",
        "expected_actions",
        "expected_state",
        "y",
        "y_label",
        "ground_truth",
        "conflict_label",
        "conflict_type",
        "repair_actions",
        "repair_output",
        "diagnosis",
        "ars",
        "decision_output",
        "component_oracle_traces",
        "local_expected_actions",
        "operator",
    }
)

FORBIDDEN_IMPORT_PREFIXES = (
    "smarthome_mdf.evaluation",
    "smarthome_mdf.compositional_repair",
    "smarthome_mdf.multi_action_vnext.gt_",
    "smarthome_mdf.multi_action_vnext.conflict_",
    "smarthome_mdf.multi_action_vnext.parent_executor",
)
