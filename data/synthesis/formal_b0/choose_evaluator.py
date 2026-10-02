from __future__ import annotations

from smarthome_mdf.synthesis_v3 import yaml_control_flow_evaluator as legacy

evaluate_choose_paths = legacy.evaluate_yaml_control_flow

__all__ = ["evaluate_choose_paths"]
