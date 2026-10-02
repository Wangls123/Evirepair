from __future__ import annotations

from typing import Any, Optional

from smarthome_mdf.formal_b0.runtime_evaluator import evaluate_unified_yaml_runtime
from smarthome_mdf.synthesis_v3.blueprint_parser import BlueprintSpec

class UnifiedYamlRuntimeExecutor:

    @staticmethod
    def execute(
        spec: BlueprintSpec,
        inputs: dict,
        *,
        trigger_context: dict,
        observed: dict,
        runtime_memory: Optional[dict] = None,
        entity_states: Optional[dict] = None,
        output_contract: str = "v3",
    ) -> dict[str, Any]:
        return evaluate_unified_yaml_runtime(
            spec,
            inputs,
            trigger_context=trigger_context,
            observed=observed,
            runtime_memory=runtime_memory,
            entity_states=entity_states,
            output_contract=output_contract,
        )

def execute_unified_yaml_runtime(
    spec: BlueprintSpec,
    inputs: dict,
    *,
    trigger_context: dict,
    observed: dict,
    runtime_memory: Optional[dict] = None,
    entity_states: Optional[dict] = None,
    output_contract: str = "v3",
) -> dict[str, Any]:
    return UnifiedYamlRuntimeExecutor.execute(
        spec,
        inputs,
        trigger_context=trigger_context,
        observed=observed,
        runtime_memory=runtime_memory,
        entity_states=entity_states,
        output_contract=output_contract,
    )
