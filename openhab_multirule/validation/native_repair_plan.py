from __future__ import annotations

from typing import Any

from openhab_multirule.validation.action_adapter import token_to_openhab_command
from openhab_multirule.validation.adapter_execution import merge_b0_repair_commands, tokens_to_final_item_commands

def repair_tokens_to_native_plan(repaired_tokens: list[str]) -> list[tuple[str, str]]:

    return tokens_to_final_item_commands(repaired_tokens)

def build_native_plan_from_repair(
    b0_tokens: list[str],
    repaired_tokens: list[str],
) -> list[tuple[str, str]]:

    if not repaired_tokens:
        return []
    return merge_b0_repair_commands(b0_tokens, repaired_tokens)

def native_plan_to_records(plan: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"type": "sendCommand", "item": item, "command": cmd} for item, cmd in plan]
