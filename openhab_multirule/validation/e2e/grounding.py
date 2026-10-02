from __future__ import annotations

from typing import Any

from openhab_multirule.validation.native_repair_plan import repair_tokens_to_native_plan

def ground_repair_plan(repaired_tokens: list[str]) -> dict[str, Any]:
    native = repair_tokens_to_native_plan(repaired_tokens)
    records = [
        {"kind": "item_command", "item": item, "command": cmd.upper()}
        for item, cmd in native
    ]
    return {
        "grounding_success": bool(native),
        "native_plan": native,
        "grounded_actions": records,
        "supports": ["item_command", "state_update_via_command", "rule_modification_deferred"],
    }
