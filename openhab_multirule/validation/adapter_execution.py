from __future__ import annotations

from openhab_multirule.validation.action_adapter import token_to_openhab_command

def ordered_unique_tokens(tokens: list[str]) -> list[str]:

    item_to_tok: dict[str, str] = {}
    item_order: list[str] = []
    for tok in tokens:
        parsed = token_to_openhab_command(tok)
        key = parsed[0] if parsed else str(tok)
        if key not in item_to_tok:
            item_order.append(key)
        item_to_tok[key] = tok
    return [item_to_tok[k] for k in item_order]

def tokens_to_final_item_commands(tokens: list[str]) -> list[tuple[str, str]]:

    final: dict[str, str] = {}
    order: list[str] = []
    for tok in ordered_unique_tokens(tokens):
        parsed = token_to_openhab_command(tok)
        if not parsed:
            continue
        item, cmd = parsed
        if item not in final:
            order.append(item)
        final[item] = cmd.upper()
    return [(item, final[item]) for item in order]

def merge_b0_repair_commands(
    b0_tokens: list[str],
    repaired_tokens: list[str],
) -> list[tuple[str, str]]:

    repaired_cmds = tokens_to_final_item_commands(repaired_tokens)
    repaired_items = {item for item, _ in repaired_cmds}
    b0_items_on = set()
    for tok in b0_tokens:
        parsed = token_to_openhab_command(tok)
        if parsed and parsed[1].upper() == "ON":
            b0_items_on.add(parsed[0])

    extra_off: list[tuple[str, str]] = []
    for item in sorted(b0_items_on):
        if item not in repaired_items:
            extra_off.append((item, "OFF"))

    seen = {item for item, _ in repaired_cmds}
    plan = list(repaired_cmds)
    for item, cmd in extra_off:
        if item not in seen:
            plan.append((item, cmd))
    return plan
