from __future__ import annotations

from typing import Any

def openhab_command_to_token(item_name: str, command: str) -> str:

    cmd = str(command).upper().strip()
    item = str(item_name).strip()
    if cmd in ("ON", "OFF"):
        return f"openhab.item.{item}.{cmd.lower()}"
    return f"openhab.item.{item}.{cmd.lower()}"

def token_to_openhab_command(token: str) -> tuple[str, str] | None:

    parts = str(token).split(".")
    if len(parts) >= 4 and parts[0] == "openhab" and parts[1] == "item":
        item = ".".join(parts[2:-1]) if len(parts) > 4 else parts[2]
        cmd = parts[-1].upper()
        if cmd in ("ON", "OFF"):
            return item, cmd
        return item, cmd
    return None

def native_action_record(item: str, command: str, *, rule_uid: str, component_id: str) -> dict[str, Any]:
    return {
        "platform": "openhab",
        "item": item,
        "command": command,
        "rule_uid": rule_uid,
        "component_id": component_id,
        "repair_token": openhab_command_to_token(item, command),
    }

def tokens_from_native_actions(actions: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for a in actions:
        tok = a.get("repair_token")
        if tok:
            out.append(str(tok))
            continue
        item = a.get("item")
        cmd = a.get("command")
        if item and cmd:
            out.append(openhab_command_to_token(str(item), str(cmd)))
    return sorted(set(out))
