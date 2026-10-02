from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_vnext.blueprint_ir import parse_blueprint_ir
from smarthome_mdf.single_scene_blueprint_complete.yaml_mapping import resolve_yaml_path
from smarthome_mdf.synthesis_v3.blueprint_parser import parse_blueprint_yaml

_SERVICE_LINE_RE = re.compile(
    r"(?:service|action):\s*([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)",
    re.IGNORECASE,
)
_ACTION_SELECTOR_RE = re.compile(r"!input\s+['\"]?(\w+)['\"]?")

def _collect_ir_services(nodes: list[dict[str, Any]]) -> set[str]:
    services: set[str] = set()
    for node in nodes or []:
        ntype = node.get("node_type")
        if ntype == "ActionNode":
            svc = str(node.get("service") or "")
            if svc and "{{" not in svc:
                services.add(svc)
        for key in ("sequence", "default_sequence"):
            services |= _collect_ir_services(node.get(key) or [])
        for branch in node.get("branches") or []:
            services |= _collect_ir_services(branch.get("sequence") or [])
        for key in ("then_sequence", "else_sequence"):
            services |= _collect_ir_services(node.get(key) or [])
        for branch in node.get("parallel_branches") or node.get("branches") or []:
            if isinstance(branch, dict) and branch.get("sequence"):
                services |= _collect_ir_services(branch.get("sequence") or [])
    return services

def _regex_fallback_services(text: str) -> set[str]:
    services: set[str] = set()
    for m in _SERVICE_LINE_RE.finditer(text):
        services.add(f"{m.group(1).lower()}.{m.group(2).lower()}")
    return services

def _dynamic_action_input_keys(text: str) -> list[str]:
    keys: set[str] = set()
    if "selector:" in text and "action:" in text:
        for m in _ACTION_SELECTOR_RE.finditer(text):
            keys.add(m.group(1))
    return sorted(keys)

@lru_cache(maxsize=64)
def _parse_blueprint_static(blueprint_id: str) -> tuple[set[str], list[str], bool]:

    ypath = resolve_yaml_path(blueprint_id)
    if not ypath or not ypath.is_file():
        return set(), [], False

    text = ypath.read_text(encoding="utf-8", errors="ignore")
    services: set[str] = set()

    spec = parse_blueprint_yaml(blueprint_id, ypath, {})
    for ap in spec.action_paths:
        svc = str(ap.service or "")
        if svc and "{{" not in svc:
            services.add(svc)

    try:
        ir = parse_blueprint_ir(blueprint_id, ypath, {})
        services |= _collect_ir_services(ir.root_sequence)
        for tmpl in ir.action_templates or []:
            svc = str(tmpl.get("service") or "")
            if svc and "{{" not in svc:
                services.add(svc)
    except Exception:
        pass

    services |= _regex_fallback_services(text)
    dynamic_keys = _dynamic_action_input_keys(text)
    return services, dynamic_keys, True

def extract_static_yaml_services(blueprint_id: str) -> dict[str, Any]:

    services, dynamic_keys, yaml_found = _parse_blueprint_static(blueprint_id)
    return {
        "blueprint_id": blueprint_id,
        "services": sorted(services),
        "dynamic_action_inputs": dynamic_keys,
        "has_dynamic_action_selector": bool(dynamic_keys),
        "yaml_found": yaml_found,
    }

def static_yaml_service_list(blueprint_id: str) -> list[str]:
    return list(extract_static_yaml_services(blueprint_id)["services"])
