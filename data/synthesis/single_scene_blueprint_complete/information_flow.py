from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

FORBIDDEN_IMPORT_PREFIXES = (
    "smarthome_mdf.evaluation",
    "smarthome_mdf.repair",
    "smarthome_mdf.conflict",
    "smarthome_mdf.baselines",
)

FORBIDDEN_SYMBOLS = (
    "bind_sample_blueprint_id",
    "run_phase9",
    "conflict_builder",
    "repair_",
)

def audit_package_imports() -> list[str]:
    violations: list[str] = []
    for py in PACKAGE_DIR.glob("*.py"):
        if py.name.startswith("_"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    violations.extend(_check_import(py.name, alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                violations.extend(_check_import(py.name, mod))
    return violations

def _check_import(file_name: str, module: str) -> list[str]:
    out: list[str] = []
    for prefix in FORBIDDEN_IMPORT_PREFIXES:
        if module.startswith(prefix):
            out.append(f"{file_name}: forbidden import {module}")
    for sym in FORBIDDEN_SYMBOLS:
        if sym in module:
            out.append(f"{file_name}: forbidden symbol in import {module}")
    return out
