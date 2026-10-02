from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from smarthome_mdf.multi_action_frozen_v1.config import FORBIDDEN_FIELDS, FORBIDDEN_IMPORT_PREFIXES

PACKAGE_ROOT = Path(__file__).resolve().parent

def _scan_file(path: Path, *, skip_token_scan: bool = False) -> list[str]:
    violations: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return [f"SYNTAX_ERROR:{path}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in FORBIDDEN_IMPORT_PREFIXES:
                    if alias.name.startswith(prefix):
                        violations.append(f"FORBIDDEN_IMPORT:{path.name}:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for prefix in FORBIDDEN_IMPORT_PREFIXES:
                if mod.startswith(prefix):
                    violations.append(f"FORBIDDEN_IMPORT:{path.name}:{mod}")
    if not skip_token_scan:
        text = path.read_text(encoding="utf-8").lower()
        for token in ("gt_builder", "ars_evaluator", "parent_executor", "conflict_builder"):
            if token in text:
                violations.append(f"FORBIDDEN_TOKEN:{path.name}:{token}")
    return violations

def run_information_flow_audit() -> dict[str, Any]:
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        if path.name in {"information_flow.py", "config.py"}:
            if path.name != "information_flow.py":
                violations.extend(_scan_file(path, skip_token_scan=True))
            continue
        violations.extend(_scan_file(path))
    scripts = PACKAGE_ROOT.parents[1] / "scripts" / "multi_action_frozen_v1"
    if scripts.is_dir():
        for path in sorted(scripts.glob("*.py")):
            violations.extend(_scan_file(path))
    return {
        "information_flow_violation_count": len(violations),
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }
