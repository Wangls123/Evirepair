from __future__ import annotations

from pathlib import Path
from typing import Any

def write_composability_audit_md(matrix: dict[str, Any], path: Path) -> None:
    lines = [
        "# Multi-action Composability Audit (Frozen V1)",
        "",
        "Construction-only Blueprint-pair composability model. No B0/Y/conflict inputs.",
        "",
        "## Blueprint nodes",
        f"- Generated Blueprint nodes (inventory): **{matrix.get('generated_blueprint_count', 0)}**",
        f"- Grounded-capable with samples: **{matrix.get('grounded_capable_with_samples', 0)}**",
        "",
        "## Pairwise composability",
        f"- Total pair combinations: **{matrix.get('pair_combinations_total', 0)}**",
    ]
    for k, v in (matrix.get("pair_counts") or {}).items():
        lines.append(f"- {k}: **{v}**")
    lines.extend(["", "## Triple summary"])
    for k, v in (matrix.get("triple_summary") or {}).items():
        lines.append(f"- {k}: **{v}**")
    lines.extend(["", "## Top incompatibility reasons"])
    for reason, count in matrix.get("top_incompatibility_reasons") or []:
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Shared-entity opportunities", f"- Pairs with shared-entity opportunity: **{matrix.get('shared_entity_opportunities', 0)}**", ""])
    lines.extend(["## Source-mode opportunities"])
    for k, v in (matrix.get("source_mode_opportunities") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Scene-pair coverage (sample)", ""])
    for i, (k, v) in enumerate(sorted((matrix.get("scene_pair_coverage") or {}).items())):
        if i >= 20:
            lines.append(f"- ... and {len(matrix['scene_pair_coverage']) - 20} more")
            break
        lines.append(f"- `{k}`: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
