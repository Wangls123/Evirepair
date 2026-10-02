from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smarthome_mdf.single_scene_blueprint_complete.action_path_inventory import ExecutableActionPath

@dataclass(frozen=True)
class GenerationPathRequest:
    blueprint_id: str
    target_branch: str
    target_signature: str
    target_template_key: str
    strict_path_key: str
    service: str = ""
    parameter_keys: tuple[str, ...] = ()
    template_provenance: str = ""
    target_entity_raw: str | None = None

    @classmethod
    def from_behavior_target(cls, target: Any) -> GenerationPathRequest:

        tmpl_key = target.action_template_key
        raw = tmpl_key.split("|", 1)[1] if "|" in tmpl_key else ""
        provenance = raw if raw.startswith(("!input ", "{{")) else ""
        return cls(
            blueprint_id=target.blueprint_id,
            target_branch=target.branch_id,
            target_signature=target.action_path_signature,
            target_template_key=tmpl_key,
            strict_path_key=target.strict_path_key or f"{target.branch_id}::{target.action_path_signature}::{tmpl_key}",
            template_provenance=provenance,
            target_entity_raw=raw or None,
        )

    @classmethod
    def from_executable_path(cls, path: ExecutableActionPath) -> GenerationPathRequest:
        raw = path.target_entity or ""
        provenance = raw
        if raw.startswith("!input "):
            provenance = raw
        elif raw.startswith("{{") and raw.endswith("}}"):
            provenance = raw
        return cls(
            blueprint_id=path.blueprint_id,
            target_branch=path.branch_id,
            target_signature=path.action_path_signature,
            target_template_key=path.action_template_key,
            strict_path_key=path.path_key,
            service=path.service,
            parameter_keys=tuple(path.parameter_keys or ()),
            template_provenance=provenance,
            target_entity_raw=raw or None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "target_branch": self.target_branch,
            "target_signature": self.target_signature,
            "target_template_key": self.target_template_key,
            "strict_path_key": self.strict_path_key,
            "service": self.service,
            "parameter_keys": list(self.parameter_keys),
            "template_provenance": self.template_provenance,
            "target_entity_raw": self.target_entity_raw,
        }
