from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from smarthome_mdf.multi_action_vnext.requirement_types import RequirementBundle
from smarthome_mdf.multi_action_vnext.reuse_policy import ReuseTracker
from smarthome_mdf.multi_action_vnext.source_adapters import SourceObservation, SourcePool
from smarthome_mdf.multi_action_vnext.unified_runtime_matcher import viable_observations_for_bundle
from smarthome_mdf.single_scene_blueprint_complete.stable_signatures import (
    behavior_source_dedup_key,
    behavior_source_pair_key,
    behavior_temporal_dedup_signature,
    behavior_temporal_key_from_target,
    canonical_sample_signature,
)

@dataclass
class SourceUsageStats:
    attempt_count: int = 0
    accept_count: int = 0
    padding_reject_count: int = 0
    last_used_step: int = -1

@dataclass
class DiversitySourceSelector:

    reuse_tracker: ReuseTracker = field(default_factory=ReuseTracker)
    source_stats: dict[str, SourceUsageStats] = field(default_factory=dict)
    _viable_cache: dict[str, list[tuple[SourceObservation, list[dict]]]] = field(default_factory=dict)
    covered_behavior_source_pairs: set[str] = field(default_factory=set)
    covered_coarse_pairs: set[str] = field(default_factory=set)
    saturated_behavior_source_pairs: set[str] = field(default_factory=set)
    saturated_coarse_pairs: set[str] = field(default_factory=set)
    canonical_by_dedup_key: dict[str, str] = field(default_factory=dict)
    generation_step: int = 0

    def _bundle_cache_key(self, bundle: RequirementBundle, automation_instance_id: str) -> str:
        payload = json.dumps(
            {
                "scene": bundle.scene,
                "blueprint_id": getattr(bundle, "blueprint_id", None) or bundle.to_dict().get("blueprint_id"),
                "instance": automation_instance_id,
                "requirements": bundle.to_dict(),
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def viable_for_bundle(
        self,
        pool: SourcePool,
        bundle: RequirementBundle,
        automation_instance_id: str,
    ) -> list[tuple[SourceObservation, list[dict]]]:
        key = self._bundle_cache_key(bundle, automation_instance_id)
        if key not in self._viable_cache:
            viable = viable_observations_for_bundle(pool, bundle)
            viable.sort(key=lambda x: x[0].record_id)
            self._viable_cache[key] = viable
        return self._viable_cache[key]

    def _stats(self, record_id: str) -> SourceUsageStats:
        if record_id not in self.source_stats:
            self.source_stats[record_id] = SourceUsageStats()
        return self.source_stats[record_id]

    def _tiebreak(self, behavior_temporal_key: str, record_id: str, step: int) -> int:
        payload = f"{behavior_temporal_key}|{record_id}|{step}"
        return int(hashlib.sha256(payload.encode()).hexdigest()[:12], 16)

    def coarse_pair_key(
        self,
        *,
        blueprint_id: str,
        automation_instance_id: str,
        branch_id: str,
        action_path_signature: str,
        source_record_id: str,
    ) -> str:
        return f"{blueprint_id}|{automation_instance_id}|{branch_id}|{action_path_signature}|{source_record_id}"

    def coarse_pair_from_sample(self, sample: dict) -> str:
        bb = sample.get("blueprint_binding") or {}
        meta = sample.get("synthesis_metadata") or {}
        prov = sample.get("provenance") or {}
        return self.coarse_pair_key(
            blueprint_id=str(bb.get("blueprint_id") or ""),
            automation_instance_id=str(bb.get("automation_instance_id") or ""),
            branch_id=str(meta.get("branch_id") or ""),
            action_path_signature=str(meta.get("action_path_signature") or ""),
            source_record_id=str(prov.get("source_record_id") or ""),
        )

    def behavior_temporal_key_for_target(
        self,
        *,
        blueprint_id: str,
        automation_instance_id: str,
        branch_id: str,
        action_path_signature: str,
        entity_binding_signature: str = "",
    ) -> str:
        return behavior_temporal_key_from_target(
            blueprint_id=blueprint_id,
            automation_instance_id=automation_instance_id,
            branch_id=branch_id,
            action_path_signature=action_path_signature,
            entity_binding_signature=entity_binding_signature,
        )

    def unused_compatible_count(
        self,
        pool: SourcePool,
        bundle: RequirementBundle,
        automation_instance_id: str,
        behavior_temporal_key: str,
        *,
        blueprint_id: str = "",
        branch_id: str = "",
        action_path_signature: str = "",
        visual_frame_id: str = "",
    ) -> int:
        viable = self.viable_for_bundle(pool, bundle, automation_instance_id)
        count = 0
        for obs, _ in viable:
            coarse = self.coarse_pair_key(
                blueprint_id=blueprint_id,
                automation_instance_id=automation_instance_id,
                branch_id=branch_id,
                action_path_signature=action_path_signature,
                source_record_id=obs.record_id,
            )
            pair = behavior_source_pair_key(
                behavior_temporal_key=behavior_temporal_key,
                source_record_id=obs.record_id,
                visual_frame_id=visual_frame_id,
            )
            if coarse in self.saturated_coarse_pairs or coarse in self.covered_coarse_pairs:
                continue
            if pair in self.saturated_behavior_source_pairs or pair in self.covered_behavior_source_pairs:
                continue
            count += 1
        return count

    def select_source(
        self,
        pool: SourcePool,
        bundle: RequirementBundle,
        *,
        blueprint_id: str,
        automation_instance_id: str,
        branch_id: str,
        action_path_signature: str,
        entity_binding_signature: str = "",
        visual_frame_id: str = "",
    ) -> tuple[SourceObservation | None, str, list[dict]]:

        viable = self.viable_for_bundle(pool, bundle, automation_instance_id)
        if not viable:
            return None, "NO_RUNTIME_MATCH", [{"unsatisfied_requirements": ["NO_VIABLE_SOURCES"]}]

        behavior_key = self.behavior_temporal_key_for_target(
            blueprint_id=blueprint_id,
            automation_instance_id=automation_instance_id,
            branch_id=branch_id,
            action_path_signature=action_path_signature,
            entity_binding_signature=entity_binding_signature,
        )
        step = self.generation_step
        ranked: list[tuple[tuple, SourceObservation, list[dict]]] = []

        for obs, traces in viable:
            coarse = self.coarse_pair_key(
                blueprint_id=blueprint_id,
                automation_instance_id=automation_instance_id,
                branch_id=branch_id,
                action_path_signature=action_path_signature,
                source_record_id=obs.record_id,
            )
            pair = behavior_source_pair_key(
                behavior_temporal_key=behavior_key,
                source_record_id=obs.record_id,
                visual_frame_id=visual_frame_id,
            )
            if coarse in self.saturated_coarse_pairs or pair in self.saturated_behavior_source_pairs:
                continue
            if coarse in self.covered_coarse_pairs or pair in self.covered_behavior_source_pairs:
                continue
            stats = self._stats(obs.record_id)
            pair_covered = 0
            never_used = 0 if stats.attempt_count == 0 else 1
            score = (
                never_used,
                stats.padding_reject_count,
                stats.attempt_count,
                self.reuse_tracker.source_reuse(obs.record_id),
                self._tiebreak(behavior_key, obs.record_id, step),
            )
            ranked.append((score, obs, traces))

        if not ranked:
            return None, "BEHAVIOR_SOURCE_SATURATED", [{"unsatisfied_requirements": ["ALL_PAIRS_SATURATED"]}]

        ranked.sort(key=lambda x: x[0])
        _, obs, traces = ranked[0]
        stats = self._stats(obs.record_id)
        stats.attempt_count += 1
        stats.last_used_step = step
        self.reuse_tracker.record_source(obs.record_id)
        return obs, "MATCH_OK", traces

    def advance_step(self) -> None:
        self.generation_step += 1

    def register_accept(self, sample: dict) -> None:
        prov = sample.get("provenance") or {}
        rid = str(prov.get("source_record_id") or "")
        if rid:
            stats = self._stats(rid)
            stats.accept_count += 1
            self.reuse_tracker.record_source(rid)
        dedup = behavior_source_dedup_key(sample)
        self.canonical_by_dedup_key[dedup] = canonical_sample_signature(sample)
        pair = behavior_source_pair_key(
            behavior_temporal_key=behavior_temporal_dedup_signature(sample),
            source_record_id=rid,
            visual_frame_id=str((prov.get("visual_source") or {}).get("frame_id") or ""),
        )
        self.covered_behavior_source_pairs.add(pair)
        self.covered_coarse_pairs.add(self.coarse_pair_from_sample(sample))

    def register_padding_reject(self, sample: dict) -> None:
        prov = sample.get("provenance") or {}
        rid = str(prov.get("source_record_id") or "")
        if rid:
            self._stats(rid).padding_reject_count += 1
        pair = behavior_source_pair_key(
            behavior_temporal_key=behavior_temporal_dedup_signature(sample),
            source_record_id=rid,
            visual_frame_id=str((prov.get("visual_source") or {}).get("frame_id") or ""),
        )
        self.saturated_behavior_source_pairs.add(pair)
        self.saturated_coarse_pairs.add(self.coarse_pair_from_sample(sample))

    def restore_from_samples(self, samples: list[dict]) -> None:
        for sample in samples:
            self.register_accept(sample)
            prov = sample.get("provenance") or {}
            bb = sample.get("blueprint_binding") or {}
            bp = bb.get("blueprint_id", "")
            if bp:
                self.reuse_tracker.record_blueprint(str(bp))

    def blueprint_has_remaining_capacity(
        self,
        pool: SourcePool,
        bundle: RequirementBundle,
        automation_instance_id: str,
        behavior_targets: list[dict[str, str]],
    ) -> bool:
        for bt in behavior_targets:
            key = self.behavior_temporal_key_for_target(
                blueprint_id=bt["blueprint_id"],
                automation_instance_id=bt["automation_instance_id"],
                branch_id=bt["branch_id"],
                action_path_signature=bt["action_path_signature"],
                entity_binding_signature=bt.get("entity_binding_signature", ""),
            )
            if self.unused_compatible_count(
                pool,
                bundle,
                automation_instance_id,
                key,
                blueprint_id=bt["blueprint_id"],
                branch_id=bt["branch_id"],
                action_path_signature=bt["action_path_signature"],
            ) > 0:
                return True
        return False

    def unused_compatible_sources_total(self) -> int:
        total = 0
        for key, viable in self._viable_cache.items():
            _ = key
            for obs, _tr in viable:
                if obs.record_id not in self.source_stats or self.source_stats[obs.record_id].attempt_count == 0:
                    total += 1
        return total

    def stats_summary(self) -> dict[str, Any]:
        saturated_bps = len({p.split("|")[0] for p in self.saturated_behavior_source_pairs})
        return {
            "unique_sources_tracked": len(self.source_stats),
            "covered_behavior_source_pairs": len(self.covered_behavior_source_pairs),
            "saturated_behavior_source_pairs": len(self.saturated_behavior_source_pairs),
            "cached_bundle_count": len(self._viable_cache),
        }

    def diagnose_blueprint(
        self,
        pool: SourcePool,
        bundle: RequirementBundle,
        automation_instance_id: str,
        behavior_temporal_key: str,
    ) -> dict[str, Any]:
        viable = self.viable_for_bundle(pool, bundle, automation_instance_id)
        record_ids = [obs.record_id for obs, _ in viable]
        attempts: dict[str, int] = {}
        accepts: dict[str, int] = {}
        for rid in record_ids:
            st = self.source_stats.get(rid)
            if st:
                attempts[rid] = st.attempt_count
                accepts[rid] = st.accept_count
        top_rid = max(record_ids, key=lambda r: attempts.get(r, 0)) if record_ids else ""
        top_attempts = attempts.get(top_rid, 0)
        total_attempts = sum(attempts.values()) or 1
        unused = sum(
            1
            for rid in record_ids
            if behavior_source_pair_key(
                behavior_temporal_key=behavior_temporal_key,
                source_record_id=rid,
            )
            not in self.saturated_behavior_source_pairs
        )
        return {
            "candidate_source_count": len(record_ids),
            "unique_sources_attempted": len(attempts),
            "unique_sources_accepted": len([r for r, c in accepts.items() if c > 0]),
            "top_source_attempt_count": top_attempts,
            "top_source_share": round(top_attempts / total_attempts, 4) if total_attempts else 0.0,
            "unused_compatible_sources": unused,
        }
