from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

@dataclass
class ComponentTemporal:
    component_id: str
    source_timestamp_original: str
    aligned_timestamp: str
    alignment_offset_seconds: float
    alignment_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except ValueError:
        return None

def _extract_source_timestamp(sample: dict) -> str:
    prov = sample.get("provenance") or {}
    if prov.get("synthesis_timestamp"):
        return prov["synthesis_timestamp"]
    obs = sample.get("observed") or {}
    if obs.get("timestamp"):
        return obs["timestamp"]
    for eo in sample.get("entity_observations") or []:
        if eo.get("timestamp"):
            return eo["timestamp"]
    return ""

def align_temporal_components(
    components: list[tuple[str, dict]],
    *,
    force_synthetic: bool = False,
) -> dict[str, Any]:

    datasets = {(s.get("provenance") or {}).get("source_dataset") for _, s in components}
    datasets.discard(None)
    comp_timelines: list[ComponentTemporal] = []
    originals: list[tuple[str, datetime | None, str]] = []

    for cid, sample in components:
        src_ts = _extract_source_timestamp(sample)
        dt = _parse_ts(src_ts)
        originals.append((cid, dt, src_ts))

    same_dataset = len(datasets) == 1
    all_have_ts = all(dt is not None for _, dt, _ in originals)

    if force_synthetic or len(datasets) > 1:
        alignment_mode = "CROSS_DATASET_SYNTHETIC"
        base = min((dt for _, dt, _ in originals if dt), default=datetime.now(timezone.utc))
        comp_ts = base.isoformat()
        for cid, dt, src_ts in originals:
            offset = (dt - base).total_seconds() if dt and base else 0.0
            comp_timelines.append(
                ComponentTemporal(
                    component_id=cid,
                    source_timestamp_original=src_ts,
                    aligned_timestamp=comp_ts,
                    alignment_offset_seconds=offset,
                    alignment_class=alignment_mode,
                )
            )
        return {
            "composition_timeline": {
                "alignment_mode": alignment_mode,
                "composition_timestamp": comp_ts,
                "original_timestamps_preserved": True,
                "alignment_validity_reason": "Cross-dataset synthetic alignment; originals preserved with offsets",
                "components": [c.to_dict() for c in comp_timelines],
            },
            "source_mode": "CROSS_DATASET_SYNTHETIC",
        }

    if same_dataset and all_have_ts:
        dts = [dt for _, dt, _ in originals if dt]
        span = (max(dts) - min(dts)).total_seconds() if dts else 0
        if span <= 86400:
            alignment_mode = "SAME_SOURCE_TIME_ALIGNED"
            anchor = min(dts)
            comp_ts = anchor.isoformat()
            for cid, dt, src_ts in originals:
                offset = (dt - anchor).total_seconds() if dt else 0.0
                comp_timelines.append(
                    ComponentTemporal(
                        component_id=cid,
                        source_timestamp_original=src_ts,
                        aligned_timestamp=comp_ts,
                        alignment_offset_seconds=offset,
                        alignment_class=alignment_mode,
                    )
                )
            return {
                "composition_timeline": {
                    "alignment_mode": alignment_mode,
                    "composition_timestamp": comp_ts,
                    "original_timestamps_preserved": True,
                    "alignment_validity_reason": "Same source dataset; timestamps aligned within 24h window",
                    "components": [c.to_dict() for c in comp_timelines],
                },
                "source_mode": "SAME_SOURCE" if len({s.get("provenance", {}).get("source_record_id") for _, s in components}) == 1 else "CROSS_SOURCE",
            }

    alignment_mode = "CROSS_SOURCE_TIME_ALIGNED"
    anchor = min((dt for _, dt, _ in originals if dt), default=datetime.now(timezone.utc))
    comp_ts = anchor.isoformat()
    for cid, dt, src_ts in originals:
        offset = (dt - anchor).total_seconds() if dt else 0.0
        comp_timelines.append(
            ComponentTemporal(
                component_id=cid,
                source_timestamp_original=src_ts,
                aligned_timestamp=comp_ts,
                alignment_offset_seconds=offset,
                alignment_class=alignment_mode,
            )
        )
    return {
        "composition_timeline": {
            "alignment_mode": alignment_mode,
            "composition_timestamp": comp_ts,
            "original_timestamps_preserved": True,
            "alignment_validity_reason": "Cross-source within composition; offsets recorded explicitly",
            "components": [c.to_dict() for c in comp_timelines],
        },
        "source_mode": "CROSS_SOURCE",
    }
