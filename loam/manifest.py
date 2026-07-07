"""The manifest — work as data.

loam's execution-agnostic contract rests on this module. A run is not a live job with an
ARN and a session; it is a *manifest* (a JSON document) describing scenes and the shards
they're grouped into, plus an operation to apply. A runner (spawn, nf-spawn, cwl-spawn, or a
laptop for-loop) reads the manifest, runs ``loam run-shard --index N`` for each shard, and
that's the whole protocol. loam never provisions or schedules anything.

State is NOT stored here. Whether a shard is *done* is derived from S3 (its output exists),
never from a field in the manifest — see ``loam.state``. The manifest is immutable once
written; progress lives in the object store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

MANIFEST_VERSION = 1


@dataclass
class Scene:
    """One source scene: an id, a datetime, and band-name → COG href map."""

    id: str
    datetime: str
    assets: dict[str, str]  # band name (e.g. "red") -> COG href (s3:// or https:// /vsicurl)


@dataclass
class Shard:
    """A unit of work a runner schedules: a contiguous slice of scenes.

    A shard is the atom. ``loam run-shard --index i`` processes ``scenes[start:end]``,
    writes outputs + a checkpoint to S3, and is idempotent (safe to re-run after a spot
    reclaim). Neighbors are invisible to it — no cross-shard state.
    """

    index: int
    scene_ids: list[str]


@dataclass
class Manifest:
    """The full plan. Serialized to JSON; the single source of truth a runner consumes."""

    version: int
    op: str  # "band-math" | "cloud-mask" | ...
    params: dict[str, Any]  # op-specific, e.g. {"indices": ["NDVI", "BSI"]}
    collection: str
    aoi: list[float]  # [west, south, east, north] WGS84
    output_uri: str  # s3://bucket/prefix/ — where shard outputs + checkpoints land
    scenes: list[Scene]
    shards: list[Shard] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(_asdict(self), indent=2, sort_keys=False)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        d = json.loads(text)
        scenes = [Scene(**s) for s in d.pop("scenes", [])]
        shards = [Shard(**s) for s in d.pop("shards", [])]
        return cls(scenes=scenes, shards=shards, **d)

    def scenes_for(self, shard_index: int) -> list[Scene]:
        """Return the Scene objects belonging to a given shard index."""
        by_id = {s.id: s for s in self.scenes}
        shard = next((sh for sh in self.shards if sh.index == shard_index), None)
        if shard is None:
            raise IndexError(f"no shard with index {shard_index} (have {len(self.shards)})")
        return [by_id[sid] for sid in shard.scene_ids]


def shard_scenes(scenes: list[Scene], shard_size: int) -> list[Shard]:
    """Group scenes into shards of at most ``shard_size`` each.

    Deterministic: shard i always contains the same scenes for a given manifest, so a
    re-run addresses the same work. That determinism is what makes retries idempotent.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be >= 1")
    shards: list[Shard] = []
    for i, start in enumerate(range(0, len(scenes), shard_size)):
        chunk = scenes[start : start + shard_size]
        shards.append(Shard(index=i, scene_ids=[s.id for s in chunk]))
    return shards


def _asdict(m: Manifest) -> dict[str, Any]:
    """asdict that preserves top-level key order (scenes/shards last for readability)."""
    return {
        "version": m.version,
        "op": m.op,
        "params": m.params,
        "collection": m.collection,
        "aoi": m.aoi,
        "output_uri": m.output_uri,
        "scenes": [asdict(s) for s in m.scenes],
        "shards": [asdict(s) for s in m.shards],
    }
