"""plan — build a manifest from a search + an operation, and report status.

``plan`` is the pre-compute step: it searches the catalog, shards the scenes, attaches the
operation, and writes the manifest to S3. No pixels are read here. The result is a document a
runner fans out over via ``run-shard``.

``status`` derives progress purely from S3 (done shards = checkpoints present) — no control
plane, no job to poll.
"""

from __future__ import annotations

from . import catalog, state
from .indices import parse_spec
from .manifest import Manifest, MANIFEST_VERSION, shard_scenes


def build_manifest(
    *,
    op: str,
    collection: str,
    aoi: list[float],
    start: str,
    end: str,
    output_uri: str,
    indices: list[str] | None = None,
    max_cloud: float | None = None,
    shard_size: int = 50,
    limit: int | None = None,
    fmt: str = "cog",
    stac_url: str = catalog.DEFAULT_STAC_URL,
) -> Manifest:
    """Search, shard, and assemble a Manifest (does not write it — caller persists)."""
    params: dict = {"format": fmt}
    wanted: set[str] = {"scl"}  # always fetch SCL so ops can cloud-mask

    if op == "band-math":
        if not indices:
            raise ValueError("band-math requires --indices")
        defs = [parse_spec(s) for s in indices]
        params["indices"] = indices
        from .indices import bands_in

        for d in defs:
            wanted |= bands_in(d.equation)
    elif op == "cloud-mask":
        pass  # only needs scl, already in wanted
    else:
        raise ValueError(f"unknown op {op!r} (known: band-math, cloud-mask)")

    scenes = catalog.search(
        collection=collection,
        aoi=aoi,
        start=start,
        end=end,
        max_cloud=max_cloud,
        limit=limit,
        stac_url=stac_url,
        wanted_bands=wanted,
    )
    shards = shard_scenes(scenes, shard_size)

    return Manifest(
        version=MANIFEST_VERSION,
        op=op,
        params=params,
        collection=catalog.resolve_collection(collection),
        aoi=aoi,
        output_uri=output_uri,
        scenes=scenes,
        shards=shards,
    )


def write_manifest(manifest: Manifest, manifest_uri: str, *, region: str | None = None) -> None:
    state.put_text(manifest_uri, manifest.to_json(), region=region)


def status(manifest_uri: str, *, region: str | None = None) -> dict:
    """Return progress derived entirely from S3 (done = checkpoint object present)."""
    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))
    total = len(manifest.shards)
    done = sum(
        1
        for sh in manifest.shards
        if state.shard_done(manifest.output_uri, sh.index, region=region)
    )
    return {
        "op": manifest.op,
        "scenes": len(manifest.scenes),
        "shards_total": total,
        "shards_done": done,
        "shards_remaining": total - done,
        "complete": done == total and total > 0,
    }
