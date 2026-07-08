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
    bands: list[str] | None = None,
    dst_crs: str | None = None,
    dst_res: float | None = None,
    resampling: str = "bilinear",
    max_cloud: float | None = None,
    shard_size: int = 50,
    limit: int | None = None,
    fmt: str = "cog",
    target_res: float | None = 100.0,
    stac_url: str = catalog.DEFAULT_STAC_URL,
) -> Manifest:
    """Search, shard, and assemble a Manifest (does not write it — caller persists).

    ``target_res`` (metres) sets the overview level ops read. Default 100m is fine for
    continental-scale change detection; pass ``None`` for native full resolution when the
    features of interest are small (e.g. Sentinel-2 10m for fairy-circle detection).
    """
    params: dict = {"format": fmt, "target_res": target_res}
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
    elif op == "resample":
        if not bands:
            raise ValueError("resample requires --bands")
        if not dst_crs:
            raise ValueError("resample requires --dst-crs")
        params.update(bands=bands, dst_crs=dst_crs, dst_res=dst_res, resampling=resampling)
        wanted |= set(bands)
    else:
        raise ValueError(f"unknown op {op!r} (known: band-math, cloud-mask, resample)")

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


def status(manifest_uri: str, *, region: str | None = None, detail: bool = False) -> dict:
    """Return progress derived entirely from S3 (done = checkpoint object present).

    With ``detail=True``, also aggregate a **job ledger** from the per-shard summaries that
    ``run_shard`` wrote into each checkpoint — total outputs/bytes/seconds and a failed-scene
    rollup, plus per-shard rows. This is read-only (no writes at status time); a shard whose
    checkpoint is absent or unreadable simply doesn't contribute a row.
    """
    import json

    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))
    total = len(manifest.shards)
    done = sum(
        1
        for sh in manifest.shards
        if state.shard_done(manifest.output_uri, sh.index, region=region)
    )
    out = {
        "op": manifest.op,
        "scenes": len(manifest.scenes),
        "shards_total": total,
        "shards_done": done,
        "shards_remaining": total - done,
        "complete": done == total and total > 0,
    }
    if not detail:
        return out

    rows: list[dict] = []
    for sh in manifest.shards:
        cp = state.checkpoint_uri(manifest.output_uri, sh.index)
        if not state.exists(cp, region=region):
            continue
        try:
            rows.append(json.loads(state.get_text(cp, region=region)))
        except (ValueError, OSError):
            continue  # a malformed/partial checkpoint must not sink the whole status view
    out["ledger"] = {
        "outputs": sum(r.get("outputs", 0) for r in rows),
        "bytes_written": sum(r.get("bytes_written", 0) for r in rows),
        "seconds": round(sum(r.get("seconds", 0.0) for r in rows), 3),
        "failed_scenes": sum(len(r.get("failed", [])) for r in rows),
        "shards": sorted(rows, key=lambda r: r.get("shard", 0)),
    }
    return out
