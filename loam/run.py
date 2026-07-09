"""run-shard — the executor-agnostic atom.

``loam run-shard --manifest <uri> --index N`` is the single command every runner schedules:
spawn as an instance ``--command``, an nf-spawn Nextflow process, a cwl-spawn tool, or a bare
for-loop on a laptop. It is:

  * idempotent   — if the shard's checkpoint already exists in S3, it's a no-op (spot-safe
                   resume: a reclaimed shard is simply re-run, losing at most one in flight).
  * self-contained — needs only the manifest URI + index; nothing about its neighbors.
  * delete-after-durable — writes the checkpoint LAST, after outputs are durable in S3, so
                   "checkpoint exists" strictly implies "output is safe".

That's the whole contract. loam never launches this; a runner does.

Output format (``params["format"]`` on the manifest, default ``cog``):
  * ``cog`` / ``gtiff`` — georeferenced (Cloud-Optimized) GeoTIFF (opens in GDAL/QGIS; the
    format downstream geo tools expect, e.g. the fieldwork SAM prep step).
  * ``npy`` — headerless NumPy array, for the fast/no-geo path.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np

from . import ops, state
from .indices import parse_spec
from .manifest import Manifest, Scene
from .raster import Raster, write_geotiff


def _save_npy(uri: str, arr: np.ndarray, *, region: str | None) -> int:
    buf = io.BytesIO()
    np.save(buf, arr)
    data = buf.getvalue()
    state.put_bytes(uri, data, region=region)
    return len(data)


def _save_raster(output_uri: str, index: int, scene_id: str, name: str,
                 raster: Raster, fmt: str, *, region: str | None) -> tuple[str, int]:
    """Write one output raster in the requested format; return (uri, bytes_written)."""
    if fmt == "npy":
        uri = state.output_uri_for(output_uri, index, f"{scene_id}__{name}.npy")
        nbytes = _save_npy(uri, raster.data, region=region)
        return uri, nbytes
    # gtiff / cog
    ext = "tif"
    uri = state.output_uri_for(output_uri, index, f"{scene_id}__{name}.{ext}")
    data = write_geotiff(uri, raster, cog=(fmt != "gtiff-plain"))
    state.put_bytes(uri, data, region=region)
    return uri, len(data)


def _process_scene(op: str, params: dict, scene: Scene) -> dict[str, Raster]:
    """Apply the manifest's op to one scene; return {output_name: Raster}.

    ``params["target_res"]`` (metres) controls the overview level read: None = native full
    resolution (e.g. 10m Sentinel-2 — needed to resolve small features), a number = downsample
    for speed/memory. It defaults to None here because the SAM/fairy-circle use case needs full
    res; the CLI default (100m) is set at plan time in plan.build_manifest.
    """
    target_res = params.get("target_res")
    if op == "band-math":
        out: dict[str, Raster] = {}
        for spec in params["indices"]:
            idx = parse_spec(spec)
            out[idx.name] = ops.band_math(scene.assets, idx, target_res=target_res)
        return out
    if op == "cloud-mask":
        return {"mask": ops.cloud_mask(scene.assets, target_res=target_res)}
    if op == "resample":
        return ops.resample(
            scene.assets,
            params["bands"],
            dst_crs=params["dst_crs"],
            dst_res=params.get("dst_res"),
            resampling=params.get("resampling", "bilinear"),
            target_res=target_res,
        )
    raise ValueError(f"unknown op {op!r}")


def _composite_shard(
    params: dict, scenes: list[Scene]
) -> tuple[str, Raster | None, list[dict]]:
    """Reduce a shard's scenes (one tile's time series) into one composite Raster.

    Reads each date via ``ops.scene_layer`` so an unreadable date is dropped and recorded rather
    than sinking the tile. Returns (output_name, raster_or_None, failed); raster is None if no date
    survives (caller marks the shard failed).
    """
    target_res = params.get("target_res")
    idx, band = params.get("index"), params.get("band")
    name = idx or band or "composite"

    layers: list[Raster] = []
    failed: list[dict] = []
    for scene in scenes:
        try:
            layers.append(ops.scene_layer(
                scene.assets, index=idx, band=band, target_res=target_res,
            ))
        except Exception as e:  # a bad date must not sink the tile; drop + record
            failed.append({"scene": scene.id, "error": str(e)})

    if not layers:
        return name, None, failed
    raster = ops.reduce_layers(layers, params.get("reducer", "median"))
    return name, raster, failed


def _enrich_rows(
    output_uri: str, index: int, scene: Scene, params: dict, *, region: str | None
) -> tuple[str, int]:
    """Reverse-geocode one chunk of points; write the enriched rows. Returns (uri, bytes)."""
    from . import vector

    fmt = params.get("format", "csv")
    text = state.get_text(scene.assets["rows"], region=region)
    rows, coords = vector.read_points(
        text, fmt, lat_field=params.get("lat_field"), lon_field=params.get("lon_field")
    )
    enrich = vector.reverse_geocode(coords, backend=params.get("backend", "offline"))
    out_text = vector.write_enriched(rows, enrich, fmt)
    uri = state.output_uri_for(output_uri, index, f"{scene.id}__enriched.{fmt}")
    state.put_text(uri, out_text, region=region)
    return uri, len(out_text.encode("utf-8"))


def run_shard(manifest_uri: str, index: int, *, region: str | None = None, force: bool = False) -> dict:
    """Run a single shard. Returns a small summary dict (also useful for tests)."""
    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))

    if not force and state.shard_done(manifest.output_uri, index, region=region):
        return {"shard": index, "status": "skipped", "reason": "checkpoint exists"}

    fmt = manifest.params.get("format", "cog")
    scenes = manifest.scenes_for(index)
    written: list[str] = []
    failed: list[dict] = []
    bytes_written = 0
    started = time.monotonic()

    if manifest.op == "reverse-geocode":
        # Row op: enrich each chunk's points and write one enriched file per chunk.
        for scene in scenes:
            try:
                uri, nbytes = _enrich_rows(
                    manifest.output_uri, index, scene, manifest.params, region=region
                )
            except Exception as e:  # a bad chunk must not sink the shard; record and continue
                failed.append({"scene": scene.id, "error": str(e)})
                continue
            written.append(uri)
            bytes_written += nbytes
    elif manifest.op == "temporal-composite":
        # Shard-level op: reduce the whole tile's time series into ONE output. A bad date is
        # dropped (recorded in failed); the shard fails only if no date survives.
        name, raster, failed = _composite_shard(manifest.params, scenes)
        if raster is not None:
            uri, nbytes = _save_raster(
                manifest.output_uri, index, "composite", name, raster, fmt, region=region
            )
            written.append(uri)
            bytes_written += nbytes
    else:
        for scene in scenes:
            try:
                results = _process_scene(manifest.op, manifest.params, scene)
            except Exception as e:  # a bad scene must not sink the shard; record and continue
                failed.append({"scene": scene.id, "error": str(e)})
                continue
            for name, raster in results.items():
                uri, nbytes = _save_raster(
                    manifest.output_uri, index, scene.id, name, raster, fmt, region=region
                )
                written.append(uri)
                bytes_written += nbytes

    # Checkpoint LAST — only now is the shard's output durable (delete-after-durable). The summary
    # doubles as the job-ledger row `loam status --detail` aggregates (bytes/seconds/failures).
    summary = {
        "shard": index,
        "status": "done",
        "format": fmt,
        "scenes": len(scenes),
        "outputs": len(written),
        "bytes_written": bytes_written,
        "seconds": round(time.monotonic() - started, 3),
        "failed": failed,
    }
    state.put_text(
        state.checkpoint_uri(manifest.output_uri, index),
        json.dumps(summary, indent=2),
        region=region,
    )
    return summary
