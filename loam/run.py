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

import numpy as np

from . import ops, state
from .indices import parse_spec
from .manifest import Manifest, Scene
from .raster import Raster, write_geotiff


def _save_npy(uri: str, arr: np.ndarray, *, region: str | None) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    state.put_bytes(uri, buf.getvalue(), region=region)


def _save_raster(output_uri: str, index: int, scene_id: str, name: str,
                 raster: Raster, fmt: str, *, region: str | None) -> str:
    """Write one output raster in the requested format; return the URI written."""
    if fmt == "npy":
        uri = state.output_uri_for(output_uri, index, f"{scene_id}__{name}.npy")
        _save_npy(uri, raster.data, region=region)
        return uri
    # gtiff / cog
    ext = "tif"
    uri = state.output_uri_for(output_uri, index, f"{scene_id}__{name}.{ext}")
    data = write_geotiff(uri, raster, cog=(fmt != "gtiff-plain"))
    state.put_bytes(uri, data, region=region)
    return uri


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
    raise ValueError(f"unknown op {op!r}")


def run_shard(manifest_uri: str, index: int, *, region: str | None = None, force: bool = False) -> dict:
    """Run a single shard. Returns a small summary dict (also useful for tests)."""
    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))

    if not force and state.shard_done(manifest.output_uri, index, region=region):
        return {"shard": index, "status": "skipped", "reason": "checkpoint exists"}

    fmt = manifest.params.get("format", "cog")
    scenes = manifest.scenes_for(index)
    written: list[str] = []
    failed: list[dict] = []

    for scene in scenes:
        try:
            results = _process_scene(manifest.op, manifest.params, scene)
        except Exception as e:  # a bad scene must not sink the shard; record and continue
            failed.append({"scene": scene.id, "error": str(e)})
            continue
        for name, raster in results.items():
            uri = _save_raster(manifest.output_uri, index, scene.id, name, raster, fmt, region=region)
            written.append(uri)

    # Checkpoint LAST — only now is the shard's output durable (delete-after-durable).
    summary = {
        "shard": index,
        "status": "done",
        "format": fmt,
        "scenes": len(scenes),
        "outputs": len(written),
        "failed": failed,
    }
    state.put_text(
        state.checkpoint_uri(manifest.output_uri, index),
        json.dumps(summary, indent=2),
        region=region,
    )
    return summary
