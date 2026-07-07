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
"""

from __future__ import annotations

import io
import json

import numpy as np

from . import ops, state
from .indices import parse_spec
from .manifest import Manifest, Scene


def _save_npy(uri: str, arr: np.ndarray, *, region: str | None) -> None:
    buf = io.BytesIO()
    np.save(buf, arr)
    state.put_bytes(uri, buf.getvalue(), region=region)


def _process_scene(op: str, params: dict, scene: Scene) -> dict[str, np.ndarray]:
    """Apply the manifest's op to one scene; return {output_name: array}."""
    if op == "band-math":
        out: dict[str, np.ndarray] = {}
        for spec in params["indices"]:
            idx = parse_spec(spec)
            out[idx.name] = ops.band_math(scene.assets, idx)
        return out
    if op == "cloud-mask":
        return {"mask": ops.cloud_mask(scene.assets).astype(np.uint8)}
    raise ValueError(f"unknown op {op!r}")


def run_shard(manifest_uri: str, index: int, *, region: str | None = None, force: bool = False) -> dict:
    """Run a single shard. Returns a small summary dict (also useful for tests)."""
    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))

    if not force and state.shard_done(manifest.output_uri, index, region=region):
        return {"shard": index, "status": "skipped", "reason": "checkpoint exists"}

    scenes = manifest.scenes_for(index)
    written: list[str] = []
    failed: list[dict] = []

    for scene in scenes:
        try:
            results = _process_scene(manifest.op, manifest.params, scene)
        except Exception as e:  # a bad scene must not sink the shard; record and continue
            failed.append({"scene": scene.id, "error": str(e)})
            continue
        for name, arr in results.items():
            uri = state.output_uri_for(manifest.output_uri, index, f"{scene.id}__{name}.npy")
            _save_npy(uri, arr, region=region)
            written.append(uri)

    # Checkpoint LAST — only now is the shard's output durable (delete-after-durable).
    summary = {
        "shard": index,
        "status": "done",
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
