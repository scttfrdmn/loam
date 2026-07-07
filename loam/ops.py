"""The operations — cloud-mask and band-math over Sentinel-2 COGs.

This is the compute half of what SageMaker Geospatial's EOJ did, done in ~a page of numpy.
Bands are read directly from the scene's COG hrefs via ``/vsicurl`` (rasterio), at the
overview level nearest the requested resolution — so we fetch tens of times fewer bytes than
the full 10m tile, exactly as the fieldwork deforestation tutorial does.

Each function processes ONE scene and returns arrays. The shard runner (``run.py``) loops
these over a shard's scenes and writes results to S3. Nothing here knows about shards, S3, or
runners — pure content, unit-testable with local rasters.
"""

from __future__ import annotations

import numpy as np

from .indices import IndexDef, bands_in

# Sentinel-2 L2A Scene Classification (SCL) values that are cloud / cloud-shadow / cirrus.
# 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus, 11=snow(optional).
_SCL_CLOUD = {3, 8, 9, 10}


def read_band(href: str, *, target_res: float | None = None) -> np.ndarray:
    """Read a single COG band as float, optionally at a coarser overview level.

    target_res in metres. If given, we request an out_shape scaled from the native
    resolution so GDAL serves the matching overview (10-50x fewer bytes) instead of full res.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(href) as src:
        if target_res is None:
            return src.read(1).astype(np.float32)
        native = src.res[0]
        scale = max(1.0, target_res / native)
        out_h = max(1, int(src.height / scale))
        out_w = max(1, int(src.width / scale))
        return src.read(
            1, out_shape=(out_h, out_w), resampling=Resampling.average
        ).astype(np.float32)


def _align(arrays: list[np.ndarray]) -> list[np.ndarray]:
    """Clip a set of band arrays to their common (min) shape so arithmetic broadcasts."""
    min_h = min(a.shape[0] for a in arrays)
    min_w = min(a.shape[1] for a in arrays)
    return [a[:min_h, :min_w] for a in arrays]


def band_math(
    assets: dict[str, str],
    index: IndexDef,
    *,
    target_res: float | None = 100.0,
    scl_mask: bool = True,
) -> np.ndarray:
    """Evaluate one index's equation for a scene; returns a float32 array (NaN where masked).

    Only the bands the equation references are read. If ``scl_mask`` and an ``scl`` asset is
    present, cloudy pixels are set to NaN first (band-math over clouds is meaningless).
    """
    needed = bands_in(index.equation)
    missing = needed - assets.keys()
    if missing:
        raise KeyError(f"scene missing bands {sorted(missing)} for {index.name}")

    raw = {b: read_band(assets[b], target_res=target_res) for b in needed}
    aligned = _align(list(raw.values()))
    env = dict(zip(raw.keys(), aligned))

    if scl_mask and "scl" in assets:
        scl = read_band(assets["scl"], target_res=target_res)
        scl = _align([scl, aligned[0]])[0]
        cloud = np.isin(np.rint(scl).astype(int), list(_SCL_CLOUD))
    else:
        cloud = None

    # Evaluate the equation in a numpy namespace. Equations are our own catalog constants or
    # a user-supplied NAME=equation; there is no untrusted input path here, but we still
    # restrict builtins so a typo fails loudly rather than reaching into the interpreter.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = eval(index.equation, {"__builtins__": {}}, env)  # noqa: S307 - constrained namespace
    result = np.asarray(result, dtype=np.float32)
    if cloud is not None:
        result = np.where(cloud, np.nan, result)
    return result


def cloud_mask(assets: dict[str, str], *, target_res: float | None = 100.0) -> np.ndarray:
    """Return a boolean cloud mask for a scene from its SCL band (True = cloud/shadow/cirrus)."""
    if "scl" not in assets:
        raise KeyError("cloud-mask requires an 'scl' asset (Sentinel-2 L2A Scene Classification)")
    scl = read_band(assets["scl"], target_res=target_res)
    return np.isin(np.rint(scl).astype(int), list(_SCL_CLOUD))
