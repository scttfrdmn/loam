"""The operations — cloud-mask and band-math over Sentinel-2 COGs.

This is the compute half of what SageMaker Geospatial's EOJ did, done in ~a page of numpy.
Bands are read directly from the scene's COG hrefs via ``/vsicurl`` (rasterio), at the
overview level nearest the requested resolution — so we fetch tens of times fewer bytes than
the full 10m tile, exactly as the fieldwork deforestation tutorial does.

Each function processes ONE scene and returns a georeferenced ``Raster`` (array + transform +
CRS + nodata) — so results open correctly in GDAL/QGIS and downstream tools can turn pixels
into lat/lon. The shard runner (``run.py``) loops these over a shard's scenes and writes each
Raster as a (COG) GeoTIFF. Nothing here knows about shards, S3, or runners — pure content,
unit-testable with local rasters.
"""

from __future__ import annotations

import numpy as np

from .indices import IndexDef, bands_in
from .raster import Raster, read_band as _read_band_raster

# Sentinel-2 L2A Scene Classification (SCL) values that are cloud / cloud-shadow / cirrus.
# 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus, 11=snow(optional).
_SCL_CLOUD = {3, 8, 9, 10}


def read_band(href: str, *, target_res: float | None = None) -> np.ndarray:
    """Read a single COG band as a float32 array (no georeferencing).

    Thin wrapper over ``raster.read_band`` kept for the compute paths (and tests) that only
    need pixels. When georeferencing must be preserved, ops use ``raster.read_band`` directly.
    """
    return _read_band_raster(href, target_res=target_res).data


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
) -> Raster:
    """Evaluate one index's equation for a scene; return a georeferenced float32 Raster.

    NaN where masked. Only the bands the equation references are read. If ``scl_mask`` and an
    ``scl`` asset is present, cloudy pixels are set to NaN first (band-math over clouds is
    meaningless). The output carries the transform/CRS of the referenced bands (clipped to the
    common grid), so downstream georeferencing is exact.
    """
    needed = bands_in(index.equation)
    missing = needed - assets.keys()
    if missing:
        raise KeyError(f"scene missing bands {sorted(missing)} for {index.name}")

    rasters = {b: _read_band_raster(assets[b], target_res=target_res) for b in needed}
    # Reference georeferencing = whichever band has the largest grid (finest); we clip all to
    # the common (min) shape, and that band's transform is valid for the clipped top-left.
    ref = max(rasters.values(), key=lambda r: r.height * r.width)
    aligned = _align([r.data for r in rasters.values()])
    env = dict(zip(rasters.keys(), aligned))

    cloud = None
    if scl_mask and "scl" in assets:
        scl = _read_band_raster(assets["scl"], target_res=target_res).data
        scl = _align([scl, aligned[0]])[0]
        cloud = np.isin(np.rint(scl).astype(int), list(_SCL_CLOUD))

    # Evaluate the equation in a numpy namespace. Equations are our own catalog constants or a
    # user-supplied NAME=equation; the empty __builtins__ blocks the obvious attacks (a fuller
    # allowlist evaluator is tracked in issue #4).
    with np.errstate(invalid="ignore", divide="ignore"):
        result = eval(index.equation, {"__builtins__": {}}, env)  # noqa: S307 - constrained namespace
    result = np.asarray(result, dtype=np.float32)
    if cloud is not None:
        result = np.where(cloud, np.nan, result)

    return Raster(data=result, transform=ref.transform, crs=ref.crs, nodata=float("nan"))


def cloud_mask(assets: dict[str, str], *, target_res: float | None = 100.0) -> Raster:
    """Return a georeferenced uint8 cloud mask (1 = cloud/shadow/cirrus) from the SCL band."""
    if "scl" not in assets:
        raise KeyError("cloud-mask requires an 'scl' asset (Sentinel-2 L2A Scene Classification)")
    scl = _read_band_raster(assets["scl"], target_res=target_res)
    mask = np.isin(np.rint(scl.data).astype(int), list(_SCL_CLOUD)).astype(np.uint8)
    return Raster(data=mask, transform=scl.transform, crs=scl.crs, nodata=None)
