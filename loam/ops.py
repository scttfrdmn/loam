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

from .indices import IndexDef, bands_in, safe_eval
from .raster import Raster, read_band as _read_band_raster, reproject_raster

# Sentinel-2 L2A Scene Classification (SCL) values that are cloud / cloud-shadow / cirrus.
# 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus, 11=snow(optional).
_SCL_CLOUD = {3, 8, 9, 10}


def read_band(href: str, *, target_res: float | None = None) -> np.ndarray:
    """Read a single COG band as a float32 array (no georeferencing).

    Thin wrapper over ``raster.read_band`` kept for the compute paths (and tests) that only
    need pixels. When georeferencing must be preserved, ops use ``raster.read_band`` directly.
    """
    return _read_band_raster(href, target_res=target_res).data


def _resample_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample a 2D array to ``shape``.

    Sentinel-2 bands have different native resolutions (NIR/red 10 m, SWIR/SCL 20 m, etc.), so
    band-math like BSI (swir16 vs nir) or NDSI (green vs swir16) mixes grids. We resample every
    band up/down to the finest grid before arithmetic. Nearest-neighbour keeps it dependency-
    free and is appropriate for index math over categorical-ish spectral ratios; the coarse
    band's pixels are exactly replicated onto the fine grid (no invented values). Same shape →
    returned unchanged (the common 10 m-only case, e.g. NDVI, pays nothing).
    """
    if arr.shape == shape:
        return arr
    h, w = shape
    ri = (np.arange(h) * arr.shape[0] // h).clip(0, arr.shape[0] - 1)
    ci = (np.arange(w) * arr.shape[1] // w).clip(0, arr.shape[1] - 1)
    return arr[np.ix_(ri, ci)]


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
    # Reference grid = the finest band (largest pixel count). Resample every band ONTO that grid
    # so mixed-resolution math (e.g. BSI: 20 m swir16 with 10 m nir) broadcasts correctly and
    # stays geographically aligned. The ref band's transform is exact for the output.
    ref = max(rasters.values(), key=lambda r: r.height * r.width)
    ref_shape = (ref.height, ref.width)
    env = {b: _resample_to(r.data, ref_shape) for b, r in rasters.items()}

    cloud = None
    if scl_mask and "scl" in assets:
        scl = _read_band_raster(assets["scl"], target_res=target_res).data
        scl = _resample_to(scl, ref_shape)
        cloud = np.isin(np.rint(scl).astype(int), list(_SCL_CLOUD))

    # Evaluate the equation over the band arrays via the AST allowlist in indices.safe_eval (no
    # Python eval — a user-supplied or manifest-sourced equation can't reach the interpreter).
    # safe_eval does no arithmetic itself; each numpy op runs under this errstate context, so
    # NaN/inf/divide-by-zero handling is identical to the prior eval path. Keep it inside the with.
    with np.errstate(invalid="ignore", divide="ignore"):
        result = safe_eval(index.equation, env)
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


def resample(
    assets: dict[str, str],
    bands: list[str],
    *,
    dst_crs: str,
    dst_res: float | None = None,
    resampling: str = "bilinear",
    target_res: float | None = None,
) -> dict[str, Raster]:
    """Reproject/resample each requested band to ``dst_crs`` (+ optional ``dst_res``).

    One georeferenced Raster per band. ``target_res`` still controls the overview level read (so
    we fetch few bytes); ``dst_res``/``dst_crs`` set the output grid. All rasterio/warp detail is
    in ``raster.reproject_raster`` — this just reads and delegates per band.
    """
    missing = set(bands) - assets.keys()
    if missing:
        raise KeyError(f"scene missing bands {sorted(missing)} for resample")
    out: dict[str, Raster] = {}
    for b in bands:
        src = _read_band_raster(assets[b], target_res=target_res)
        out[b] = reproject_raster(src, dst_crs=dst_crs, dst_res=dst_res, resampling=resampling)
    return out
