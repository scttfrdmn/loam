"""Raster — an array that remembers where it is on Earth.

loam's operations don't just compute pixels; their outputs must be usable in QGIS/GDAL and
must let downstream tools georeference results (the fieldwork SAM step turns detections into
lat/lon polygons using exactly this transform). So ops carry a ``Raster`` — the array plus its
affine ``transform``, ``crs``, and ``nodata`` — end to end, and ``run`` writes it as a
(Cloud-Optimized) GeoTIFF.

This module is the only place that knows GDAL/rasterio write details. ``ops`` builds Rasters;
``run`` persists them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Raster:
    """A single-band georeferenced array.

    ``transform`` is a 6-tuple (rasterio Affine coefficients a,b,c,d,e,f); we keep it as a
    plain tuple so a Raster is trivially serializable and rasterio-version-agnostic.
    ``crs`` is a string (e.g. "EPSG:32629") — whatever rasterio's ``CRS.to_string`` produced.
    """

    data: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str | None
    nodata: float | None = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])


def read_band(
    href: str, *, target_res: float | None = None
) -> Raster:
    """Read one COG band as a georeferenced float32 Raster.

    target_res in metres: if given, request an out_shape scaled from the native resolution so
    GDAL serves the matching overview (10-50x fewer bytes). When we downsample, the pixel size
    grows, so we scale the affine transform to match the returned grid — otherwise the output
    would be georeferenced at the wrong resolution.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(href) as src:
        crs = src.crs.to_string() if src.crs else None
        if target_res is None:
            data = src.read(1).astype(np.float32)
            transform = src.transform
        else:
            native = src.res[0]
            scale = max(1.0, target_res / native)
            out_h = max(1, int(src.height / scale))
            out_w = max(1, int(src.width / scale))
            data = src.read(
                1, out_shape=(out_h, out_w), resampling=Resampling.average
            ).astype(np.float32)
            # Scale the transform to the actual returned shape (x and y independently, since
            # rounding out_h/out_w can make the two scale factors differ slightly).
            sx = src.width / out_w
            sy = src.height / out_h
            transform = src.transform * rasterio.Affine.scale(sx, sy)
        return Raster(
            data=data,
            transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
            crs=crs,
            nodata=src.nodata,
        )


def reproject_raster(
    raster: Raster,
    *,
    dst_crs: str,
    dst_res: float | None = None,
    resampling: str = "bilinear",
) -> Raster:
    """Reproject/resample a Raster to ``dst_crs`` (and optionally ``dst_res`` metres/deg).

    Uses ``rasterio.warp`` to warp onto a grid derived from the source footprint. If ``dst_res``
    is given it fixes the output pixel size; otherwise rasterio picks a resolution that preserves
    the source's pixel count. ``resampling`` is any ``rasterio.enums.Resampling`` name
    (nearest|bilinear|cubic|average|…) — use ``nearest`` for categorical bands (e.g. SCL).
    Masked pixels stay NaN. This is rasterio detail, so it lives here, not in ``ops``.
    """
    import rasterio
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    if raster.crs is None:
        raise ValueError("cannot reproject a raster with no CRS")
    try:
        method = Resampling[resampling]
    except KeyError as e:
        raise ValueError(f"unknown resampling {resampling!r}") from e

    src_transform = rasterio.Affine(*raster.transform)
    h, w = raster.height, raster.width
    left, bottom, right, top = array_bounds(h, w, src_transform)

    kw = {"resolution": dst_res} if dst_res is not None else {}
    dst_transform, dst_w, dst_h = calculate_default_transform(
        raster.crs, dst_crs, w, h, left, bottom, right, top, **kw
    )

    nodata = raster.nodata if raster.nodata is not None else float("nan")
    dst = np.full((dst_h, dst_w), nodata, dtype=raster.data.dtype)
    reproject(
        source=raster.data,
        destination=dst,
        src_transform=src_transform,
        src_crs=raster.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=raster.nodata,
        dst_nodata=nodata,
        resampling=method,
    )
    return Raster(
        data=dst,
        transform=(
            dst_transform.a, dst_transform.b, dst_transform.c,
            dst_transform.d, dst_transform.e, dst_transform.f,
        ),
        crs=dst_crs,
        nodata=raster.nodata,
    )


def write_geotiff(uri_or_path: str, raster: Raster, *, cog: bool = True) -> bytes:
    """Serialize a Raster to (COG) GeoTIFF bytes and return them.

    Returns the encoded bytes so the caller (run.py) can hand them to loam.state for S3/local
    write — keeping this module free of any storage knowledge. Writing goes through a rasterio
    MemoryFile so we never touch the filesystem here.
    """
    import rasterio
    from rasterio.io import MemoryFile

    data = raster.data
    if data.ndim != 2:
        raise ValueError(f"expected a single-band 2D array, got shape {data.shape}")

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": raster.height,
        "width": raster.width,
        "count": 1,
        "dtype": data.dtype.name,
        "transform": rasterio.Affine(*raster.transform),
        "crs": raster.crs,
    }
    if raster.nodata is not None:
        profile["nodata"] = raster.nodata
    if cog:
        # Cloud-Optimized: tiled + internal overviews + compression. Written directly via the
        # GTiff driver's COG-compatible options (works without the separate COG driver).
        profile.update(tiled=True, blockxsize=256, blockysize=256, compress="deflate")

    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(data, 1)
            if cog:
                factors = _overview_factors(raster.height, raster.width)
                if factors:
                    dst.build_overviews(factors, rasterio.enums.Resampling.average)
                    dst.update_tags(ns="rio_overview", resampling="average")
        return mem.read()


def _overview_factors(h: int, w: int) -> list[int]:
    """Powers-of-two overview levels down to ~256px on the long side (COG convention)."""
    factors: list[int] = []
    f = 2
    while max(h, w) // f >= 256:
        factors.append(f)
        f *= 2
    return factors
