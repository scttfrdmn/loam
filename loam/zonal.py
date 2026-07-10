"""Zonal statistics — aggregate raster pixels within vector zones.

The one raster×vector op: given an existing single-band raster COG (e.g. an NDVI COG produced by a
prior ``band-math`` run) and a set of polygon zones, reduce the raster's pixels inside each zone to
per-zone statistics (count, mean, min, max, sum, median, std, percentiles). It is the second step
of a two-op workflow — ``band-math`` writes the index COG, then ``zonal-stats`` summarizes it over
zones — which keeps each op single-purpose and keeps ``run-shard`` self-contained: it reads one
raster href from the manifest, never the network/STAC.

Pure content (mirrors ``ops``): no S3, shard, or STAC knowledge. All rasterio detail (windowed
reads, reprojecting the zone into the raster CRS, masking) lives here. Cloud/nodata exclusion is
free — ``band-math`` already writes NaN over masked pixels, so we just ignore non-finite values.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

# Statistic name -> NaN-aware reducer over the 1-D array of a zone's finite in-polygon pixels.
_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.nanmean,
    "min": np.nanmin,
    "max": np.nanmax,
    "sum": np.nansum,
    "median": np.nanmedian,
    "std": np.nanstd,
}


def _percentile(name: str) -> Callable[[np.ndarray], float] | None:
    """Return a reducer for a ``pNN`` percentile stat (e.g. ``p90``), or None if not one."""
    if len(name) >= 2 and name[0] == "p" and name[1:].isdigit():
        q = float(name[1:])
        return lambda a: float(np.nanpercentile(a, q))
    return None


def _reducer(stat: str) -> Callable[[np.ndarray], float]:
    if stat in _STATS:
        return _STATS[stat]
    pct = _percentile(stat)
    if pct is not None:
        return pct
    raise ValueError(
        f"unknown stat {stat!r}; known: {', '.join(_STATS)}, count, or pNN (e.g. p90)"
    )


def zonal_stats(raster_href: str, geom_wgs84: dict, stats: list[str]) -> dict[str, Any]:
    """Reduce ``raster_href``'s pixels inside one WGS84 polygon to a ``{zs_<stat>: value}`` dict.

    ``geom_wgs84`` is a GeoJSON geometry (Polygon/MultiPolygon, lon/lat). Reads only the zone's
    window (COGs are tiled, so this fetches a few tiles, not the whole raster). ``zs_count`` is the
    number of finite pixels inside the polygon; if that's zero, every stat is ``None`` (which
    serializes to JSON ``null`` — never ``NaN``, which is invalid JSON).
    """
    import rasterio
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.warp import transform_geom

    out: dict[str, Any] = {}
    with rasterio.open(raster_href) as src:
        # Zones are WGS84 (GeoJSON per RFC 7946); the raster is usually UTM — reproject the geometry
        # into the raster CRS so the mask lines up.
        geom = transform_geom("EPSG:4326", src.crs, geom_wgs84) if src.crs else geom_wgs84
        try:
            win = geometry_window(src, [geom])
        except Exception:  # zone fully outside the raster → no pixels
            win = None
        if win is None or win.width == 0 or win.height == 0:
            data = np.empty((0,), dtype=np.float32)
        else:
            arr = src.read(1, window=win).astype(np.float32)
            # Mask AGAINST THE WINDOW's transform — not src.transform — or every zone misregisters.
            mask = geometry_mask(
                [geom], out_shape=arr.shape, transform=src.window_transform(win), invert=True
            )
            vals = arr[mask]
            data = vals[np.isfinite(vals)]

    out["zs_count"] = int(data.size)
    for stat in stats:
        if stat == "count":
            continue  # already emitted as zs_count
        reducer = _reducer(stat)
        if data.size == 0:
            out[f"zs_{stat}"] = None
            continue
        with warnings.catch_warnings():  # all-NaN handled by the size guard; silence edge warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"zs_{stat}"] = round(float(reducer(data)), 6)
    return out
