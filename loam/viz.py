"""Map viewer — render a completed run's COGs onto a slippy map (replaces the SM Studio map).

``loam view`` is a read-only *consumer* of outputs (like ``status``): it reads the (Cloud-Optimized)
GeoTIFFs a run wrote, colorizes each to a PNG, and writes ONE self-contained ``view.html`` with the
overlays on a Leaflet basemap. No tile server, no running process — the same static file works for
local and s3:// outputs, and opens in any browser.

Pure content (mirrors ``ops``/``vector``): this module knows nothing about S3, shards, or runners —
``cli`` handles discovery and hands Rasters here. Rendering uses **folium** (optional ``[viz]``
extra); colormaps are small hand-rolled numpy LUTs so there's no matplotlib/Pillow dependency, and
PNG encoding goes through GDAL (rasterio) which is already core.
"""

from __future__ import annotations

import base64

import numpy as np

from .raster import Raster, reproject_raster


def _ramp(stops: list[tuple[int, tuple[int, int, int]]]) -> np.ndarray:
    """Build a 256×3 uint8 LUT by linearly interpolating between (position, rgb) stops."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        for i in range(p0, p1 + 1):
            t = (i - p0) / (p1 - p0) if p1 > p0 else 0.0
            lut[i] = [round(c0[k] + t * (c1[k] - c0[k])) for k in range(3)]
    return lut


# Built-in colormaps (256×3 uint8). Enough for the common loam outputs; no matplotlib dependency.
_LUTS: dict[str, np.ndarray] = {
    # green ramp for vegetation indices (NDVI, GNDVI, SAVI, …)
    "greens": _ramp([(0, (247, 252, 245)), (128, (116, 196, 118)), (255, (0, 68, 27))]),
    # red→yellow→green diverging (good for NDVI-like -1..1 fields)
    "rdylgn": _ramp([(0, (165, 0, 38)), (128, (255, 255, 191)), (255, (0, 104, 55))]),
    # perceptually-uniform default (viridis-ish)
    "viridis": _ramp([(0, (68, 1, 84)), (85, (59, 82, 139)), (170, (33, 145, 140)),
                      (255, (253, 231, 37))]),
    # binary for masks (0 → transparent handled via alpha; 1 → red)
    "binary": _ramp([(0, (0, 0, 0)), (255, (215, 48, 39))]),
}

# Index name (from the ``__<name>`` output suffix) → default colormap.
_INDEX_CMAP: dict[str, str] = {
    "NDVI": "rdylgn", "GNDVI": "greens", "SAVI": "greens", "EVI": "greens",
    "NDWI": "viridis", "MNDWI": "viridis", "NDMI": "viridis",
    "mask": "binary",
}


def cmap_for(name: str) -> str:
    """Pick a colormap for an output named ``<scene>__<name>`` (default viridis)."""
    return _INDEX_CMAP.get(name, "viridis")


def colorize(data: np.ndarray, *, cmap: str = "viridis", nodata: float | None = None) -> np.ndarray:
    """Map a 2D array to an RGBA uint8 image via a LUT. Masked pixels get alpha 0 (transparent).

    The NaN/nodata mask is computed BEFORE the min/max stretch (a NaN would poison the range). The
    binary colormap treats the data as a 0/1 mask (no continuous stretch): 0 → transparent, else red.
    """
    lut = _LUTS.get(cmap, _LUTS["viridis"])
    arr = data.astype(np.float64)
    masked = ~np.isfinite(arr)
    if nodata is not None:
        masked |= arr == nodata

    if cmap == "binary":
        idx = np.where(arr > 0, 255, 0).astype(np.uint8)
        masked |= arr <= 0  # only paint the "true" pixels; background is transparent
    else:
        finite = arr[~masked]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        span = hi - lo or 1.0
        scaled = np.clip((arr - lo) / span, 0.0, 1.0)
        idx = np.nan_to_num(scaled * 255).astype(np.uint8)

    rgb = lut[idx]  # (H, W, 3)
    alpha = np.where(masked, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])  # (H, W, 4) RGBA


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an (H, W, 4) RGBA uint8 array as PNG bytes via GDAL (no Pillow)."""
    import warnings

    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.io import MemoryFile

    h, w = rgba.shape[:2]
    profile = {"driver": "PNG", "width": w, "height": h, "count": 4, "dtype": "uint8"}
    with warnings.catch_warnings():
        # a PNG has no geotransform — that's expected here (georeferencing lives in the overlay
        # bounds, not the image), so silence GDAL's "not georeferenced" note.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile() as mem:
            with mem.open(**profile) as dst:
                for b in range(4):
                    dst.write(rgba[:, :, b], b + 1)
            return mem.read()


def overlay_bounds(raster: Raster) -> list[list[float]]:
    """Return a folium ImageOverlay bounds ``[[south, west], [north, east]]`` in WGS84.

    Reprojects the raster to EPSG:4326 (reusing ``raster.reproject_raster``) and takes its bounds.
    Note the axis flip: folium wants lat/lon (south,west / north,east), not the raster's x,y order.
    """
    import rasterio
    from rasterio.transform import array_bounds

    r = raster if raster.crs in ("EPSG:4326", "epsg:4326") else reproject_raster(
        raster, dst_crs="EPSG:4326", resampling="nearest"
    )
    west, south, east, north = array_bounds(r.height, r.width, rasterio.Affine(*r.transform))
    return [[south, west], [north, east]]


def build_map(layers: list[dict], *, fit: list[float] | None = None):
    """Build a folium Map from prepared layers. Returns the Map (caller writes ``._repr_html_``).

    ``layers`` items: ``{"name": str, "png": bytes, "bounds": [[s,w],[n,e]]}``. ``fit`` is an
    optional ``[W, S, E, N]`` AOI (from a manifest) for the initial extent; otherwise the union of
    the overlay bounds is used. This is the only function that touches folium — lazy-imported so the
    core install needs no viewer deps.
    """
    try:
        import folium
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError("loam view needs the 'viz' extra: pip install 'loam-geo[viz]'") from e

    m = folium.Map(tiles="OpenStreetMap")
    for layer in layers:
        uri = "data:image/png;base64," + base64.b64encode(layer["png"]).decode("ascii")
        folium.raster_layers.ImageOverlay(
            image=uri, bounds=layer["bounds"], name=layer["name"], opacity=0.8,
        ).add_to(m)
    folium.LayerControl().add_to(m)

    if fit is not None:
        west, south, east, north = fit
        m.fit_bounds([[south, west], [north, east]])
    elif layers:
        souths = [b[0][0] for b in (lyr["bounds"] for lyr in layers)]
        wests = [b[0][1] for b in (lyr["bounds"] for lyr in layers)]
        norths = [b[1][0] for b in (lyr["bounds"] for lyr in layers)]
        easts = [b[1][1] for b in (lyr["bounds"] for lyr in layers)]
        m.fit_bounds([[min(souths), min(wests)], [max(norths), max(easts)]])
    return m
