"""Compute-shape estimation — describe the work's compute demand, never provision it.

`loam plan` knows a shard = N scenes × the bands an op reads × pixels at ``target_res``. That is
enough to *estimate* how much compute each shard demands — bands read, decoded bytes read, peak
working-set RAM, and a rough runtime — with **zero pixel reads and zero network I/O**. This is the
loam-side signal ``truffle`` (pick the instance) / a cost model / ``loam dispatch`` consume.

Execution-agnostic contract: loam *describes* demand; it never chooses an instance or provisions.
So this module is deliberately pure — it imports only ``indices.bands_in`` (a regex) and computes
from the in-memory manifest. It must NEVER import ``ops``/``raster``/``rasterio``/``pystac_client``
or open a COG; ``tests/test_core.py`` enforces that with an import scan.

All numbers are ORDER-OF-MAGNITUDE sizing estimates from documented constants below — knobs for
right-sizing a box, not SLAs. ``approx_bytes_read`` is *decoded* working volume (float32 arrays),
not wire bytes: COGs are compressed and we read the overview nearest ``target_res``, so bytes on
the wire are much smaller. Treat it as an upper-ish proxy.
"""

from __future__ import annotations

from .indices import bands_in

# ── Sizing knobs (order-of-magnitude, overridable) ──────────────────────────
DTYPE_BYTES = 4  # ops read bands as float32
THROUGHPUT_BYTES_PER_S = 1e8  # ~100 MB/s decoded-equivalent over /vsicurl COG reads
OPEN_OVERHEAD_S = 0.5  # per-COG open/latency floor (small coarse shards are latency-bound)

# Per-collection nominal full tile: (pixels_per_side_at_native, native_res_metres). Matched by
# substring on the resolved STAC collection id stored in the manifest. Sentinel-2 L2A tiles are
# ~110 km at 10 m → 10980 px; Landsat C2 L2 ~185 km at 30 m → ~7811 px.
_NOMINAL_TILE: dict[str, tuple[int, float]] = {
    "sentinel-2": (10980, 10.0),
    "landsat": (7811, 30.0),
}
_DEFAULT_TILE = (10980, 10.0)  # assume Sentinel-2 when unknown

# Sentinel-2 per-band native resolution (metres). Bands differ (10/20/60 m), so each band's grid
# is sized correctly instead of assuming the finest for all. Unknown bands default to the tile's
# native res. ``scl`` is 20 m — matters for cloud-mask peak RAM.
_BAND_NATIVE_RES: dict[str, float] = {
    "blue": 10.0, "green": 10.0, "red": 10.0, "nir": 10.0,
    "rededge1": 20.0, "rededge2": 20.0, "rededge3": 20.0,
    "nir08": 20.0, "swir16": 20.0, "swir22": 20.0, "scl": 20.0,
    "coastal": 60.0, "nir09": 60.0, "aot": 60.0, "wvp": 60.0,
}


def _tile_for(collection: str) -> tuple[int, float]:
    for key, tile in _NOMINAL_TILE.items():
        if key in collection:
            return tile
    return _DEFAULT_TILE


def _px_side(band: str, target_res: float | None, nominal_px: int, native_res: float) -> int:
    """Pixels/side loam would read for one band, mirroring ops.read_band's overview choice.

    ``read_band`` uses ``scale = max(1, target_res/band_native)`` — so you can never read finer
    than the band's native resolution. ``target_res=None`` means native full-res.
    """
    band_native = _BAND_NATIVE_RES.get(band, native_res)
    r = band_native if target_res is None else max(target_res, band_native)
    return max(1, int(nominal_px * native_res / r))


def _band_px(band: str, target_res: float | None, nominal_px: int, native_res: float) -> int:
    side = _px_side(band, target_res, nominal_px, native_res)
    return side * side


def _bands_and_outputs(op: str, params: dict) -> tuple[list[str], int, list[int]]:
    """Return (bands_read, n_outputs, per_output_band_counts) for an op — matches run._process_scene.

    ``per_output_band_counts`` is the number of input bands each output holds simultaneously; its
    max drives peak RAM (a single scene's largest concurrent working set).
    """
    if op == "band-math":
        specs = params.get("indices", [])
        per_index = [len(bands_in(_equation_of(s))) for s in specs]
        bands = sorted({b for s in specs for b in bands_in(_equation_of(s))} | {"scl"})
        return bands, len(specs), (per_index or [0])
    if op == "cloud-mask":
        return ["scl"], 1, [1]
    if op == "resample":
        bands = list(params.get("bands", []))
        # one output per band; each output is produced from a single input band
        return bands, len(bands), [1] if bands else [0]
    if op == "temporal-composite":
        idx, band = params.get("index"), params.get("band")
        read = (sorted(bands_in(_equation_of(idx)) | {"scl"}) if idx
                else ([band, "scl"] if band else []))
        return read, 1, [len(read)]  # one mosaic output
    return [], 0, [0]


def _is_raster_op(op: str) -> bool:
    return op in ("band-math", "cloud-mask", "resample", "temporal-composite")


def _equation_of(spec: str) -> str:
    """Equation string of an index spec (``NAME=eq`` custom, or a catalog name)."""
    from .indices import parse_spec

    return parse_spec(spec).equation


def shape_for(op: str, params: dict, n_scenes: int, collection: str) -> dict:
    """Estimate a shard's compute shape from metadata alone (no pixel reads, no network).

    Returns a dict: scenes, bands_read, outputs, approx_bytes_read, peak_rss_bytes, est_seconds.
    """
    if not _is_raster_op(op):
        # Row ops (reverse-geocode) aren't pixel-bound; compute-shape's byte/RAM model doesn't
        # apply. Report a trivial shape keyed on scene (chunk) count so status/dispatch still work.
        return {
            "scenes": n_scenes, "bands_read": 0, "outputs": 1,
            "approx_bytes_read": 0, "peak_rss_bytes": 0, "est_seconds": 0.0,
        }

    nominal_px, native_res = _tile_for(collection)
    target_res = params.get("target_res")
    bands, n_out, per_out_counts = _bands_and_outputs(op, params)

    band_px = {b: _band_px(b, target_res, nominal_px, native_res) for b in bands}
    bytes_per_scene = sum(band_px.values()) * DTYPE_BYTES
    approx_bytes_read = n_scenes * bytes_per_scene

    finest_px = max(band_px.values()) if band_px else 0
    if op == "temporal-composite":
        # A composite materializes the WHOLE tile stack (all n_scenes layers) at once to reduce
        # over time → peak scales with scene count, not one scene. This is why full-res composites
        # are refused (see plan.build_manifest) and target_res is the memory knob.
        peak_rss_bytes = finest_px * (n_scenes + 1) * DTYPE_BYTES
    else:
        # Peak working set = ONE scene (run_shard loops scenes sequentially, freeing each). For
        # band-math a scene holds every index's result at once, but each index loads only its own
        # bands → max_i(bands_i) input arrays + n_outputs result arrays (+1 slack for transients).
        peak_arrays = max(per_out_counts) + n_out + 1
        peak_rss_bytes = finest_px * peak_arrays * DTYPE_BYTES

    est_seconds = n_scenes * len(bands) * OPEN_OVERHEAD_S + approx_bytes_read / THROUGHPUT_BYTES_PER_S

    return {
        "scenes": n_scenes,
        "bands_read": len(bands),
        "outputs": n_out,
        "approx_bytes_read": approx_bytes_read,
        "peak_rss_bytes": peak_rss_bytes,
        "est_seconds": round(est_seconds, 1),
    }


def human_bytes(n: float) -> str:
    """Compact human-readable byte size (order-of-magnitude display)."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"
