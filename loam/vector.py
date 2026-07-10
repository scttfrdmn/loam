"""Vector enrichment — the row-oriented (VEJ) half of SageMaker Geospatial.

loam's raster ops read COGs; this reads a table of points. ``reverse-geocode`` takes a CSV or
GeoJSON of lat/lon points and appends place attributes (name / admin1 / admin2 / country). It is
the first non-raster op, but rides the same manifest/shard machinery: ``plan`` chunks the input
rows into shards, and ``run-shard`` enriches one chunk (see ``loam.plan`` / ``loam.run``).

This module is pure content — parse points, look them up, write enriched rows. It knows nothing
about S3, shards, or runners (mirrors ``loam.ops``), and imports no raster stack, so it stays
cheap and independently testable.

Backend: an OFFLINE reverse geocoder (GeoNames KD-tree) — deterministic, network-free, spot-safe,
and CI-testable, matching loam's execution-agnostic values. It is an optional dependency
(``pip install loam-geo[vector]``). The backend is pluggable (``_BACKENDS``) so an online option
(e.g. Nominatim, street-level) can be added later without changing callers.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any, Callable

# Enrichment fields appended to each row (CSV columns / GeoJSON feature properties).
GEO_FIELDS = ["geo_name", "geo_admin1", "geo_admin2", "geo_cc"]

# Common lat/lon column-name aliases tried when the caller doesn't specify.
_LAT_ALIASES = ("lat", "latitude", "y")
_LON_ALIASES = ("lon", "lng", "long", "longitude", "x")


def _offline_backend(points: list[tuple[float, float]]) -> list[dict[str, str]]:
    """Reverse-geocode via the offline ``reverse_geocoder`` package (GeoNames KD-tree)."""
    try:
        import reverse_geocoder as rg
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reverse-geocode needs the 'vector' extra: pip install 'loam-geo[vector]'"
        ) from e

    if not points:
        return []
    # mode=1 = single-threaded (deterministic, no multiprocessing pool — safe on a small box).
    results = rg.search([(lat, lon) for lat, lon in points], mode=1)
    return [
        {
            "geo_name": r.get("name", ""),
            "geo_admin1": r.get("admin1", ""),
            "geo_admin2": r.get("admin2", ""),
            "geo_cc": r.get("cc", ""),
        }
        for r in results
    ]


# Pluggable backends. Add an online one (Nominatim) here without touching callers.
_BACKENDS: dict[str, Callable[[list[tuple[float, float]]], list[dict[str, str]]]] = {
    "offline": _offline_backend,
}


def reverse_geocode(
    points: list[tuple[float, float]], *, backend: str = "offline"
) -> list[dict[str, str]]:
    """Return an enrichment dict (``GEO_FIELDS``) per (lat, lon) point, in order."""
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {', '.join(_BACKENDS)}")
    return _BACKENDS[backend](points)


# ── I/O: CSV and GeoJSON points in, enriched rows out (stdlib only) ───────────

def _pick_field(header: Sequence[str], preferred: str | None, aliases: tuple[str, ...]) -> str:
    """Resolve a lat/lon column: the caller's choice, else the first known alias present."""
    if preferred:
        if preferred not in header:
            raise ValueError(f"column {preferred!r} not in CSV header {header}")
        return preferred
    lower = {h.lower(): h for h in header}
    for a in aliases:
        if a in lower:
            return lower[a]
    raise ValueError(f"no lat/lon column found in {header}; pass --lat-field/--lon-field")


def read_points(
    text: str, fmt: str, *, lat_field: str | None = None, lon_field: str | None = None
) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    """Parse points from CSV or GeoJSON. Returns (rows, coords) with coords aligned to rows.

    CSV: each row is a dict; lat/lon read from the resolved columns. GeoJSON: a FeatureCollection
    of Point features; each row is the feature, coords from its geometry (GeoJSON is [lon, lat]).
    """
    rows: list[dict[str, Any]] = []
    coords: list[tuple[float, float]] = []
    if fmt == "csv":
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return [], []
        header = reader.fieldnames or list(rows[0].keys())
        latf = _pick_field(header, lat_field, _LAT_ALIASES)
        lonf = _pick_field(header, lon_field, _LON_ALIASES)
        coords = [(float(r[latf]), float(r[lonf])) for r in rows]
        return rows, coords
    if fmt == "geojson":
        doc = json.loads(text)
        for f in doc.get("features", []):
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                raise ValueError("reverse-geocode GeoJSON must contain only Point features")
            lon, lat = geom["coordinates"][0], geom["coordinates"][1]
            rows.append(f)
            coords.append((float(lat), float(lon)))
        return rows, coords
    raise ValueError(f"unsupported vector format {fmt!r} (use csv or geojson)")


def read_polygons(text: str) -> list[dict[str, Any]]:
    """Parse a GeoJSON FeatureCollection of Polygon/MultiPolygon zones → list of features.

    Each feature keeps its ``geometry`` (used by ``zonal.zonal_stats``) and ``properties`` (carried
    through to the output). Rejects non-polygon geometries — zonal statistics needs areas, not
    points/lines.
    """
    doc = json.loads(text)
    feats = doc.get("features", [])
    for f in feats:
        gtype = (f.get("geometry") or {}).get("type")
        if gtype not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"zonal-stats zones must be Polygon/MultiPolygon features, got {gtype!r}")
    return feats


def write_chunk(rows: list[dict[str, Any]], fmt: str) -> str:
    """Serialize rows as-is (no enrichment) — used to persist input chunks for run-shard."""
    return write_enriched(rows, [{} for _ in rows], fmt)


def write_enriched(rows: list[dict[str, Any]], enrich: list[dict[str, str]], fmt: str) -> str:
    """Serialize rows with their enrichment merged in, in the same format they came from."""
    if len(rows) != len(enrich):
        raise ValueError(f"rows ({len(rows)}) and enrichment ({len(enrich)}) length mismatch")
    if fmt == "csv":
        if not rows:
            return ""
        fieldnames = list(rows[0].keys()) + [f for f in GEO_FIELDS if f not in rows[0]]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row, e in zip(rows, enrich):
            writer.writerow({**row, **e})
        return buf.getvalue()
    if fmt == "geojson":
        out_feats = []
        for feat, e in zip(rows, enrich):
            merged = dict(feat)
            merged["properties"] = {**(feat.get("properties") or {}), **e}
            out_feats.append(merged)
        return json.dumps({"type": "FeatureCollection", "features": out_feats}, indent=2)
    raise ValueError(f"unsupported vector format {fmt!r} (use csv or geojson)")
