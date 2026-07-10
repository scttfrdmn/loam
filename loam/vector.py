"""Vector enrichment — the row-oriented (VEJ) half of SageMaker Geospatial.

loam's raster ops read COGs; this reads tables of points. It provides the VEJ operations:
  * ``reverse-geocode`` — points → place attributes (offline GeoNames, or online Nominatim).
  * ``map-match`` — GPS traces (ordered points grouped by a trace id) → matched road geometry +
    edge/way ids, via a routing service (Valhalla by default, or OSRM).

All are non-raster but ride the same manifest/shard machinery: ``plan`` chunks the input into
shards, ``run-shard`` processes one chunk (see ``loam.plan`` / ``loam.run``).

This module is pure content — parse, look up / match, write results. It knows nothing about S3,
shards, or runners (mirrors ``loam.ops``), and imports no raster stack, so it stays cheap and
independently testable. Online backends (Nominatim, Valhalla, OSRM) use stdlib ``urllib`` and call
a *data* HTTP service — never a compute provisioner, so the execution-agnostic contract holds.

Backends are pluggable: ``_BACKENDS`` for reverse-geocode, ``_MATCH_BACKENDS`` for map-match. The
default reverse-geocode backend is OFFLINE (deterministic, network-free) via the optional
``pip install loam-geo[vector]`` extra.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from typing import Any, Callable

# Enrichment fields appended to each row (CSV columns / GeoJSON feature properties). The offline
# backend fills name/admin1/admin2/cc (city/admin level); the online Nominatim backend additionally
# fills geo_address (a full street-level display string). Both share this schema so output columns
# are stable regardless of backend.
GEO_FIELDS = ["geo_name", "geo_admin1", "geo_admin2", "geo_cc", "geo_address"]

# Public Nominatim endpoint. Overridable for a self-hosted instance (higher rate limits).
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

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
            "geo_address": "",  # offline backend is city/admin level, no street address
        }
        for r in results
    ]


# Nominatim is rate-limited to 1 req/s by usage policy; sleep this long between requests.
_NOMINATIM_MIN_INTERVAL_S = 1.0


def _nominatim_backend(points: list[tuple[float, float]]) -> list[dict[str, str]]:
    """Reverse-geocode via the online Nominatim (OpenStreetMap) API — street-level addresses.

    Online, rate-limited (≤1 req/s), non-deterministic — opt-in, not the default. Respects the
    Nominatim usage policy (a real User-Agent, ≤1 req/s). For volume, self-host and override
    ``NOMINATIM_URL``. Endpoint host is read from ``NOMINATIM_URL`` so a self-hosted instance drops
    in without code changes.
    """
    import time
    import urllib.parse
    import urllib.request

    from . import __version__

    out: list[dict[str, str]] = []
    for i, (lat, lon) in enumerate(points):
        if i:  # polite spacing between requests (skip before the first)
            time.sleep(_NOMINATIM_MIN_INTERVAL_S)
        qs = urllib.parse.urlencode({"lat": lat, "lon": lon, "format": "jsonv2"})
        req = urllib.request.Request(
            f"{NOMINATIM_URL}?{qs}",
            headers={"User-Agent": f"loam-geo/{__version__} (https://github.com/scttfrdmn/loam)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https endpoint
            data = json.loads(resp.read().decode("utf-8"))
        addr = data.get("address", {})
        out.append({
            "geo_name": (addr.get("city") or addr.get("town") or addr.get("village")
                         or addr.get("hamlet") or addr.get("county") or ""),
            "geo_admin1": addr.get("state", ""),
            "geo_admin2": addr.get("county", ""),
            "geo_cc": (addr.get("country_code") or "").upper(),
            "geo_address": data.get("display_name", ""),
        })
    return out


# Pluggable backends. ``offline`` is the default (deterministic, network-free); ``nominatim`` is
# opt-in for street-level detail. Add more (a self-hosted service, etc.) here without touching callers.
_BACKENDS: dict[str, Callable[[list[tuple[float, float]]], list[dict[str, str]]]] = {
    "offline": _offline_backend,
    "nominatim": _nominatim_backend,
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


# ── map-match: snap ordered GPS traces to roads (the second VEJ half) ─────────

# Routing service endpoints — overridable module constants (point at a self-hosted instance).
VALHALLA_URL = "https://valhalla1.openstreetmap.de"      # community demo; self-host for volume
OSRM_URL = "https://router.project-osrm.org"             # public demo; ~100-coord/req cap

# Rate-limit spacing between routing requests (polite to shared/demo servers).
_MATCH_MIN_INTERVAL_S = 1.0

# Stable output-property schema (like GEO_FIELDS). A backend that can't fill one emits None.
MATCH_FIELDS = ["match_trace_id", "match_confidence", "match_way_ids", "match_names",
                "match_point_count"]

_TRACE_ALIASES = ("trace_id", "trace", "track_id", "id")


def read_trace_points(
    text: str, fmt: str, *, trace_field: str | None = "trace_id",
    lat_field: str | None = None, lon_field: str | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Parse points and GROUP them into ordered traces by ``trace_field``.

    Returns an insertion-ordered ``{trace_id: [(lat, lon), ...]}`` — deterministic, and within a
    trace the points keep input order (the GPS sequence). If ``trace_field`` is None, the whole file
    is one trace (id ``"trace"``). If ``trace_field`` is set but absent from the data, raises (a
    silent one-giant-trace fallback would corrupt results by matching unrelated logs as one path).
    """
    rows, coords = read_points(text, fmt, lat_field=lat_field, lon_field=lon_field)
    traces: dict[str, list[tuple[float, float]]] = {}
    if trace_field is None:
        traces["trace"] = coords
        return traces
    for row, coord in zip(rows, coords):
        props = row.get("properties", row) if fmt == "geojson" else row
        if trace_field not in props:
            raise ValueError(f"trace field {trace_field!r} not found in the input")
        traces.setdefault(str(props[trace_field]), []).append(coord)
    return traces


def _decode_polyline6(encoded: str) -> list[list[float]]:
    """Decode a Google-encoded polyline at precision 1e-6 → list of [lon, lat] (GeoJSON order).

    Valhalla's ``trace_attributes`` returns geometry as a precision-6 encoded polyline. Standard
    algorithm, pure stdlib. Precision 6 is hard-coded (Valhalla's default; documented assumption).
    """
    coords: list[list[float]] = []
    index = lat = lon = 0
    while index < len(encoded):
        for is_lon in (False, True):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lon:
                lon += delta
            else:
                lat += delta
        coords.append([lon / 1e6, lat / 1e6])
    return coords


def _valhalla_backend(coords: list[tuple[float, float]]) -> dict[str, Any]:
    """Map-match one trace via Valhalla ``/trace_attributes`` — geometry + edge/way ids."""
    import urllib.request

    from . import __version__

    body = json.dumps({
        "shape": [{"lat": lat, "lon": lon} for lat, lon in coords],
        "costing": "auto",
        "shape_match": "map_snap",
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{VALHALLA_URL}/trace_attributes", data=body,
        headers={"User-Agent": f"loam-geo/{__version__} (https://github.com/scttfrdmn/loam)",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https endpoint
        data = json.loads(resp.read().decode("utf-8"))
    edges = data.get("edges", [])
    way_ids = [e["way_id"] for e in edges if e.get("way_id") is not None]
    names = sorted({n for e in edges for n in e.get("names", [])})
    line = _decode_polyline6(data["shape"]) if data.get("shape") else []
    return {
        "geometry": {"type": "LineString", "coordinates": line},
        "match_confidence": None,          # Valhalla trace_attributes has no single confidence
        "match_way_ids": way_ids,
        "match_names": names,
    }


def _osrm_backend(coords: list[tuple[float, float]]) -> dict[str, Any]:
    """Map-match one trace via OSRM ``/match`` — GeoJSON geometry + confidence."""
    import urllib.request

    from . import __version__

    path = ";".join(f"{lon},{lat}" for lat, lon in coords)  # OSRM wants lon,lat
    url = f"{OSRM_URL}/match/v1/driving/{path}?geometries=geojson&overview=full"
    req = urllib.request.Request(
        url, headers={"User-Agent": f"loam-geo/{__version__} (https://github.com/scttfrdmn/loam)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https endpoint
        data = json.loads(resp.read().decode("utf-8"))
    matchings = data.get("matchings") or []
    if not matchings:
        raise ValueError("OSRM returned no matching for the trace")
    m = matchings[0]
    return {
        "geometry": m.get("geometry") or {"type": "LineString", "coordinates": []},
        "match_confidence": m.get("confidence"),
        "match_way_ids": [],               # OSRM /match doesn't return OSM way ids by default
        "match_names": [],
    }


# Pluggable map-match backends — separate registry from _BACKENDS (different signature: one
# trace's ordered coords → one match dict with geometry). Valhalla is the default (tiled/arm64 fit,
# edge/way-id output); OSRM is the alternative (faster pure-snap, existing-instance users).
_MATCH_BACKENDS: dict[str, Callable[[list[tuple[float, float]]], dict[str, Any]]] = {
    "valhalla": _valhalla_backend,
    "osrm": _osrm_backend,
}


def map_match(coords: list[tuple[float, float]], *, backend: str = "valhalla") -> dict[str, Any]:
    """Match one ordered trace to roads. Returns {geometry, match_confidence, match_way_ids, ...}."""
    if backend not in _MATCH_BACKENDS:
        raise ValueError(f"unknown map-match backend {backend!r}; known: {', '.join(_MATCH_BACKENDS)}")
    return _MATCH_BACKENDS[backend](coords)


def write_matched(matches: list[tuple[str, dict[str, Any]]]) -> str:
    """Serialize matched traces as a GeoJSON FeatureCollection — one LineString Feature per trace.

    ``matches`` is a list of (trace_id, match_dict). The match dict's ``geometry`` becomes the
    feature geometry; its MATCH_FIELDS go to properties (missing → None, JSON-safe).
    """
    feats = []
    for trace_id, m in matches:
        props = {
            "match_trace_id": trace_id,
            "match_confidence": m.get("match_confidence"),
            "match_way_ids": m.get("match_way_ids"),
            "match_names": m.get("match_names"),
            "match_point_count": len((m.get("geometry") or {}).get("coordinates", [])),
        }
        feats.append({"type": "Feature", "properties": props,
                      "geometry": m.get("geometry")})
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=2)
