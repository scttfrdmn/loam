"""Catalog search — the replacement for SageMaker's RasterDataCollectionQuery.

SageMaker Geospatial's "Raster Data Collections" were, underneath, a STAC catalog plus a
query API over public imagery that AWS did not own (Sentinel-2 is ESA's, in the free
``sentinel-2-l2a`` bucket). We query the same imagery directly via a public STAC API
(Element84 Earth Search by default), so this runs in ANY region and in a fresh account —
no onboarding, no us-west-2 lock, no 2026-07-30 cutoff.

Output is a list of ``Scene`` objects (band name -> COG href), ready to drop into a Manifest.
"""

from __future__ import annotations

from .manifest import Scene

# Default public STAC endpoint. Overridable for Planetary Computer, a private catalog, etc.
DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"

# Collection alias -> STAC collection id on Earth Search v1.
# (Mirrors the reference impl's CollectionAlias, mapped to STAC ids instead of SM names.)
COLLECTIONS: dict[str, str] = {
    "sentinel-2": "sentinel-2-l2a",
    "sentinel2": "sentinel-2-l2a",
    "sentinel-2-l2a": "sentinel-2-l2a",
    "landsat-8": "landsat-c2-l2",
    "landsat-9": "landsat-c2-l2",
    "landsat": "landsat-c2-l2",
}

# STAC asset key -> our canonical band name, for Sentinel-2 L2A on Earth Search.
# Earth Search exposes bands as asset keys like "red", "nir", "swir16", "scl" already, but
# older catalogs use "B04" etc. — normalize both so equations always see canonical names.
_S2_ASSET_ALIASES: dict[str, str] = {
    "B01": "coastal", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3",
    "B08": "nir", "B8A": "nir08", "B09": "nir09",
    "B11": "swir16", "B12": "swir22", "SCL": "scl",
}


def resolve_collection(name: str) -> str:
    key = name.strip().lower()
    if key not in COLLECTIONS:
        raise KeyError(f"unknown collection {name!r}; known: {', '.join(sorted(set(COLLECTIONS)))}")
    return COLLECTIONS[key]


def _canonical_band(asset_key: str) -> str:
    """Map a STAC asset key to a canonical band name."""
    if asset_key in _S2_ASSET_ALIASES:
        return _S2_ASSET_ALIASES[asset_key]
    return asset_key.lower()


def search(
    *,
    collection: str,
    aoi: list[float],
    start: str,
    end: str,
    max_cloud: float | None = None,
    limit: int | None = None,
    stac_url: str = DEFAULT_STAC_URL,
    wanted_bands: set[str] | None = None,
) -> list[Scene]:
    """Search a STAC catalog and return Scenes.

    Args:
        collection: alias or STAC id (e.g. "sentinel-2").
        aoi: [west, south, east, north] in WGS84.
        start, end: RFC3339 / "YYYY-MM-DD" datetimes (Earth Search wants RFC3339 range).
        max_cloud: optional eo:cloud_cover upper bound (percent).
        limit: optional cap on scenes returned (handy for tests/dev).
        wanted_bands: if given, only keep these bands' hrefs in each Scene's assets.
    """
    from pystac_client import Client  # imported lazily so `loam --help` needs no geo stack

    coll_id = resolve_collection(collection)
    west, south, east, north = aoi

    client = Client.open(stac_url)
    query = {"eo:cloud_cover": {"lte": max_cloud}} if max_cloud is not None else None
    search_result = client.search(
        collections=[coll_id],
        bbox=[west, south, east, north],
        datetime=f"{start}/{end}",
        query=query,
        max_items=limit,
    )

    scenes: list[Scene] = []
    for item in search_result.items():
        assets: dict[str, str] = {}
        for asset_key, asset in item.assets.items():
            band = _canonical_band(asset_key)
            if wanted_bands is not None and band not in wanted_bands:
                continue
            assets[band] = asset.href
        if not assets:
            continue
        scenes.append(
            Scene(
                id=item.id,
                datetime=item.datetime.isoformat() if item.datetime else "",
                assets=assets,
            )
        )
    scenes.sort(key=lambda s: (s.datetime, s.id))
    return scenes
