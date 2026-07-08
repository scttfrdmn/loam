"""Live integration tests — the one seam the offline unit suite deliberately can't cover.

`tests/test_core.py` fakes the network (monkeypatches `ops._read_band_raster`) so it runs with
zero I/O. These tests exercise the REAL path instead: `catalog.search` against the public
Earth Search STAC API, its asset-key → canonical-band mapping and pagination, and — optionally
— a full `run_shard` reading real Sentinel-2 COGs via `/vsicurl`.

They hit the network (Earth Search + the public `sentinel-cogs` bucket) but need NO AWS
credentials. They are **opt-in**: skipped unless ``LOAM_LIVE_TESTS=1`` so default `pytest` (and
CI) stays fully offline and hermetic. Run them with:

    LOAM_LIVE_TESTS=1 pytest -m integration

The AOI/date window below is a small tile off the Mauritania coast (the H2 "fairy-circle"
prospecting area of fieldwork Tutorial 01), chosen to reliably return a couple of low-cloud
Sentinel-2 scenes.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from loam import catalog

pytestmark = pytest.mark.integration

# Skip the whole module unless explicitly opted in — keeps `pytest` offline by default.
if os.environ.get("LOAM_LIVE_TESTS") != "1":
    pytest.skip("live network tests; set LOAM_LIVE_TESTS=1 to run", allow_module_level=True)


# A small AOI + short window that reliably returns low-cloud Sentinel-2 scenes.
_AOI = [-5.0, 20.0, -4.9, 20.1]
_START, _END = "2023-06-01", "2023-06-30"


def test_search_returns_scenes_with_canonical_bands():
    scenes = catalog.search(
        collection="sentinel-2", aoi=_AOI, start=_START, end=_END, max_cloud=20, limit=2
    )
    assert len(scenes) >= 1
    for s in scenes:
        # NDVI's bands plus SCL must be present under canonical names.
        assert {"red", "nir", "scl"} <= s.assets.keys()
        # datetime is populated (RFC3339 from the STAC item), ids look like Sentinel-2 granules.
        assert s.datetime
        assert s.id.startswith("S2")
        # hrefs are /vsicurl-able COGs (public HTTPS), not s3:// requiring creds.
        assert s.assets["red"].startswith("http")
    # search sorts by (datetime, id) — deterministic ordering for reproducible shards.
    assert scenes == sorted(scenes, key=lambda s: (s.datetime, s.id))


def test_asset_keys_are_lowercase_canonical_not_bXX():
    """Resolve the #3 asset-alias question: Earth Search v1 exposes lowercase canonical keys.

    Confirms `_S2_ASSET_ALIASES` (the B04→red map) is a no-op on Earth Search — it's retained
    only for other catalogs (e.g. Planetary Computer uses B04). If this ever fails, Earth Search
    changed its asset naming and the alias map / _canonical_band needs revisiting.
    """
    scenes = catalog.search(
        collection="sentinel-2", aoi=_AOI, start=_START, end=_END, max_cloud=20, limit=1
    )
    keys = set(scenes[0].assets)
    assert "red" in keys and "nir" in keys and "scl" in keys
    # No raw "B04"-style keys survive to a Scene (they'd only appear if Earth Search emitted them
    # and _canonical_band failed to map them).
    assert not any(k.upper().startswith("B") and k[1:].isdigit() for k in keys)


def test_wanted_bands_filters_to_requested_only():
    """wanted_bands must keep only the requested canonical bands (drops Earth Search's -jp2 dups)."""
    scenes = catalog.search(
        collection="sentinel-2", aoi=_AOI, start=_START, end=_END, max_cloud=20, limit=1,
        wanted_bands={"red", "nir", "scl"},
    )
    assert set(scenes[0].assets) == {"red", "nir", "scl"}


def test_end_to_end_run_shard_over_real_cogs(tmp_path):
    """Full pull-model loop over REAL COGs: plan → run_shard → georeferenced GeoTIFF on disk.

    Reads real Sentinel-2 bands via /vsicurl at a coarse overview (target_res) so it stays fast
    and light. Writes to a local tmp_path 'object store' (state.py treats non-s3:// as files),
    so no AWS. Proves the network read + compute + write path end-to-end.
    """
    import rasterio

    from loam import plan, run, state

    out = str(tmp_path / "out")
    manifest_uri = str(tmp_path / "manifest.json")

    m = plan.build_manifest(
        op="band-math", collection="sentinel-2", aoi=_AOI, start=_START, end=_END,
        indices=["NDVI"], max_cloud=20, shard_size=2, limit=2,
        target_res=200.0,  # coarse overview → tiny read, fast test
        output_uri=out,
    )
    assert len(m.scenes) >= 1
    plan.write_manifest(m, manifest_uri)

    summary = run.run_shard(manifest_uri, 0)
    assert summary["status"] == "done"
    assert summary["outputs"] >= 1
    assert not summary["failed"], summary["failed"]
    assert state.shard_done(out, 0)

    # The written GeoTIFF is georeferenced and holds plausible NDVI values (in [-1, 1], NaN where
    # cloud-masked). Open the first scene's NDVI output.
    tif = state.output_uri_for(out, 0, f"{m.scenes[0].id}__NDVI.tif")
    with rasterio.open(tif) as src:
        assert src.crs is not None
        data = src.read(1)
    finite = data[np.isfinite(data)]
    assert finite.size > 0
    assert finite.min() >= -1.0001 and finite.max() <= 1.0001

    # Second run is a no-op (checkpoint exists) — the spot-safe resume property, over the wire.
    assert run.run_shard(manifest_uri, 0)["status"] == "skipped"


def test_end_to_end_resample_over_real_cogs(tmp_path):
    """Resample a real Sentinel-2 band to WGS84 via /vsicurl → reprojected COG on disk."""
    import rasterio

    from loam import plan, run, state

    out = str(tmp_path / "out")
    manifest_uri = str(tmp_path / "manifest.json")

    m = plan.build_manifest(
        op="resample", collection="sentinel-2", aoi=_AOI, start=_START, end=_END,
        bands=["red"], dst_crs="EPSG:4326", resampling="bilinear",
        max_cloud=20, shard_size=2, limit=1,
        target_res=200.0,  # coarse overview → tiny read
        output_uri=out,
    )
    assert len(m.scenes) >= 1
    plan.write_manifest(m, manifest_uri)

    summary = run.run_shard(manifest_uri, 0)
    assert summary["status"] == "done"
    assert not summary["failed"], summary["failed"]

    tif = state.output_uri_for(out, 0, f"{m.scenes[0].id}__red.tif")
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:4326"  # reprojected off UTM
