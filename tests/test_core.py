"""Core tests — the parts that must hold without any AWS or network.

These exercise the execution-agnostic contract directly: manifests round-trip, sharding is
deterministic, band-math evaluates correctly over synthetic rasters, and run-shard is
idempotent against a local-path "object store" (state.py treats non-s3:// URIs as files).
"""

from __future__ import annotations

import numpy as np
import pytest

from loam import indices
from loam.manifest import Manifest, Scene, shard_scenes, MANIFEST_VERSION


def test_index_catalog_ported():
    ndvi = indices.resolve("ndvi")
    assert ndvi.equation == "(nir - red) / (nir + red)"
    assert indices.resolve("BSI").equation == "(swir16 - nir) / (swir16 + nir)"
    assert len(indices.INDICES) == 7


def test_custom_index_spec():
    d = indices.parse_spec("NDWI=(green - nir) / (green + nir)")
    assert d.name == "NDWI"
    assert d.description == "custom"


def test_unknown_index_helpful_error():
    with pytest.raises(KeyError, match="unknown index"):
        indices.resolve("NOPE")


def test_bands_in_equation():
    assert indices.bands_in("(nir - red) / (nir + red)") == {"nir", "red"}
    assert "swir16" in indices.bands_in("(swir16 - nir) / (swir16 + nir)")


def test_sharding_deterministic():
    scenes = [Scene(id=f"s{i}", datetime="2023-01-01", assets={}) for i in range(125)]
    shards = shard_scenes(scenes, 50)
    assert len(shards) == 3
    assert [len(s.scene_ids) for s in shards] == [50, 50, 25]
    # deterministic: same input -> same shard membership
    assert shard_scenes(scenes, 50)[1].scene_ids == shards[1].scene_ids


def test_manifest_roundtrip():
    m = Manifest(
        version=MANIFEST_VERSION,
        op="band-math",
        params={"indices": ["NDVI"]},
        collection="sentinel-2-l2a",
        aoi=[-7.0, 19.0, -3.0, 22.0],
        output_uri="s3://b/out/",
        scenes=[Scene(id="s0", datetime="2023-01-01", assets={"red": "r", "nir": "n"})],
    )
    m.shards = shard_scenes(m.scenes, 50)
    m2 = Manifest.from_json(m.to_json())
    assert m2.op == "band-math"
    assert m2.scenes[0].assets["nir"] == "n"
    assert m2.scenes_for(0)[0].id == "s0"


# A synthetic georeferenced band, used to fake raster.read_band in ops. UTM-ish transform.
def _fake_raster(value, h=4, w=4):
    from loam.raster import Raster

    return Raster(
        data=np.full((h, w), float(value), np.float32),
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 2200000.0),
        crs="EPSG:32629",
        nodata=None,
    )


def test_band_math_math(monkeypatch):
    # Feed known band arrays; NDVI of nir=3, red=1 -> (3-1)/(3+1) = 0.5.
    from loam import ops

    fake = {"nir": _fake_raster(3.0), "red": _fake_raster(1.0)}
    monkeypatch.setattr(ops, "_read_band_raster", lambda href, target_res=None: fake[href])
    out = ops.band_math({"nir": "nir", "red": "red"}, indices.resolve("NDVI"), scl_mask=False)
    assert np.allclose(out.data, 0.5)
    # georeferencing carried through
    assert out.crs == "EPSG:32629"
    assert out.transform[0] == 10.0 and out.transform[2] == 500000.0


def test_run_shard_idempotent_and_geotiff(tmp_path, monkeypatch):
    from loam import ops, run, state

    # local "object store": output_uri is a filesystem path
    out = str(tmp_path / "out")
    scene = Scene(id="scene0", datetime="2023-01-01", assets={"nir": "nir", "red": "red"})
    m = Manifest(
        version=MANIFEST_VERSION, op="band-math", params={"indices": ["NDVI"], "format": "cog"},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=[scene],
    )
    m.shards = shard_scenes(m.scenes, 50)
    manifest_uri = str(tmp_path / "manifest.json")
    state.put_text(manifest_uri, m.to_json())

    monkeypatch.setattr(
        ops, "_read_band_raster",
        lambda href, target_res=None: _fake_raster(3.0 if href == "nir" else 1.0, 512, 512),
    )

    s1 = run.run_shard(manifest_uri, 0)
    assert s1["status"] == "done"
    assert s1["outputs"] == 1
    assert s1["format"] == "cog"
    assert state.shard_done(out, 0)

    # the written output is a real georeferenced GeoTIFF that round-trips through rasterio
    import rasterio

    tif = state.output_uri_for(out, 0, "scene0__NDVI.tif")
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:32629"
        assert src.transform.a == 10.0
        assert abs(float(src.read(1).mean()) - 0.5) < 1e-4

    # second run is a no-op (checkpoint exists) — the spot-safe resume property
    s2 = run.run_shard(manifest_uri, 0)
    assert s2["status"] == "skipped"


def test_run_shard_npy_format(tmp_path, monkeypatch):
    from loam import ops, run, state

    out = str(tmp_path / "out")
    scene = Scene(id="s0", datetime="2023-01-01", assets={"nir": "nir", "red": "red"})
    m = Manifest(
        version=MANIFEST_VERSION, op="band-math", params={"indices": ["NDVI"], "format": "npy"},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=[scene],
    )
    m.shards = shard_scenes(m.scenes, 50)
    manifest_uri = str(tmp_path / "m.json")
    state.put_text(manifest_uri, m.to_json())
    monkeypatch.setattr(
        ops, "_read_band_raster",
        lambda href, target_res=None: _fake_raster(3.0 if href == "nir" else 1.0),
    )
    s = run.run_shard(manifest_uri, 0)
    assert s["format"] == "npy"
    assert state.exists(state.output_uri_for(out, 0, "s0__NDVI.npy"))


def test_status_from_store(tmp_path):
    from loam import run, state, plan

    out = str(tmp_path / "out")
    scenes = [Scene(id=f"s{i}", datetime="2023-01-01", assets={}) for i in range(3)]
    m = Manifest(
        version=MANIFEST_VERSION, op="cloud-mask", params={}, collection="sentinel-2-l2a",
        aoi=[0, 0, 1, 1], output_uri=out, scenes=scenes,
    )
    m.shards = shard_scenes(scenes, 1)  # 3 shards
    manifest_uri = str(tmp_path / "m.json")
    state.put_text(manifest_uri, m.to_json())

    assert plan.status(manifest_uri)["shards_done"] == 0
    # fake-complete one shard by writing its checkpoint
    state.put_text(state.checkpoint_uri(out, 1), "{}")
    st = plan.status(manifest_uri)
    assert st["shards_done"] == 1
    assert st["shards_remaining"] == 2
    assert st["complete"] is False
