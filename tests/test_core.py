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
    # 7 original + 6 curated (NDWI, SAVI, GNDVI, NDMI, NDRE, ARVI)
    assert len(indices.INDICES) == 13
    assert indices.resolve("ndwi").equation == "(green - nir) / (green + nir)"
    assert indices.resolve("NDRE").equation == "(nir - rededge1) / (nir + rededge1)"


def test_every_catalog_equation_validates_and_computes():
    # Every built-in index must pass the safe-eval allowlist and evaluate over its bands.
    # Feed a distinct positive scalar per band so denominators can't be zero.
    env = {b: float(i + 2) for i, b in enumerate(sorted(indices._BAND_TOKENS))}
    for name, d in indices.INDICES.items():
        indices.validate_equation(d.equation)  # never raises for a catalog entry
        needed = indices.bands_in(d.equation)
        result = indices.safe_eval(d.equation, {b: env[b] for b in needed})
        assert isinstance(result, float), name
        # every band the equation references must be a known token (so ops can fetch it)
        assert needed <= indices._BAND_TOKENS, name


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


def test_band_math_mixed_resolution(monkeypatch):
    # BSI mixes swir16 (20 m → half grid) with nir (10 m → full grid). loam must resample the
    # coarse band onto the fine grid, not slice — regression test for the Tut01 smoke bug.
    from loam import ops
    from loam.raster import Raster

    def fake(href, target_res=None):
        if href == "swir16":
            data = np.full((2, 2), 2.0, np.float32)      # coarse 20 m band
            return Raster(data, (20.0, 0, 0, 0, -20.0, 0), "EPSG:32629", None)
        # nir on the fine 10 m grid
        return Raster(np.full((4, 4), 6.0, np.float32), (10.0, 0, 0, 0, -10.0, 0), "EPSG:32629", None)

    monkeypatch.setattr(ops, "_read_band_raster", fake)
    out = ops.band_math({"swir16": "swir16", "nir": "nir"}, indices.resolve("BSI"), scl_mask=False)
    # BSI = (swir16 - nir)/(swir16 + nir) = (2-6)/(2+6) = -0.5, on the FINE 4x4 grid
    assert out.data.shape == (4, 4)
    assert np.allclose(out.data, -0.5)
    assert out.transform[0] == 10.0  # ref (fine) transform carried


def test_reproject_raster_changes_crs():
    # A UTM raster reprojected to WGS84 must come back tagged EPSG:4326 with a lon/lat transform.
    from loam.raster import reproject_raster

    src = _fake_raster(0.5, 32, 32)  # EPSG:32629, 10 m
    out = reproject_raster(src, dst_crs="EPSG:4326", resampling="nearest")
    assert out.crs == "EPSG:4326"
    assert out.data.ndim == 2 and out.data.size > 0
    # WGS84 pixel size is in degrees (far below 1), unlike the 10 m source.
    assert abs(out.transform[0]) < 1.0
    # value preserved (constant field, nearest resampling)
    finite = out.data[np.isfinite(out.data)]
    assert finite.size > 0 and np.allclose(finite, 0.5)


def test_resample_op_per_band(monkeypatch):
    from loam import ops

    monkeypatch.setattr(ops, "_read_band_raster",
                        lambda href, target_res=None: _fake_raster(1.0, 16, 16))
    out = ops.resample(
        {"red": "red", "nir": "nir"}, ["red", "nir"],
        dst_crs="EPSG:4326", resampling="nearest",
    )
    assert set(out) == {"red", "nir"}
    for r in out.values():
        assert r.crs == "EPSG:4326"


def test_resample_missing_band_raises(monkeypatch):
    from loam import ops
    with pytest.raises(KeyError, match="missing bands"):
        ops.resample({"red": "red"}, ["red", "nir"], dst_crs="EPSG:4326")


def test_resample_bad_method_raises():
    from loam.raster import reproject_raster
    with pytest.raises(ValueError, match="unknown resampling"):
        reproject_raster(_fake_raster(1.0), dst_crs="EPSG:4326", resampling="nope")


def test_safe_eval_all_catalog_equations():
    # Every catalog equation must compute identically under the AST evaluator. Feed synthetic
    # per-band scalars and compare to the plain Python arithmetic (the acceptance guard).
    env = {b: float(i + 2) for i, b in enumerate(
        ["coastal", "blue", "green", "red", "nir", "swir16", "swir22"]
    )}
    blue, green, red, nir, swir16, swir22 = (
        env["blue"], env["green"], env["red"], env["nir"], env["swir16"], env["swir22"]
    )
    expected = {
        "NDVI": (nir - red) / (nir + red),
        "BSI": (swir16 - nir) / (swir16 + nir),
        "EVI": 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0),
        "MNDWI": (green - swir16) / (green + swir16),
        "NDBI": (swir16 - nir) / (swir16 + nir),
        "NBR": (nir - swir22) / (nir + swir22),
        "NDSI": (green - swir16) / (green + swir16),
    }
    # Byte-identical to plain Python for these representative equations. (Full-catalog coverage —
    # every index validates + computes — is test_every_catalog_equation_validates_and_computes.)
    for name, want in expected.items():
        assert indices.safe_eval(indices.INDICES[name].equation, env) == pytest.approx(want), name


def test_safe_eval_power_and_unary():
    env = {"nir": 3.0, "red": 2.0}
    assert indices.safe_eval("nir ** 2", env) == pytest.approx(9.0)
    assert indices.safe_eval("-nir + red", env) == pytest.approx(-1.0)


@pytest.mark.parametrize("equation", [
    '__import__("os").system("echo pwned")',  # Call
    "().__class__.__bases__",                  # Attribute
    "nir.__class__",                            # Attribute on a band
    "nir[0]",                                   # Subscript
    "nir if red else blue",                    # IfExp
    "[nir]",                                     # List
    "(nir := red)",                             # walrus
    "nir % red",                                # Mod
    "nir // red",                               # FloorDiv
    "nir @ red",                                # MatMult
    "nir | red",                                # BitOr
    "nir ^ red",                                # BitXor
    "~nir",                                      # Invert
    "nir and red",                              # BoolOp
    "nir > red",                                # Compare
    "True",                                      # bool constant
    "None",                                      # None constant
    '"nir"',                                    # str constant
    "1j",                                        # complex constant
])
def test_safe_eval_rejects_hostile(equation):
    # A hostile / out-of-grammar equation must raise ValueError and NEVER execute.
    with pytest.raises(ValueError):
        indices.safe_eval(equation, {"nir": 1.0, "red": 1.0, "blue": 1.0})


def test_safe_eval_unknown_band():
    with pytest.raises(ValueError, match="unknown band"):
        indices.safe_eval("foo + nir", {"nir": 1.0})


@pytest.mark.parametrize("equation", ["nir +", ""])
def test_safe_eval_malformed_syntax(equation):
    # SyntaxError from ast.parse is surfaced as ValueError (the caller's contract).
    with pytest.raises(ValueError, match="invalid equation"):
        indices.safe_eval(equation, {"nir": 1.0})


def test_parse_spec_rejects_hostile_equation_at_plan_time():
    # A malicious custom index fails at parse_spec (plan time) — no numpy, no env, no execution.
    with pytest.raises(ValueError):
        indices.parse_spec('BAD=__import__("os").system("echo pwned")')
    # a legitimate custom equation still parses
    d = indices.parse_spec("NDWI=(green - nir) / (green + nir)")
    assert d.name == "NDWI"


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


def test_run_shard_resample_end_to_end(tmp_path, monkeypatch):
    # The resample op through the full manifest→run path: reprojected COGs, one per band.
    import rasterio

    from loam import ops, run, state

    out = str(tmp_path / "out")
    scene = Scene(id="scene0", datetime="2023-01-01", assets={"red": "red", "nir": "nir"})
    m = Manifest(
        version=MANIFEST_VERSION, op="resample",
        params={"format": "cog", "bands": ["red", "nir"], "dst_crs": "EPSG:4326",
                "dst_res": None, "resampling": "nearest"},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=[scene],
    )
    m.shards = shard_scenes(m.scenes, 50)
    manifest_uri = str(tmp_path / "manifest.json")
    state.put_text(manifest_uri, m.to_json())
    monkeypatch.setattr(ops, "_read_band_raster",
                        lambda href, target_res=None: _fake_raster(1.0, 64, 64))

    s = run.run_shard(manifest_uri, 0)
    assert s["status"] == "done"
    assert s["outputs"] == 2  # one COG per band
    for band in ("red", "nir"):
        tif = state.output_uri_for(out, 0, f"scene0__{band}.tif")
        with rasterio.open(tif) as src:
            assert src.crs.to_string() == "EPSG:4326"


def test_target_res_threads_to_ops(tmp_path, monkeypatch):
    # The manifest's target_res must reach ops.band_math (None = full res for small features).
    from loam import ops, run, state

    seen = {}

    def spy(assets, index, *, target_res=100.0, scl_mask=True):
        seen["target_res"] = target_res
        return _fake_raster(0.5)

    out = str(tmp_path / "out")
    scene = Scene(id="s0", datetime="2023-01-01", assets={"nir": "n", "red": "r"})
    m = Manifest(
        version=MANIFEST_VERSION, op="band-math",
        params={"indices": ["NDVI"], "format": "npy", "target_res": None},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=[scene],
    )
    m.shards = shard_scenes(m.scenes, 50)
    mu = str(tmp_path / "m.json")
    state.put_text(mu, m.to_json())
    monkeypatch.setattr(ops, "band_math", spy)
    run.run_shard(mu, 0)
    assert seen["target_res"] is None  # full-res request threaded through


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
    from loam import state, plan

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
    # default status has no ledger
    assert "ledger" not in st


def test_status_detail_ledger(tmp_path):
    import json

    from loam import state, plan

    out = str(tmp_path / "out")
    scenes = [Scene(id=f"s{i}", datetime="2023-01-01", assets={}) for i in range(3)]
    m = Manifest(
        version=MANIFEST_VERSION, op="band-math", params={"indices": ["NDVI"]},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=scenes,
    )
    m.shards = shard_scenes(scenes, 1)  # 3 shards
    manifest_uri = str(tmp_path / "m.json")
    state.put_text(manifest_uri, m.to_json())

    # two shards done, with realistic run_shard summaries; one has a failed scene
    state.put_text(state.checkpoint_uri(out, 0), json.dumps(
        {"shard": 0, "status": "done", "outputs": 2, "bytes_written": 1000, "seconds": 1.5,
         "failed": []}))
    state.put_text(state.checkpoint_uri(out, 1), json.dumps(
        {"shard": 1, "status": "done", "outputs": 1, "bytes_written": 500, "seconds": 2.0,
         "failed": [{"scene": "s1", "error": "boom"}]}))

    st = plan.status(manifest_uri, detail=True)
    assert st["shards_done"] == 2
    led = st["ledger"]
    assert led["outputs"] == 3
    assert led["bytes_written"] == 1500
    assert led["seconds"] == 3.5
    assert led["failed_scenes"] == 1
    # per-shard rows present and sorted; shard 2 (no checkpoint) absent
    assert [r["shard"] for r in led["shards"]] == [0, 1]


def test_status_detail_survives_malformed_checkpoint(tmp_path):
    # A partial/garbage checkpoint must not sink the whole ledger view.
    from loam import state, plan

    out = str(tmp_path / "out")
    scenes = [Scene(id="s0", datetime="2023-01-01", assets={})]
    m = Manifest(
        version=MANIFEST_VERSION, op="cloud-mask", params={}, collection="sentinel-2-l2a",
        aoi=[0, 0, 1, 1], output_uri=out, scenes=scenes,
    )
    m.shards = shard_scenes(scenes, 1)
    manifest_uri = str(tmp_path / "m.json")
    state.put_text(manifest_uri, m.to_json())
    state.put_text(state.checkpoint_uri(out, 0), "{not valid json")

    st = plan.status(manifest_uri, detail=True)
    assert st["shards_done"] == 1          # checkpoint exists → counts as done
    assert st["ledger"]["shards"] == []    # but contributes no ledger row
