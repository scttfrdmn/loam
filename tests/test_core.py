"""Core tests — the parts that must hold without any AWS or network.

These exercise the execution-agnostic contract directly: manifests round-trip, sharding is
deterministic, band-math evaluates correctly over synthetic rasters, and run-shard is
idempotent against a local-path "object store" (state.py treats non-s3:// URIs as files).
"""

from __future__ import annotations

import numpy as np
import pytest

from loam import indices
from loam.manifest import Manifest, Scene, shard_by_tile, shard_scenes, MANIFEST_VERSION


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


def test_shape_band_math_ndvi():
    from loam import shape

    # NDVI reads nir,red (10m) + scl (20m). At target_res=100 all collapse to 1098 px/side.
    s = shape.shape_for("band-math", {"indices": ["NDVI"], "target_res": 100.0}, 5, "sentinel-2-l2a")
    side = int(10980 * 10 / 100)  # 1098
    px = side * side
    assert s["scenes"] == 5
    assert s["bands_read"] == 3          # nir, red, scl
    assert s["outputs"] == 1
    assert s["approx_bytes_read"] == 5 * 3 * px * 4
    # peak = one scene: max_i(bands)=2 inputs + 1 output + 1 slack, on the finest grid
    assert s["peak_rss_bytes"] == px * (2 + 1 + 1) * 4


def test_shape_multi_index_peak_uses_max_not_union():
    from loam import shape

    # NDVI(2) + BSI(2) + EVI(3): union of bands is larger than any single index. Peak RAM must be
    # driven by max_i(n_bands_i)=3, NOT the union — a scene loads one index's bands at a time.
    s = shape.shape_for(
        "band-math", {"indices": ["NDVI", "BSI", "EVI"], "target_res": 100.0}, 10, "sentinel-2-l2a"
    )
    side = int(10980 * 10 / 100)
    px = side * side
    assert s["outputs"] == 3
    assert s["peak_rss_bytes"] == px * (3 + 3 + 1) * 4  # max_i=3 inputs + 3 outputs + 1


def test_shape_per_op_bands():
    from loam import shape

    # cloud-mask reads only scl; resample reads only its --bands (NO scl injected).
    cm = shape.shape_for("cloud-mask", {"target_res": 100.0}, 4, "sentinel-2-l2a")
    assert cm["bands_read"] == 1 and cm["outputs"] == 1
    rs = shape.shape_for("resample", {"bands": ["red", "nir"], "target_res": 100.0}, 4, "sentinel-2-l2a")
    assert rs["bands_read"] == 2 and rs["outputs"] == 2  # scl not counted


def test_shape_target_res_scales_and_clamps():
    from loam import shape

    coarse = shape.shape_for("band-math", {"indices": ["NDVI"], "target_res": 100.0}, 1, "sentinel-2-l2a")
    full = shape.shape_for("band-math", {"indices": ["NDVI"], "target_res": None}, 1, "sentinel-2-l2a")
    # full-res reads far more bytes than 100 m
    assert full["approx_bytes_read"] > coarse["approx_bytes_read"] * 50
    # can't read finer than native: target_res below 10 m clamps to the 10 m grid (== full-res)
    finer = shape.shape_for("band-math", {"indices": ["NDVI"], "target_res": 1.0}, 1, "sentinel-2-l2a")
    assert finer["approx_bytes_read"] == full["approx_bytes_read"]


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
    m.shards[0].shape = {"scenes": 1, "peak_rss_bytes": 123}
    m2 = Manifest.from_json(m.to_json())
    assert m2.op == "band-math"
    assert m2.scenes[0].assets["nir"] == "n"
    assert m2.scenes_for(0)[0].id == "s0"
    assert m2.shards[0].shape == {"scenes": 1, "peak_rss_bytes": 123}


def test_manifest_loads_old_and_unknown_keys():
    # A v1 manifest (no shape) still loads (shape defaults None); a manifest with an unknown extra
    # key still loads (from_json filters to known fields — forward-compat for additive fields).
    import json

    old = json.dumps({
        "version": 1, "op": "cloud-mask", "params": {}, "collection": "sentinel-2-l2a",
        "aoi": [0, 0, 1, 1], "output_uri": "s3://b/o/",
        "scenes": [{"id": "s0", "datetime": "2023-01-01", "assets": {}}],
        "shards": [{"index": 0, "scene_ids": ["s0"]}],
    })
    m = Manifest.from_json(old)
    assert m.shards[0].shape is None

    future = json.dumps({
        "version": 99, "op": "cloud-mask", "params": {}, "collection": "sentinel-2-l2a",
        "aoi": [0, 0, 1, 1], "output_uri": "s3://b/o/", "future_top_key": "ignored",
        "scenes": [{"id": "s0", "datetime": "2023-01-01", "assets": {}, "future_scene_key": 1}],
        "shards": [{"index": 0, "scene_ids": ["s0"], "shape": {"scenes": 1}, "future_key": 2}],
    })
    m2 = Manifest.from_json(future)  # must not raise on unknown keys
    assert m2.shards[0].shape == {"scenes": 1}


def test_build_manifest_attaches_shape(monkeypatch):
    # build_manifest must populate each shard's shape (offline: fake the STAC search).
    from loam import catalog, plan

    fake_scenes = [Scene(id=f"s{i}", datetime="2023-01-01", assets={"nir": "n", "red": "r"})
                   for i in range(3)]
    monkeypatch.setattr(catalog, "search", lambda **kw: fake_scenes)

    m = plan.build_manifest(
        op="band-math", collection="sentinel-2", aoi=[0, 0, 1, 1],
        start="2023-01-01", end="2023-12-31", indices=["NDVI"],
        shard_size=2, output_uri="s3://b/o/",
    )
    assert m.version == 2
    assert len(m.shards) == 2  # 3 scenes / 2
    for sh in m.shards:
        assert sh.shape is not None
        assert sh.shape["scenes"] == len(sh.scene_ids)
        assert sh.shape["bands_read"] == 3  # nir, red, scl
    # last shard (1 scene) has smaller read estimate than the full one (2 scenes)
    assert m.shards[1].shape["approx_bytes_read"] < m.shards[0].shape["approx_bytes_read"]


def test_build_manifest_temporal_composite_shards_by_tile(monkeypatch):
    from loam import catalog, plan

    scenes = [
        Scene(id="S2B_30QTH_20230101_0_L2A", datetime="2023-01-01", assets={"nir": "n", "red": "r"}),
        Scene(id="S2A_30QTH_20230201_0_L2A", datetime="2023-02-01", assets={"nir": "n", "red": "r"}),
        Scene(id="S2B_31QAB_20230101_0_L2A", datetime="2023-01-01", assets={"nir": "n", "red": "r"}),
    ]
    monkeypatch.setattr(catalog, "search", lambda **kw: scenes)
    m = plan.build_manifest(
        op="temporal-composite", collection="sentinel-2", aoi=[0, 0, 1, 1],
        start="2023-01-01", end="2023-12-31", indices=["NDVI"], reducer="median",
        target_res=100.0, output_uri="s3://b/o/",
    )
    assert len(m.shards) == 2  # one per tile, NOT per scene-count
    assert m.params["reducer"] == "median" and m.params["index"] == "NDVI"
    assert m.shards[0].shape["outputs"] == 1


def test_build_manifest_temporal_composite_gates(monkeypatch):
    from loam import catalog, plan
    monkeypatch.setattr(catalog, "search", lambda **kw: [])

    # non-Sentinel-2 collection is refused in v1
    with pytest.raises(ValueError, match="only sentinel-2"):
        plan.build_manifest(op="temporal-composite", collection="landsat-8", aoi=[0, 0, 1, 1],
                            start="2023-01-01", end="2023-12-31", indices=["NDVI"],
                            target_res=100.0, output_uri="s3://b/o/")
    # full-res composite is refused (would stack the whole tile in memory)
    with pytest.raises(ValueError, match="coarse --target-res"):
        plan.build_manifest(op="temporal-composite", collection="sentinel-2", aoi=[0, 0, 1, 1],
                            start="2023-01-01", end="2023-12-31", indices=["NDVI"],
                            target_res=None, output_uri="s3://b/o/")


def test_shape_module_does_no_io():
    # The compute-shape path must never read pixels or hit the network: assert loam/shape.py
    # imports none of the I/O modules (static AST scan, like tests/test_contract.py).
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "loam" / "shape.py"
    tree = ast.parse(src.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    assert not (roots & {"rasterio", "pystac_client", "boto3"}), roots
    # and no loam I/O modules
    assert "ops" not in roots and "raster" not in roots and "catalog" not in roots


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


# ── temporal-composite (#6) ──────────────────────────────────────────────────

def _layer(*rows):
    from loam.raster import Raster
    return Raster(np.array(rows, np.float32), (10.0, 0, 0, 0, -10.0, 0), "EPSG:32629", None)


@pytest.mark.parametrize("reducer,expected", [("median", 1.5), ("mean", 1.5), ("max", 2.0)])
def test_reduce_layers_reducers(reducer, expected):
    from loam import ops
    # pixel [0,0] across 3 dates: 1, 2, NaN(cloud) → median/mean 1.5, max 2 (NaN ignored)
    layers = [_layer([1.0, 9], [9, 9]), _layer([2.0, 9], [9, 9]), _layer([np.nan, 9], [9, 9])]
    out = ops.reduce_layers(layers, reducer)
    assert out.data[0, 0] == pytest.approx(expected)
    assert out.crs == "EPSG:32629"


def test_reduce_layers_all_cloud_pixel_stays_nan():
    from loam import ops
    layers = [_layer([np.nan, 1], [1, 1]) for _ in range(3)]
    out = ops.reduce_layers(layers, "median")
    assert np.isnan(out.data[0, 0]) and out.data[1, 1] == pytest.approx(1.0)


def test_reduce_layers_aligns_mismatched_shapes():
    from loam import ops
    from loam.raster import Raster
    big = Raster(np.full((4, 4), 3.0, np.float32), (10.0, 0, 0, 0, -10.0, 0), "EPSG:32629", None)
    small = Raster(np.full((2, 2), 9.0, np.float32), (20.0, 0, 0, 0, -20.0, 0), "EPSG:32629", None)
    out = ops.reduce_layers([big, small], "max")  # must not raise on shape mismatch
    assert out.data.shape == (4, 4)  # reduced onto the finest grid
    assert out.data.max() == pytest.approx(9.0)


def test_reduce_layers_bad_reducer_raises():
    from loam import ops
    with pytest.raises(ValueError, match="unknown reducer"):
        ops.reduce_layers([_layer([1.0])], "p95")


def test_temporal_composite_needs_exactly_one_target():
    from loam import ops
    with pytest.raises(ValueError, match="exactly one"):
        ops.temporal_composite([{"nir": "n"}], index="NDVI", band="red")


def test_shard_by_tile_groups_and_orders():
    scenes = [
        Scene(id="S2B_31QAB_20230101_0_L2A", datetime="2023-01-01", assets={}),
        Scene(id="S2B_30QTH_20230101_0_L2A", datetime="2023-01-01", assets={}),
        Scene(id="S2A_30QTH_20230201_0_L2A", datetime="2023-02-01", assets={}),
    ]
    shards = shard_by_tile(scenes, "sentinel-2-l2a")
    assert len(shards) == 2
    # tiles sorted → 30QTH is shard 0 (2 dates), 31QAB is shard 1 (deterministic)
    assert [s.index for s in shards] == [0, 1]
    assert len(shards[0].scene_ids) == 2 and len(shards[1].scene_ids) == 1


def test_shard_by_tile_unparseable_raises():
    with pytest.raises(ValueError, match="cannot parse a spatial tile"):
        shard_by_tile([Scene(id="not-a-sentinel-id", datetime="", assets={})], "sentinel-2-l2a")


def test_run_shard_temporal_composite_end_to_end(tmp_path, monkeypatch):
    # Full path: a 3-date tile → run_shard writes ONE composite GeoTIFF; a bad date is recorded
    # but doesn't sink the shard.
    import rasterio
    from loam import ops, run, state

    out = str(tmp_path / "out")
    scenes = [Scene(id=f"S2B_30QTH_2023010{i}_0_L2A", datetime=f"2023-01-0{i}",
                    assets={"nir": f"n{i}", "red": f"r{i}"}) for i in range(1, 4)]
    m = Manifest(
        version=MANIFEST_VERSION, op="temporal-composite",
        params={"format": "cog", "reducer": "median", "index": "NDVI", "target_res": 100.0},
        collection="sentinel-2-l2a", aoi=[0, 0, 1, 1], output_uri=out, scenes=scenes,
    )
    m.shards = shard_by_tile(scenes, "sentinel-2-l2a")
    manifest_uri = str(tmp_path / "m.json")
    state.put_text(manifest_uri, m.to_json())

    # scene_layer reads bands via band_math → _read_band_raster; nir=3,red=1 → NDVI 0.5.
    # Make the "r3" (third date red) href fail, to exercise drop-and-record.
    def fake_read(href, target_res=None):
        if href == "r3":
            raise OSError("boom reading r3")
        return _fake_raster(3.0 if href.startswith("n") else 1.0, 32, 32)
    monkeypatch.setattr(ops, "_read_band_raster", fake_read)

    s = run.run_shard(manifest_uri, 0)
    assert s["status"] == "done"
    assert s["scenes"] == 3
    assert s["outputs"] == 1                     # one mosaic
    assert len(s["failed"]) == 1                 # the r3 date dropped + recorded
    tif = state.output_uri_for(out, 0, "composite__NDVI.tif")
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:32629"
        assert abs(float(src.read(1).mean()) - 0.5) < 1e-4


def test_shape_temporal_composite_peak_scales_with_scenes():
    from loam import shape
    s = shape.shape_for(
        "temporal-composite", {"index": "NDVI", "reducer": "median", "target_res": 100.0},
        30, "sentinel-2-l2a",
    )
    assert s["outputs"] == 1
    assert s["bands_read"] == 3  # nir, red, scl
    # peak holds the WHOLE stack: finest_px * (n_scenes+1) * 4
    side = int(10980 * 10 / 100)
    assert s["peak_rss_bytes"] == side * side * (30 + 1) * 4


# ── vector enrichment / reverse-geocode (#12) ────────────────────────────────

def test_vector_read_write_csv_roundtrip():
    from loam import vector
    text = "name,lat,lon\nA,18.9,-3.5\nB,40.7,-74.0\n"
    rows, coords = vector.read_points(text, "csv")
    assert coords == [(18.9, -3.5), (40.7, -74.0)]
    out = vector.write_enriched(rows, [{"geo_name": "X", "geo_cc": "ML"},
                                       {"geo_name": "Y", "geo_cc": "US"}], "csv")
    assert "geo_name" in out.splitlines()[0] and "X" in out and "US" in out


def test_vector_read_geojson_and_custom_csv_fields():
    from loam import vector
    import json
    gj = json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": 1},
         "geometry": {"type": "Point", "coordinates": [-3.5, 18.9]}},  # GeoJSON is [lon, lat]
    ]})
    rows, coords = vector.read_points(gj, "geojson")
    assert coords == [(18.9, -3.5)]
    out = json.loads(vector.write_enriched(rows, [{"geo_cc": "ML"}], "geojson"))
    assert out["features"][0]["properties"]["geo_cc"] == "ML"
    # custom lat/lon column names
    csv_text = "y,x,v\n40.7,-74.0,q\n"
    _, c2 = vector.read_points(csv_text, "csv", lat_field="y", lon_field="x")
    assert c2 == [(40.7, -74.0)]


def test_vector_errors():
    from loam import vector
    with pytest.raises(ValueError, match="no lat/lon column"):
        vector.read_points("a,b\n1,2\n", "csv")
    with pytest.raises(ValueError, match="unsupported vector format"):
        vector.read_points("x", "parquet")
    with pytest.raises(ValueError, match="unknown backend"):
        vector.reverse_geocode([(0.0, 0.0)], backend="nominatim")


def test_reverse_geocode_offline_backend():
    pytest.importorskip("reverse_geocoder")
    from loam import vector
    enr = vector.reverse_geocode([(40.7, -74.0)])  # NYC
    assert enr[0]["geo_cc"] == "US"
    assert enr[0]["geo_name"]  # a place name resolved


def test_build_manifest_reverse_geocode_no_stac(tmp_path, monkeypatch):
    from loam import catalog, plan

    # If build_manifest touches STAC for a row op, this raises — proving the search bypass.
    def boom(**kw):
        raise AssertionError("catalog.search must not be called for reverse-geocode")
    monkeypatch.setattr(catalog, "search", boom)

    csv_path = tmp_path / "pts.csv"
    csv_path.write_text("name,lat,lon\nA,18.9,-3.5\nB,40.7,-74.0\nC,48.85,2.35\n")
    out = str(tmp_path / "out")
    m = plan.build_manifest(
        op="reverse-geocode", output_uri=out, input_uri=str(csv_path),
        rows_per_shard=2, fmt="csv",
    )
    assert m.collection == "vector" and m.aoi == []
    assert len(m.shards) == 2                       # 3 rows / 2 → 2 chunks
    assert m.scenes[0].assets["rows"].endswith(".csv")


def test_build_manifest_reverse_geocode_requires_input():
    from loam import plan
    with pytest.raises(ValueError, match="requires --input"):
        plan.build_manifest(op="reverse-geocode", output_uri="s3://b/o/", fmt="csv")


def test_run_shard_reverse_geocode_end_to_end(tmp_path):
    pytest.importorskip("reverse_geocoder")
    from loam import plan, run, state

    csv_path = tmp_path / "pts.csv"
    csv_path.write_text("name,lat,lon\nAraouane,18.9,-3.5\nNYC,40.7,-74.0\n")
    out = str(tmp_path / "out")
    manifest_uri = str(tmp_path / "m.json")
    m = plan.build_manifest(op="reverse-geocode", output_uri=out, input_uri=str(csv_path),
                            rows_per_shard=5, fmt="csv")
    plan.write_manifest(m, manifest_uri)

    s = run.run_shard(manifest_uri, 0)
    assert s["status"] == "done" and s["outputs"] == 1 and not s["failed"]
    enriched = state.get_text(state.output_uri_for(out, 0, "chunk-00000__enriched.csv"))
    assert "geo_cc" in enriched and "US" in enriched and "ML" in enriched
    # idempotent re-run
    assert run.run_shard(manifest_uri, 0)["status"] == "skipped"


def test_shape_reverse_geocode_is_trivial():
    from loam import shape
    s = shape.shape_for("reverse-geocode", {"format": "csv"}, 4, "vector")
    assert s["scenes"] == 4 and s["outputs"] == 1
    assert s["approx_bytes_read"] == 0 and s["peak_rss_bytes"] == 0


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
