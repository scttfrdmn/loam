"""plan — build a manifest from a search + an operation, and report status.

``plan`` is the pre-compute step: it searches the catalog, shards the scenes, attaches the
operation, and writes the manifest to S3. No pixels are read here. The result is a document a
runner fans out over via ``run-shard``.

``status`` derives progress purely from S3 (done shards = checkpoints present) — no control
plane, no job to poll.
"""

from __future__ import annotations

from . import catalog, state
from .indices import parse_spec
from .manifest import Manifest, MANIFEST_VERSION, shard_by_tile, shard_scenes


def build_manifest(
    *,
    op: str,
    output_uri: str,
    collection: str = "sentinel-2",
    aoi: list[float] | None = None,
    start: str | None = None,
    end: str | None = None,
    indices: list[str] | None = None,
    bands: list[str] | None = None,
    dst_crs: str | None = None,
    dst_res: float | None = None,
    resampling: str = "bilinear",
    reducer: str = "median",
    max_cloud: float | None = None,
    shard_size: int = 50,
    limit: int | None = None,
    fmt: str = "cog",
    target_res: float | None = 100.0,
    stac_url: str = catalog.DEFAULT_STAC_URL,
    # vector (row) ops — reverse-geocode reads points from a file instead of searching STAC
    input_uri: str | None = None,
    rows_per_shard: int = 5000,
    lat_field: str | None = None,
    lon_field: str | None = None,
    backend: str = "offline",
    # zonal-stats — reduce an existing raster COG within polygon zones
    zones_uri: str | None = None,
    raster_uri: str | None = None,
    stats: list[str] | None = None,
) -> Manifest:
    """Search, shard, and assemble a Manifest (does not write it — caller persists).

    ``target_res`` (metres) sets the overview level ops read. Default 100m is fine for
    continental-scale change detection; pass ``None`` for native full resolution when the
    features of interest are small (e.g. Sentinel-2 10m for fairy-circle detection).
    """
    # Row (vector) ops read features from a file, not STAC — they bypass search entirely.
    if op == "reverse-geocode":
        return _build_vector_manifest(
            op=op, output_uri=output_uri, input_uri=input_uri, fmt=fmt,
            rows_per_shard=rows_per_shard, lat_field=lat_field, lon_field=lon_field,
            backend=backend,
        )
    if op == "zonal-stats":
        return _build_zonal_manifest(
            output_uri=output_uri, zones_uri=zones_uri, raster_uri=raster_uri,
            stats=stats, fmt=fmt, rows_per_shard=rows_per_shard,
        )

    # Raster ops require a search footprint + date range.
    if aoi is None or start is None or end is None:
        raise ValueError(f"op {op!r} requires --aoi, --start, and --end")

    params: dict = {"format": fmt, "target_res": target_res}
    wanted: set[str] = {"scl"}  # always fetch SCL so ops can cloud-mask

    if op == "band-math":
        if not indices:
            raise ValueError("band-math requires --indices")
        defs = [parse_spec(s) for s in indices]
        params["indices"] = indices
        from .indices import bands_in

        for d in defs:
            wanted |= bands_in(d.equation)
    elif op == "cloud-mask":
        pass  # only needs scl, already in wanted
    elif op == "resample":
        if not bands:
            raise ValueError("resample requires --bands")
        if not dst_crs:
            raise ValueError("resample requires --dst-crs")
        params.update(bands=bands, dst_crs=dst_crs, dst_res=dst_res, resampling=resampling)
        wanted |= set(bands)
    elif op == "temporal-composite":
        # v1: Sentinel-2 only (the per-tile no-reproject shortcut is MGRS-specific), and bounded
        # memory relies on target_res (the whole tile stack is materialized), so refuse full-res.
        if "sentinel-2" not in collection and "sentinel2" not in collection:
            raise ValueError("temporal-composite supports only sentinel-2 in v1")
        if target_res is None:
            raise ValueError(
                "temporal-composite needs a coarse --target-res (full-res stacks the whole tile "
                "over all dates in memory); e.g. --target-res 100"
            )
        one_index = indices[0] if indices else None
        one_band = bands[0] if bands else None
        if (one_index is None) == (one_band is None):
            raise ValueError("temporal-composite needs exactly one of --indices / --bands")
        params.update(reducer=reducer, index=one_index, band=one_band)
        from .indices import bands_in

        if one_index is not None:
            wanted |= bands_in(parse_spec(one_index).equation)
        else:
            assert one_band is not None  # exactly-one check above guarantees this
            wanted.add(one_band)
    else:
        raise ValueError(
            f"unknown op {op!r} (known: band-math, cloud-mask, resample, temporal-composite)"
        )

    scenes = catalog.search(
        collection=collection,
        aoi=aoi,
        start=start,
        end=end,
        max_cloud=max_cloud,
        limit=limit,
        stac_url=stac_url,
        wanted_bands=wanted,
    )
    coll_id = catalog.resolve_collection(collection)
    # temporal-composite groups a whole tile's time series into one shard; everything else shards
    # by scene count.
    if op == "temporal-composite":
        shards = shard_by_tile(scenes, coll_id)
    else:
        shards = shard_scenes(scenes, shard_size)

    # Attach a compute-shape estimate per shard (pure metadata — no pixel reads). Scene count
    # differs per shard, so each shard is estimated from its own membership.
    from . import shape as shapemod

    for sh in shards:
        sh.shape = shapemod.shape_for(op, params, len(sh.scene_ids), coll_id)

    return Manifest(
        version=MANIFEST_VERSION,
        op=op,
        params=params,
        collection=coll_id,
        aoi=aoi,
        output_uri=output_uri,
        scenes=scenes,
        shards=shards,
    )


def _build_vector_manifest(
    *, op: str, output_uri: str, input_uri: str | None, fmt: str,
    rows_per_shard: int, lat_field: str | None, lon_field: str | None,
    backend: str = "offline",
    region: str | None = None,
) -> Manifest:
    """Build a manifest for a row op: read a points file, chunk rows into shards.

    No STAC search. Each shard is a chunk of input rows, written under ``output_uri/_input/`` and
    referenced by a Scene's ``assets["rows"]`` — so ``run-shard`` reads just its chunk. Mirrors the
    raster manifest contract (work-as-data, deterministic sharding) for a non-raster input.
    """
    from . import vector
    from .manifest import Scene, shard_scenes

    if not input_uri:
        raise ValueError(f"{op} requires --input (a CSV or GeoJSON of points)")
    if fmt == "cog":
        fmt = "csv"  # library default falls through as cog; a row op defaults to csv
    if fmt not in ("csv", "geojson"):
        raise ValueError(f"{op} requires --format csv or geojson (got {fmt!r})")
    if rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be >= 1")
    if backend not in vector._BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {', '.join(vector._BACKENDS)}")

    text = state.get_text(input_uri, region=region)
    rows, _ = vector.read_points(text, fmt, lat_field=lat_field, lon_field=lon_field)

    scenes: list[Scene] = []
    for i in range(0, len(rows), rows_per_shard):
        chunk = rows[i : i + rows_per_shard]
        cid = f"chunk-{i // rows_per_shard:05d}"
        chunk_uri = state.join(output_uri, "_input", f"{cid}.{fmt}")
        # write the chunk in the SAME format, so run-shard reads it back with read_points
        state.put_text(chunk_uri, vector.write_chunk(chunk, fmt), region=region)
        scenes.append(Scene(id=cid, datetime="", assets={"rows": chunk_uri}))

    params = {
        "format": fmt, "lat_field": lat_field, "lon_field": lon_field, "backend": backend,
    }
    shards = shard_scenes(scenes, 1)  # one row-chunk per shard (the chunk already sized the work)
    return Manifest(
        version=MANIFEST_VERSION, op=op, params=params, collection="vector", aoi=[],
        output_uri=output_uri, scenes=scenes, shards=shards,
    )


def _build_zonal_manifest(
    *, output_uri: str, zones_uri: str | None, raster_uri: str | None,
    stats: list[str] | None, fmt: str, rows_per_shard: int, region: str | None = None,
) -> Manifest:
    """Build a manifest for zonal-stats: chunk polygon zones into shards over an existing raster.

    No STAC search. Reads the zones GeoJSON, chunks features into shards (like the row ops), and
    stores the raster COG href + requested stats in params — each shard reads that one raster.
    """
    from . import vector
    from .manifest import Scene, shard_scenes

    if not zones_uri:
        raise ValueError("zonal-stats requires --zones (a GeoJSON of polygon zones)")
    if not raster_uri:
        raise ValueError("zonal-stats requires --raster (a single-band COG, e.g. a band-math output)")
    if fmt == "cog":
        fmt = "geojson"  # library default falls through as cog; zones are geojson-native
    if fmt not in ("csv", "geojson"):
        raise ValueError(f"zonal-stats requires --format csv or geojson (got {fmt!r})")
    if rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be >= 1")
    stat_list = stats or ["mean", "min", "max", "count"]

    text = state.get_text(zones_uri, region=region)
    zones = vector.read_polygons(text)

    scenes: list[Scene] = []
    for i in range(0, len(zones), rows_per_shard):
        chunk = zones[i : i + rows_per_shard]
        cid = f"chunk-{i // rows_per_shard:05d}"
        chunk_uri = state.join(output_uri, "_input", f"{cid}.geojson")
        state.put_text(chunk_uri, vector.write_chunk(chunk, "geojson"), region=region)
        scenes.append(Scene(id=cid, datetime="", assets={"zones": chunk_uri}))

    params = {"format": fmt, "raster": raster_uri, "stats": stat_list}
    shards = shard_scenes(scenes, 1)
    return Manifest(
        version=MANIFEST_VERSION, op="zonal-stats", params=params, collection="vector", aoi=[],
        output_uri=output_uri, scenes=scenes, shards=shards,
    )


def write_manifest(manifest: Manifest, manifest_uri: str, *, region: str | None = None) -> None:
    state.put_text(manifest_uri, manifest.to_json(), region=region)


def status(manifest_uri: str, *, region: str | None = None, detail: bool = False) -> dict:
    """Return progress derived entirely from S3 (done = checkpoint object present).

    With ``detail=True``, also aggregate a **job ledger** from the per-shard summaries that
    ``run_shard`` wrote into each checkpoint — total outputs/bytes/seconds and a failed-scene
    rollup, plus per-shard rows. This is read-only (no writes at status time); a shard whose
    checkpoint is absent or unreadable simply doesn't contribute a row.
    """
    import json

    manifest = Manifest.from_json(state.get_text(manifest_uri, region=region))
    total = len(manifest.shards)
    done = sum(
        1
        for sh in manifest.shards
        if state.shard_done(manifest.output_uri, sh.index, region=region)
    )
    out = {
        "op": manifest.op,
        "scenes": len(manifest.scenes),
        "shards_total": total,
        "shards_done": done,
        "shards_remaining": total - done,
        "complete": done == total and total > 0,
    }
    if not detail:
        return out

    rows: list[dict] = []
    for sh in manifest.shards:
        cp = state.checkpoint_uri(manifest.output_uri, sh.index)
        if not state.exists(cp, region=region):
            continue
        try:
            rows.append(json.loads(state.get_text(cp, region=region)))
        except (ValueError, OSError):
            continue  # a malformed/partial checkpoint must not sink the whole status view
    out["ledger"] = {
        "outputs": sum(r.get("outputs", 0) for r in rows),
        "bytes_written": sum(r.get("bytes_written", 0) for r in rows),
        "seconds": round(sum(r.get("seconds", 0.0) for r in rows), 3),
        "failed_scenes": sum(len(r.get("failed", [])) for r in rows),
        "shards": sorted(rows, key=lambda r: r.get("shard", 0)),
    }
    return out
