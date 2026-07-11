# loam guide — composing ops and runners

This is the "how do I actually drive it" guide: how loam's operations and runners fit **together**
to do real work. It does not document the tools loam sits on (rasterio, STAC, `spawn`/`lagotto`,
Valhalla/OSRM) — those have their own docs; loam only *points at* them. For the *why* behind the
design see [DESIGN.md](DESIGN.md); for a SageMaker Geospatial → loam mapping see
[MIGRATION.md](MIGRATION.md); for the parity matrix see [PARITY.md](PARITY.md).

## The mental model: a manifest is the unit of work

Everything in loam is one loop:

```
plan  ──►  manifest (work as data)  ──►  dispatch  ──►  run-shard × N  ──►  status
```

- **`plan`** searches a catalog (or reads a file), shards the work, and writes a **manifest** — a
  JSON document listing the scenes/rows and the operation. No pixels are read; nothing is launched.
- The **manifest is the seam.** It's the single source of truth a runner consumes. It lives in S3
  (or a local path in tests). You can read it, diff it, hand it to any runner.
- **`run-shard --manifest <uri> -i N`** is the atom: it processes one shard, writes outputs +
  a checkpoint to S3, and is **idempotent** — re-running a shard whose checkpoint exists is a
  no-op. That's what makes it spot-safe: a reclaimed shard is just re-run.
- **`status`** is derived purely from S3 (a shard is done iff its checkpoint object exists) —
  no control plane, no job to poll. `--detail` rolls up bytes/time/failures.
- **`dispatch`** *prints* the commands to run each shard on a runner you pick. It never launches
  anything — that's the seam that keeps loam execution-agnostic (you choose the runner; loam
  doesn't own it).

Two consequences worth internalizing:

- **You chain ops by URI, not by an API.** One op writes COGs/tables to `--output`; the next op
  points its `--raster`/`--input` at those files. There's no in-memory pipeline object — the
  filesystem/S3 *is* the pipeline. (See the zonal-stats recipe below.)
- **The runner is chosen per run, not baked in.** The same manifest runs on your laptop, one
  `spawn` box per shard, a `lagotto` capacity-watch fleet, or a Nextflow pipeline — with zero
  changes to loam.

## Operations reference

Each `loam plan --op <op>` produces a manifest; `run-shard` executes it. Required flags and the
input/output shape:

| `--op` | needs | reads | writes |
|---|---|---|---|
| `band-math` | `--aoi --start --end --indices` | Sentinel-2/Landsat COGs (STAC) | one COG per index per scene |
| `cloud-mask` | `--aoi --start --end` | SCL band | one mask COG per scene |
| `resample` | `--aoi --start --end --bands --dst-crs` | scene bands | reprojected COG per band |
| `temporal-composite` | `--aoi --start --end` + `--indices`/`--bands` (one) + coarse `--target-res` | a tile's time series | one mosaic COG per MGRS tile |
| `zonal-stats` | `--zones --raster` | an existing COG + polygon zones | per-zone stats table (GeoJSON/CSV) |
| `reverse-geocode` | `--input` | CSV/GeoJSON of points | enriched rows (`geo_*` columns) |
| `map-match` | `--input` | CSV/GeoJSON of trace points | matched LineString per trace (GeoJSON) |

Notes:
- **Raster ops** (`band-math`/`cloud-mask`/`resample`/`temporal-composite`) search a STAC catalog,
  so they need `--aoi W,S,E,N` + `--start/--end`. Output format is `--format cog|gtiff|npy`
  (default COG).
- **Row ops** (`reverse-geocode`/`map-match`) and **`zonal-stats`** read a file instead of
  searching — no `--aoi`/dates. `--output` is a table/GeoJSON.
- `zonal-stats` takes an **existing raster** (`--raster`) — it's step 2 after `band-math`, not a
  search itself.
- Run `loam indices` for the band-math catalog, `loam collections` for known STAC collections,
  and `loam plan --help` for the full flag list. (This table is checked against the CLI by a test.)

## Recipes

Each links a runnable script in [`examples/`](../examples).

### 1. Index → composite over a season (band-math)
Compute NDVI/BSI over an AOI and year — the founding H2-prospecting chain.
```bash
loam plan --op band-math --indices NDVI,BSI --collection sentinel-2 \
  --aoi -7.0,19.0,-3.0,22.0 --start 2023-01-01 --end 2023-12-31 --max-cloud 20 \
  --output s3://b/ndvi/ --manifest s3://b/ndvi/manifest.json
loam dispatch --manifest s3://b/ndvi/manifest.json --runner spawn   # prints commands
loam status   --manifest s3://b/ndvi/manifest.json
```
→ [`examples/h2_prospecting.sh`](../examples/h2_prospecting.sh)

### 2. Chain two ops by URI: band-math → zonal-stats
The compose pattern — one op's output is the next op's `--raster`. Summarize NDVI within field
polygons:
```bash
# step 1 writes NDVI COGs (recipe 1) → shard=00000/<scene>__NDVI.tif
loam plan --op zonal-stats --zones fields.geojson \
  --raster s3://b/ndvi/shard=00000/<scene>__NDVI.tif --stat mean,min,max,p90 \
  --output s3://b/zonal/ --manifest s3://b/zonal/manifest.json
loam run-shard --manifest s3://b/zonal/manifest.json -i 0
```
→ [`examples/zonal_stats.sh`](../examples/zonal_stats.sh)

### 3. Cloud-free mosaic (temporal-composite)
Reduce a tile's time series to one median mosaic (per MGRS tile; needs a coarse `--target-res`):
```bash
loam plan --op temporal-composite --indices NDVI --reducer median \
  --collection sentinel-2 --aoi -7,19,-3,22 --start 2023-06-01 --end 2023-08-31 \
  --target-res 100 --output s3://b/mosaic/ --manifest s3://b/mosaic/manifest.json
```

### 4. Vector enrichment (reverse-geocode, map-match)
Row ops — read a file, no STAC:
```bash
loam plan --op reverse-geocode --input points.csv --backend offline \
  --output s3://b/rg/ --manifest s3://b/rg/manifest.json         # offline city/admin, or --backend nominatim
loam plan --op map-match --input traces.csv --trace-field trace_id --backend valhalla \
  --output s3://b/mm/ --manifest s3://b/mm/manifest.json          # or --backend osrm
```
→ [`examples/map_match.sh`](../examples/map_match.sh)

## Choosing a runner

`loam dispatch --runner <r>` prints how to fan a manifest's shards over each. loam prints; you (or
an orchestrator) run. All are spot-safe because the shard is the idempotent atom — a reclaimed
shard just re-runs.

| Runner | When | Shape |
|---|---|---|
| `local` | dev, small jobs, a smoke test before a fleet | a bare `for` loop over shards on one box |
| `spawn` | scale-out; each shard is independent | one ephemeral EC2 box per shard (fan out) |
| `lagotto` | scarce/expensive capacity (GPU, large instances) | a capacity-watch fleet that *drains* the manifest pool |
| Nextflow (`nf-spawn`) | you already run Nextflow | a process per shard on ephemeral EC2 — see [`examples/nextflow/`](../examples/nextflow) |

loam depends on none of these — it emits the command and stops. See
[`spore.host`](https://github.com/spore-host) (`spawn`/`lagotto`/`truffle`) and
[`nf-spawn`](https://github.com/spore-host/nf-spawn) for the runners themselves. The pull model
means a shard interrupted by spot reclaim is simply re-run; no bookkeeping, no lost session.

## Coming from SageMaker Geospatial?

If you're porting an EOJ/VEJ workflow, [MIGRATION.md](MIGRATION.md) maps each piece of an EOJ
config to the `loam plan` flags above, with before/after code. The short version: one opaque
managed job → `plan` (a manifest you can see) + `dispatch` to a runner you choose + idempotent
shards; `IN_PROGRESS` polling → `loam status --detail` (real progress from S3).
