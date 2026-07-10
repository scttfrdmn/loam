# Changelog

All notable changes to loam are documented here. Format loosely follows Keep a Changelog;
versions follow semver.

## [Unreleased]

### Added
- **Runner integration: `loam dispatch --runner lagotto`** (closes #8): emits a capacity-watch
  fleet — a generated spawn-config whose command pulls shards from the pool + `lagotto watch`
  /`poll` lines — so a scarce-capacity fleet drains the manifest (spot reclaim is a non-event).
  Still prints only; loam calls nothing. (Dispatch also gains its first tests, covering all three
  runners.)
- **nf-spawn Nextflow example** (closes #7): `examples/nextflow/loam.nf` + README fan `run-shard`
  over shards on ephemeral EC2 — proving the contract with **zero loam code change**.
- **`docs/MIGRATION.md`** — a hands-on SageMaker Geospatial → loam transition guide: EOJ-config →
  loam-CLI mapping table, boto3 before/after, status/export differences, and a pointer to the
  planned compat shim (#9). Linked from README and PARITY.
- **`docs/PARITY.md`** — a SageMaker Geospatial parity matrix (EOJ / VEJ / executor + viewer),
  what loam covers vs. deliberately does differently, and the honest remaining gaps. Linked from
  the README and DESIGN. Filed the clearest unmet op gap: zonal statistics (#34).

### Changed
- README Status refreshed to v0.2.0 (was stale at v0.1.0).
- `loam.__version__` is now read from installed package metadata (`importlib.metadata`), so
  `pyproject.toml` is the single source of truth and the CLI can't drift from the published
  version. The release workflow now asserts the built wheel's `loam --version` equals the tag.

## [0.2.0] — 2026-07-08

Parity ops: loam now covers the operations half of SageMaker Geospatial's EOJ **and** VEJ —
band-math, cloud-mask, resample/reproject, temporal composites, and reverse-geocode — plus
per-shard compute-shape estimates and a job ledger.

### Added
- **Vector enrichment: reverse-geocode** (closes #12, v1): `loam plan --op reverse-geocode --input
  points.csv` (CSV or GeoJSON of lat/lon) chunks rows into shards; `run-shard` appends place
  columns (`geo_name/geo_admin1/geo_admin2/geo_cc`) via an **offline** backend (`reverse_geocoder`,
  optional `loam-geo[vector]` extra) — deterministic, network-free, spot-safe. The first non-raster
  op: it bypasses STAC search and rides the same manifest/shard/state machinery (`loam/vector.py`;
  backend pluggable for a future online option). Map-match and a Nominatim (online) backend are
  tracked as follow-ups.
- **Temporal composite / geomosaic op** (closes #6): `loam plan --op temporal-composite --reducer
  median|mean|max --indices NDVI` (or `--bands`) reduces a stack of scenes over time into one
  cloud-free-ish mosaic per MGRS tile. New spatial sharder `manifest.shard_by_tile` groups a tile's
  full time series into one shard (deterministic; fails loud at plan time on unparseable ids).
  `ops.reduce_layers`/`temporal_composite` do a NaN-aware pixel-wise reduce over SCL-masked layers
  resampled to a common grid — pure numpy, **no stackstac/xarray**. `run_shard` reads date-by-date
  so a bad date is dropped (recorded in `failed`), failing only if none survive. Sentinel-2 only in
  v1; memory bounded via `target_res` (full-res refused). `loam/shape.py` models the whole-stack
  peak RAM (scales with scene count). Windowed full-res reads deferred to v2.
- **Per-shard compute-shape estimation** (closes #17): `loam plan` now attaches a `shape` block to
  each shard — bands read, approx decoded bytes read, peak working-set RAM, and an estimated
  runtime — computed by the new pure `loam/shape.py` from metadata alone, with **zero pixel reads
  or network I/O** (an import-scan test enforces that). `loam plan` prints an aggregate footer and
  `loam dispatch` surfaces each shard's estimate as a comment above its runner command — a
  right-sizing signal for truffle/a cost model. loam *describes* demand; it never provisions.
  Manifest is **v2**: the optional `shape` field is backward/forward-compatible (old manifests load
  with `shape=None`; `from_json` now tolerates unknown keys, protecting all future additive fields).
- **S3 job ledger** (closes #24): `loam status --detail` aggregates a job ledger — total
  outputs/bytes/seconds and a failed-scene rollup, plus per-shard rows — from the summaries
  `run_shard` now writes into each checkpoint (enriched with `bytes_written` + `seconds`). Purely
  read-derived from S3 (no writes at status time, no control plane); a malformed checkpoint is
  skipped rather than sinking the view. Plain `loam status` output is unchanged.
- **Resample / reproject op** (closes #22): `loam plan --op resample --bands red,nir
  --dst-crs EPSG:4326 [--dst-res R] [--resampling bilinear|nearest|…]` reprojects each requested
  band to a target CRS/resolution and writes georeferenced COGs — one per band. Warp detail lives
  in `raster.reproject_raster` (rasterio.warp); the overview-read path is preserved so bytes read
  stay bounded. A core SageMaker-parity capability and a clean-grid prerequisite for temporal
  composites (#6).
- **More built-in indices** (closes #23): the band-math catalog gains NDWI (McFeeters), SAVI,
  GNDVI, NDMI, NDRE, and ARVI (7 → 13), each with a cited equation, so users get them by name
  without a `NAME=equation` custom spec. A test asserts every catalog equation validates under the
  safe evaluator and references only known bands.
- **Contributor docs + contract guard** (closes #14): `docs/DESIGN.md` (in-repo design of record —
  the two-halves split, the three execution-agnostic properties, the "executor was the liability"
  framing, scope tiers) and `CONTRIBUTING.md` (uv dev setup, offline vs `LOAM_LIVE_TESTS=1`,
  ruff/mypy, and the one rule). `tests/test_contract.py` mechanically fails the build if `loam/`
  imports the substrate (spawn/lagotto/…) or contains EC2 launch/terminate calls — via a static
  AST scan, so lazy imports are caught too.

### Changed
- Bumped pinned GitHub Actions off deprecated Node 20 runtimes: `astral-sh/setup-uv` v6 → v8.3.1,
  `softprops/action-gh-release` v2 → v3.0.1.

## [0.1.0] — 2026-07-07

Initial Tier-1 MVP. An execution-agnostic open replacement for the operations half of Amazon
SageMaker Geospatial's Earth Observation Jobs, born from the fieldwork/BuckAI engagement after
SageMaker Geospatial closed to new customers (2026-07-30).

### Security
- **Safe band-math evaluation** (closes #4): `band_math` does not use Python `eval`. A custom
  `NAME=equation` index spec — or an equation string in a manifest of unknown origin — is
  evaluated by a zero-dependency AST allowlist (`indices.safe_eval`) that permits only numeric
  literals, band names, and `+ - * / **` / unary ±. Anything else (calls, attribute/dunder
  access, subscripts, comparisons, non-numeric constants, …) raises `ValueError` and never
  executes. `parse_spec` validates at `loam plan` time so a bad spec fails early. All 7 catalog
  equations compute byte-identically.

### Added
- **Band-math catalog** (`loam.indices`): NDVI, BSI, EVI, MNDWI, NDBI, NBR, NDSI — ported
  verbatim from the reference implementation (`spawn-sagemaker/internal/sagemaker/eoj.go`).
  Plus `NAME=equation` custom-index specs.
- **STAC catalog search** (`loam.catalog`): Sentinel-2 L2A / Landsat via Element84 Earth
  Search (any region, no AWS onboarding); asset-name normalization to canonical bands.
- **Manifest model** (`loam.manifest`): work-as-data — scenes, deterministic sharding, JSON
  round-trip. The document a runner fans out over.
- **Operations** (`loam.ops`): `band_math` and `cloud_mask` over COGs read via `/vsicurl` at
  the nearest overview level; SCL-based cloud masking.
- **run-shard atom** (`loam.run`): idempotent, spot-safe, delete-after-durable (checkpoint
  written last). The single command every runner schedules.
- **State in S3** (`loam.state`): no control plane — a shard is done iff its checkpoint object
  exists. Local-path fallback so the whole pipeline runs with zero AWS in tests.
- **plan / status** (`loam.plan`): build+write a manifest; report progress from S3.
- **CLI** (`loam.cli`): `indices`, `collections`, `plan`, `run-shard`, `status`, `dispatch`.
  `dispatch` prints spawn / local runner commands but never executes them (agnostic seam).
- **Georeferenced output** (closes #5): ops return a `Raster` (array + affine transform + CRS +
  nodata), and `run-shard` writes **Cloud-Optimized GeoTIFF** by default (`--format`
  `cog`|`gtiff`|`npy`). Downsampled reads scale the transform to the returned grid, so outputs
  are georeferenced at the correct resolution. New `loam/raster.py` owns all rasterio write
  detail (read → compute → write GeoTIFF via an in-memory dataset, no filesystem touch).

### Infrastructure & tests
- **CI + uv toolchain** (closes #1): GitHub Actions runs ruff + mypy + pytest on push/PR across
  Python 3.10/3.11/3.12. loam standardizes on [uv](https://docs.astral.sh/uv/) — a committed
  `uv.lock` pins the dependency set CI installs (`uv sync --locked`). mypy target set to 3.12 so
  numpy 2.5's 3.12+ stub grammar parses.
- **Release automation** (closes #2): `.github/workflows/release.yml` triggers on a `v*` tag —
  verifies the tag matches `pyproject.toml`'s version, builds sdist+wheel with `uv build`,
  smoke-tests the wheel in a clean env, publishes to PyPI via **Trusted Publishing (OIDC, no
  stored token)**, and cuts a GitHub Release. The PyPI distribution is **`loam-geo`** (the bare
  `loam` name was taken); the import name stays `loam`.
- **Live STAC integration tests** (closes #3): opt-in `tests/test_integration.py` (skipped
  unless `LOAM_LIVE_TESTS=1`) exercises the real Earth Search path + a full `run_shard` over real
  Sentinel-2 COGs; default `pytest` stays hermetic. Confirmed Earth Search v1 exposes lowercase
  canonical asset keys (`red`, `nir`, `scl`, …), so `_S2_ASSET_ALIASES` is retained only for
  other catalogs (e.g. Planetary Computer's `B04`).
- Core test suite that runs without network or AWS.
