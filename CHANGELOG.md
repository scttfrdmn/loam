# Changelog

All notable changes to loam are documented here. Format loosely follows Keep a Changelog;
versions follow semver.

## [0.1.0] — unreleased

Initial Tier-1 MVP. An execution-agnostic open replacement for the operations half of Amazon
SageMaker Geospatial's Earth Observation Jobs, born from the fieldwork/BuckAI engagement after
SageMaker Geospatial closed to new customers (2026-07-30).

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
- Core test suite that runs without network or AWS.
