# loam vs SageMaker Geospatial — parity

Amazon SageMaker Geospatial **closed to new customers on 2026-07-30**. A fresh AWS account can
never onboard again, and it only ever ran in `us-west-2`. loam exists to replace it — not by
cloning it, but by keeping the part that had value (the **operations**) and discarding the part
that was the liability (the **managed executor**).

This page is the honest scorecard: what loam covers, what it doesn't yet, and what it
**deliberately** does differently.

> **TL;DR** — For the actual work people ran on SageMaker Geospatial (spectral indices, cloud
> masking, cloud-free composites over Sentinel-2, resampling, reverse-geocoding), **loam is a
> complete replacement today** — running in any region, in a fresh account, with no runtime cap,
> spot-priced, and observable. The remaining op gaps are **zonal statistics** and **map-match**.

## SageMaker Geospatial was three surfaces

It fused three things. loam relates to each differently:

1. **Earth Observation Jobs (EOJ)** — the raster operations. *loam is at parity here.*
2. **Vector Enrichment Jobs (VEJ)** — reverse-geocode + map-match. *loam covers reverse-geocode.*
3. **The managed executor + Studio map viewer.** *loam deliberately does not reproduce the
   executor* — that was the worst part — and the viewer is future work.

## 1. Earth Observation Jobs (raster ops)

| SageMaker Geospatial operation | loam | Status |
|---|---|---|
| Raster Data Collection query | `loam plan` over a STAC catalog (Earth Search) | ✅ **better** — any region, fresh account, no Oregon lock |
| Cloud masking (SCL) | `--op cloud-mask` | ✅ parity |
| Band math / spectral indices | `--op band-math` — 14 built-in indices + safe custom `NAME=equation` | ✅ parity+ |
| Temporal statistics / cloud-removal composite | `--op temporal-composite` (median/mean/max) | ✅ parity (Sentinel-2, v1) |
| Geomosaic | same op (the composite *is* the mosaic) | ✅ parity |
| Resampling / reprojection | `--op resample` (reproject + regrid, rasterio.warp) | ✅ parity |
| Stacking | over *time* via composite; no arbitrary multi-band stack op | ⚠️ partial |
| **Zonal statistics** | — | ❌ planned ([#34](https://github.com/scttfrdmn/loam/issues/34)) |
| Export to S3 | native — every shard writes (Cloud-Optimized) GeoTIFF to S3 | ✅ parity |

## 2. Vector Enrichment Jobs

| SageMaker Geospatial operation | loam | Status |
|---|---|---|
| Reverse geocoding | `--op reverse-geocode` (offline, city/admin-level; CSV + GeoJSON) | ✅ parity (v1) |
| — street-level addresses | online Nominatim backend | 🔜 planned ([#31](https://github.com/scttfrdmn/loam/issues/31)) |
| Map matching (GPS → roads) | — | ❌ planned ([#30](https://github.com/scttfrdmn/loam/issues/30)) |

## 3. The executor and the viewer

| SageMaker Geospatial piece | loam | Why |
|---|---|---|
| The managed job runner (opaque instance, provision → run → export → tear down) | **Deliberately not reproduced.** loam is execution-agnostic — it emits a manifest of idempotent shards and *prints* runner commands (`loam dispatch`); the [spore.host](https://github.com/spore-host) substrate (`truffle`/`lagotto`/`spawn`) runs them. | This was SM's *worst* part — see below. |
| Studio map viewer | — | 🔜 planned ([#10](https://github.com/scttfrdmn/loam/issues/10), titiler/leafmap) |
| boto3 `sagemaker-geospatial` client shape | optional compat shim (migration on-ramp only, not the primary API) | 🔜 planned ([#9](https://github.com/scttfrdmn/loam/issues/9)) |

## Why we throw the executor away (and end up better)

SageMaker Geospatial did **not** "provision the right instance" — that was the part it was worst
at:

- On the **EOJ path** the instance was **opaque**: no type choice, no SSH, `IN_PROGRESS`-only
  status with no progress %, no scene count.
- On the **Processing path** it was actively bad at *getting* one: recurrent `CapacityError` for
  hours, a **24-hour `MaxRuntimeInSeconds` wall** that killed heavy jobs mid-run, no spot, no
  retry, no capacity-watch.

loam keeps the operations and lets them ride a substrate that is capacity-aware, spot-priced,
uncapped, and observable. On the exact axes SM was weakest, loam is **better**, not "almost as
good" — and it adds things SM never had:

| Axis | SageMaker Geospatial | loam |
|---|---|---|
| Region | `us-west-2` only | any region |
| New accounts | closed (2026-07-30) | works in a fresh account |
| Runtime cap | 24h hard kill | none (your ttl; spot-safe resume) |
| Status | opaque `IN_PROGRESS` | progress is an `ls` of S3; `loam status --detail` job ledger (bytes/time/failures) |
| Instance choice | none / lottery | your choice; `truffle` picks, `lagotto` gets one when scarce |
| Right-sizing signal | — | per-shard **compute-shape** estimates (bytes/RAM/est-runtime) in the manifest |
| Observability | none | `spawn connect` shell, live logs |

## Honest remaining gaps

On **operations**: **zonal statistics** ([#34](https://github.com/scttfrdmn/loam/issues/34)) is the
clearest unmet EOJ parity item; **map-match** ([#30](https://github.com/scttfrdmn/loam/issues/30))
is the one unmet VEJ item; arbitrary multi-band stacking is only partial.

On the **product surface**: the Studio **map viewer**
([#10](https://github.com/scttfrdmn/loam/issues/10)) has no equivalent yet, and a **boto3-compat
shim** ([#9](https://github.com/scttfrdmn/loam/issues/9)) for near-drop-in migration is optional
future work — loam's primary interface is a clean native API/CLI, not the SM EOJ shape (which had
its own warts: un-chainable ops, `ConflictException` on export-while-in-progress, ARNs everywhere).

See [DESIGN.md](DESIGN.md) for why loam is structured this way.
