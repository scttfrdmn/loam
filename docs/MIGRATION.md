# Migrating from SageMaker Geospatial to loam

Amazon SageMaker Geospatial **closed to new customers on 2026-07-30**. Existing accounts can still
run Earth Observation Jobs (EOJ) for now, but a fresh account can never onboard, and it only ever
ran in `us-west-2`. This guide gets you from an EOJ workflow to the equivalent loam commands.

For the full capability map (what's at parity, what's a gap), see [PARITY.md](PARITY.md).

## The mental-model shift

SageMaker Geospatial gave you **one opaque managed job**: submit an EOJ config, get an ARN, poll
`IN_PROGRESS` (no percentage, no scene count) until it exports to S3.

loam splits that into three plain steps you can see into:

1. **`loam plan`** — search a STAC catalog and write a **manifest** (scenes + shards + the op). No
   compute; no pixels read.
2. **`loam dispatch`** — *print* the commands to run each shard on a runner you choose
   (`local` / `spawn` / `lagotto` / Nextflow). loam never provisions.
3. **`loam run-shard -i N`** — the idempotent atom a runner schedules. Progress is **`loam status`**
   — an `ls` of S3, with real per-shard numbers.

You pick the executor; loam is just the work. That's why it runs in any region, in a fresh
account, with no 24-hour runtime cap, spot-priced and observable — the axes SM Geospatial was
weakest on.

## EOJ config → loam CLI

An EOJ `InputConfig` + `JobConfig` maps almost field-for-field onto `loam plan`:

| SageMaker Geospatial EOJ config | loam `plan` flag |
|---|---|
| `RasterDataCollectionQuery.RasterDataCollectionArn` (e.g. Sentinel-2 L2A) | `--collection sentinel-2` |
| `AreaOfInterest` → `AreaOfInterestGeometry` (polygon/bbox) | `--aoi W,S,E,N` (WGS84 bbox) |
| `TimeRangeFilter.StartTime` / `EndTime` | `--start YYYY-MM-DD` / `--end YYYY-MM-DD` |
| `PropertyFilters` → `eo:cloud_cover` upper bound | `--max-cloud 20` |
| `JobConfig` = `BandMathConfig` (index equations) | `--op band-math --indices NDVI,BSI,…` |
| `JobConfig` = `CloudMaskingConfig` | `--op cloud-mask` |
| `JobConfig` = `ResamplingConfig` | `--op resample --dst-crs … [--dst-res …]` |
| `JobConfig` = temporal statistics / mosaic | `--op temporal-composite --reducer median` |
| `OutputConfig.S3Data` (export location) | `--output s3://bucket/prefix/` |
| (the manifest itself — new; loam's work-as-data) | `--manifest s3://bucket/prefix/manifest.json` |

loam's band-math index names match the ones the EOJ `BandMath` operation accepted (NDVI, BSI, EVI,
MNDWI, NDBI, NBR, NDSI, …) — plus you can pass a custom `NAME=equation` without a code change. Run
`loam indices` to see the catalog.

## Before / after

**SageMaker Geospatial (boto3):**

```python
import boto3
geo = boto3.client("sagemaker-geospatial", region_name="us-west-2")

resp = geo.start_earth_observation_job(
    Name="ndvi-bsi-2023",
    InputConfig={
        "RasterDataCollectionQuery": {
            "RasterDataCollectionArn": SENTINEL2_L2A_ARN,
            "AreaOfInterest": {"AreaOfInterestGeometry": {
                "PolygonGeometry": {"Coordinates": [[[-7,19],[-3,19],[-3,22],[-7,22],[-7,19]]]}}},
            "TimeRangeFilter": {"StartTime": "2023-01-01T00:00:00Z",
                                "EndTime": "2023-12-31T23:59:59Z"},
            "PropertyFilters": {"Properties": [
                {"Property": {"EoCloudCover": {"UpperBound": 20}}}]},
        }},
    JobConfig={"BandMathConfig": {"CustomIndices": {"Operations": [
        {"Name": "NDVI", "Equation": "(nir - red)/(nir + red)"},
        {"Name": "BSI",  "Equation": "(swir16 - nir)/(swir16 + nir)"}]}}},
    OutputConfig={"S3Data": {"S3Uri": "s3://my-bucket/out/"}},
)
arn = resp["Arn"]
# then poll geo.get_earth_observation_job(Arn=arn)["Status"] until COMPLETED — opaque.
```

**loam (any region, fresh account):**

```bash
loam plan --op band-math --indices NDVI,BSI \
    --collection sentinel-2 --aoi -7,19,-3,22 \
    --start 2023-01-01 --end 2023-12-31 --max-cloud 20 --shard-size 50 \
    --output   s3://my-bucket/out/ \
    --manifest s3://my-bucket/out/manifest.json

loam dispatch --manifest s3://my-bucket/out/manifest.json --runner spawn --instance m8g.4xlarge
# → run the printed commands (or hand the manifest to nf-spawn / lagotto / a laptop loop)

loam status --manifest s3://my-bucket/out/manifest.json --detail
```

## Status & export, without the warts

| SageMaker Geospatial | loam |
|---|---|
| `get_earth_observation_job` → `IN_PROGRESS` (no %, no counts) | `loam status --detail` — shards done/remaining + a job ledger (bytes, seconds, per-scene failures), all derived from S3 |
| `ExportEarthObservationJob` as a separate step; `ConflictException` if you export while running | no export step — each shard writes its (COG) GeoTIFF to `--output` as it finishes; partial results are usable immediately |
| Band-math couldn't chain from a cloud-mask job | one `band-math` op cloud-masks (SCL) per scene inline |
| ARNs everywhere; a job you can't SSH into | a manifest you can read; run a shard on any box (`spawn connect` for a shell) |

## What isn't 1:1

- **You choose the runner.** There's no managed executor — that's the point (it was SM's worst
  part). Pick `spawn` (one box/shard), `lagotto` (capacity-watch fleet), Nextflow via
  [nf-spawn](../examples/nextflow/), or a `local` loop. `loam dispatch` prints each.
- **Coverage gaps** (see [PARITY.md](PARITY.md)): zonal statistics and map-match aren't built yet;
  the Studio map viewer has no equivalent.
- **Region & scale**: no `us-west-2` lock, no 24-hour job wall, spot-safe resume — so long-running
  or large jobs that SM would kill mid-run simply complete.

## Want SM-shaped calls?

A thin optional **boto3-compat shim** (SageMaker-EOJ-shaped functions that translate to a loam
manifest) is planned as a migration on-ramp — see
[#9](https://github.com/scttfrdmn/loam/issues/9). It's a convenience for porting existing code,
not the primary interface: new work should use the native CLI/API above, which is cleaner and
observable by design.
