# loam

**Geospatial operations for the [spore.host](https://github.com/spore-host) substrate — an
open, execution-agnostic replacement for the parts of Amazon SageMaker Geospatial that closed
to new customers on 2026-07-30.**

SageMaker Geospatial welded two things together: a catalog of **operations** (cloud-mask,
band-math, composites over Sentinel-2) and an opaque, capacity-starved, 24-hour-capped
**executor** you couldn't see into. loam keeps the operations, throws away the executor, and
lets the operations ride a substrate that's actually good at compute — `truffle` (find the
right instance), `lagotto` (get one when capacity is scarce), `spawn` (run it, spot-priced,
uncapped, observable). It runs in **any region** and a **fresh account** — no onboarding, no
Oregon lock, no cutoff.

```
loam        pure geospatial content: STAC search, cloud_mask, band_math
  ▲   ▲   ▲
  │   │   └── nf-spawn step   (a loam op as a Nextflow process → ephemeral EC2)
  │   └────── cwl-spawn step  (a loam op as a CWL tool)
  └────────── spawn launch    (loam run-shard as the box's command)
```

## The idea: loam is *the work*, not a runner

loam is not a tool like `spawn` (substrate) and not a `-spawn` adapter like `nf-spawn` (a
bridge for an existing engine). It's a **third kind** of spore.host repo: the *workload* the
substrate carries — the first one that's native rather than borrowed from Nextflow/CWL/WDL.

That means loam **describes and computes work; it never provisions compute.** This is enforced
by three properties — the whole contract:

1. **Work is data.** `loam plan` searches a STAC catalog and writes a **manifest** (scenes,
   grouped into shards, plus an operation). No pixels are read at plan time.
2. **State lives in S3.** A shard is *done* when its output exists in the object store — there
   is no job, no ARN, no session. `loam status` is an `ls`. (This also fixes SageMaker's worst
   wart: an opaque `IN_PROGRESS` with no progress percentage.)
3. **A shard is one idempotent command.** `loam run-shard --manifest <uri> -i N` is the atom
   every runner schedules — spot-safe (re-running a reclaimed shard is a no-op if its
   checkpoint exists), self-contained, ignorant of its neighbors.

Given those, loam composes with **every** spore.host runner for free — today and with runners
that don't exist yet — because a runner only has to run one well-behaved command.

## Install

```bash
pip install loam            # once published
# or from source:
pip install -e '.[dev]'
```

## Use

```bash
loam indices                # the band-math catalog (NDVI, BSI, EVI, MNDWI, NDBI, NBR, NDSI)
loam collections            # known STAC collections

# 1. PLAN — search + shard into a manifest (no compute)
loam plan --op band-math --indices NDVI,BSI \
    --collection sentinel-2 --aoi -7.0,19.0,-3.0,22.0 \
    --start 2023-01-01 --end 2023-12-31 --max-cloud 10 \
    --shard-size 50 \
    --output   s3://my-bucket/h2/indices/ \
    --manifest s3://my-bucket/h2/manifest.json

# 2. DISPATCH — print the runner commands (loam does NOT run them; a runner does)
loam dispatch --manifest s3://my-bucket/h2/manifest.json --runner spawn --instance m8g.4xlarge

# 3. RUN — a runner (or you) executes one shard at a time; idempotent + spot-safe
loam run-shard --manifest s3://my-bucket/h2/manifest.json -i 0

# 4. STATUS — progress, derived purely from S3
loam status --manifest s3://my-bucket/h2/manifest.json
```

`loam dispatch` is the seam that keeps loam agnostic: it *shows you* the `spawn launch … -i N`
lines (one box per shard — scale-out beats one big box) or a laptop `for` loop, and stops. You,
or an outer orchestrator, run them.

### Custom indices without a code change

```bash
loam plan --op band-math --indices 'NDWI=(green - nir) / (green + nir)' ...
```

## As a library

```python
from loam import plan, run

m = plan.build_manifest(
    op="band-math", collection="sentinel-2",
    aoi=[-7.0, 19.0, -3.0, 22.0], start="2023-01-01", end="2023-12-31",
    indices=["NDVI", "BSI"], max_cloud=10, shard_size=50,
    output_uri="s3://my-bucket/h2/indices/",
)
plan.write_manifest(m, "s3://my-bucket/h2/manifest.json")
# a runner then calls, per shard:
run.run_shard("s3://my-bucket/h2/manifest.json", index=0)
```

## Status

**v0.1.0 — Tier-1 MVP.** STAC search (Earth Search), cloud-mask, band-math, `/vsicurl` COG
reads, S3 manifest + shard/checkpoint protocol, `spawn`/local dispatch. Enough to replace a
SageMaker Geospatial EOJ chain (the founding use case: natural-hydrogen "fairy circle"
prospecting on Sentinel-2). Roadmap: temporal composites & geomosaics (`stackstac`+`xarray`),
resampling, vector enrichment, an arm64 container for cheap Graviton prep, a `titiler`/`leafmap`
viewer.

## License

Apache-2.0.
