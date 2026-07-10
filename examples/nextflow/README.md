# loam × Nextflow (nf-spawn)

Run a loam manifest as a Nextflow pipeline on ephemeral EC2 via
[`spore-host/nf-spawn`](https://github.com/spore-host/nf-spawn) — one task per shard, each running
the same idempotent `loam run-shard` atom.

**This is a proof of loam's execution-agnostic contract:** loam needs no code change and imports
nothing from Nextflow or nf-spawn. `loam run-shard --manifest <uri> -i N` is a plain command; a
runner only has to schedule it.

## Steps

```bash
# 1. Plan (locally, no compute) — writes the manifest and tells you the shard count.
loam plan --op band-math --indices NDVI,BSI \
    --collection sentinel-2 --aoi -7.0,19.0,-3.0,22.0 \
    --start 2023-01-01 --end 2023-12-31 --max-cloud 20 --shard-size 50 \
    --output   s3://my-bucket/h2/indices/ \
    --manifest s3://my-bucket/h2/manifest.json
# → "planned: N scenes -> M shards …"  (M is your --n below)

# Or read the shard count back from S3 at any time:
N=$(loam status --manifest s3://my-bucket/h2/manifest.json | jq .shards_total)

# 2. Fan the shards over ephemeral EC2 with nf-spawn.
nextflow run loam.nf -profile nf-spawn \
    --manifest s3://my-bucket/h2/manifest.json --n "$N"

# 3. Progress is pure S3 — safe to poll from anywhere while the pipeline runs.
loam status --manifest s3://my-bucket/h2/manifest.json --detail
```

## Requirements on the instance

The box nf-spawn launches must have:

- **`loam` on `PATH`** — `pip install loam-geo` in the AMI/container, or a `beforeScript` that
  installs it. (`loam-geo[vector]` if you use `reverse-geocode`.)
- **S3 access** to read the manifest and write shard outputs + checkpoints (an instance role with
  read/write to the bucket).
- The `manifest` URI is passed as a Nextflow param and interpolated into each task's command — no
  staging needed, since a shard reads only the manifest + its scenes' COGs over `/vsicurl`.

## Why this is spot-safe

Shards are a pool. A task that dies to a spot reclaim is simply re-run: `run-shard` writes its
checkpoint **last**, so a completed shard is a no-op on retry and an interrupted one recomputes
cleanly. No cross-shard state, no session to lose — see [`docs/DESIGN.md`](../../docs/DESIGN.md).

## Notes / friction

loam required **no changes** to run under nf-spawn — the intended result. Any friction is in the
*environment* (getting `loam` + credentials onto the box), not in loam's contract; that's
nf-spawn/AMI configuration, documented above. If you hit a case where loam itself needs a change to
run as a plain command, that's a contract leak worth
[filing](https://github.com/scttfrdmn/loam/issues).
