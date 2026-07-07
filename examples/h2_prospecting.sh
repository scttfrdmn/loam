#!/usr/bin/env bash
# Example: reproduce the fieldwork Tutorial 01 band-math step WITHOUT SageMaker Geospatial.
#
# The original ran a cloud-mask EOJ + a band-math EOJ (NDVI, BSI) over the Taoudeni Basin
# (Mali/Mauritania) full-year Sentinel-2, then exported ~855 GiB to S3 — a ~8h managed job you
# couldn't see into, in us-west-2 only, now closed to new accounts. loam does the same math,
# any region, on spawn spot boxes you control.
set -euo pipefail

BUCKET="${BUCKAI_BUCKET:-my-bucket}"
MANIFEST="s3://${BUCKET}/h2-prospecting/manifest.json"
OUTPUT="s3://${BUCKET}/h2-prospecting/indices/"

# 1. Plan: search the year, shard 50 scenes/unit, attach NDVI+BSI band-math.
loam plan --op band-math --indices NDVI,BSI \
    --collection sentinel-2 \
    --aoi -7.0,19.0,-3.0,22.0 \
    --start 2023-01-01 --end 2023-12-31 \
    --max-cloud 20 --shard-size 50 \
    --output   "${OUTPUT}" \
    --manifest "${MANIFEST}"

# 2. See how you'd fan it out (prints spawn commands — does not launch).
loam dispatch --manifest "${MANIFEST}" --runner spawn --instance m8g.4xlarge

# 3. Run locally to smoke-test one shard before committing a fleet.
loam run-shard --manifest "${MANIFEST}" -i 0

# 4. Watch progress (pure S3 ledger; safe to poll).
loam status --manifest "${MANIFEST}"
