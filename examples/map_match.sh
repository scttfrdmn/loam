#!/usr/bin/env bash
# Example: snap noisy GPS traces to roads — the map-match VEJ op.
#
# SageMaker Geospatial offered map-matching as a Vector Enrichment Job. loam does it over a routing
# service you point at (Valhalla by default, or OSRM), one matched LineString per trace.
set -euo pipefail

BUCKET="${LOAM_BUCKET:-my-bucket}"
TRACES="s3://${BUCKET}/vej/traces.csv"        # CSV: a trace_id column + lat/lon per point (ordered)
OUT="s3://${BUCKET}/vej/matched/"
MANIFEST="s3://${BUCKET}/vej/manifest.json"

# 1. Plan: group points into traces by trace_id, bin whole traces into shards. No STAC/compute.
#    Default backend is Valhalla; set VALHALLA_URL (or --backend osrm + OSRM_URL) for self-host.
loam plan --op map-match \
    --input "${TRACES}" --trace-field trace_id \
    --backend valhalla --traces-per-shard 200 --max-trace-points 100 \
    --output "${OUT}" --manifest "${MANIFEST}"

# 2. Match each shard's traces (or dispatch to a fleet). Output is GeoJSON: one LineString per
#    trace, with match_way_ids / match_confidence / match_names in the feature properties.
loam run-shard --manifest "${MANIFEST}" -i 0

# 3. Progress + any traces skipped (over-long / unmatched are recorded, not fatal).
loam status --manifest "${MANIFEST}" --detail
