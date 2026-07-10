#!/usr/bin/env bash
# Example: per-zone NDVI statistics — the two-step compose (band-math → zonal-stats).
#
# SageMaker Geospatial offered zonal statistics as an EOJ op. loam splits it into two
# single-purpose ops: first compute the index COGs, then summarize them over your polygons.
set -euo pipefail

BUCKET="${LOAM_BUCKET:-my-bucket}"
NDVI_OUT="s3://${BUCKET}/zonal/ndvi/"
NDVI_MANIFEST="s3://${BUCKET}/zonal/ndvi-manifest.json"
ZONES="s3://${BUCKET}/zonal/fields.geojson"      # your polygons (e.g. farm fields), WGS84
STATS_OUT="s3://${BUCKET}/zonal/stats/"
STATS_MANIFEST="s3://${BUCKET}/zonal/stats-manifest.json"

# 1. Compute NDVI COGs over the AOI/time range (writes georeferenced GeoTIFFs).
loam plan --op band-math --indices NDVI \
    --collection sentinel-2 --aoi -7.0,19.0,-3.0,22.0 \
    --start 2023-06-01 --end 2023-06-30 --max-cloud 20 \
    --output "${NDVI_OUT}" --manifest "${NDVI_MANIFEST}"
loam run-shard --manifest "${NDVI_MANIFEST}" -i 0    # (or dispatch to a fleet)

# 2. Summarize one NDVI COG within each zone polygon → per-zone stats table.
#    Point --raster at a COG produced in step 1.
loam plan --op zonal-stats \
    --zones "${ZONES}" \
    --raster "${NDVI_OUT}shard=00000/<scene-id>__NDVI.tif" \
    --stat mean,min,max,count,p90 \
    --output "${STATS_OUT}" --manifest "${STATS_MANIFEST}"
loam run-shard --manifest "${STATS_MANIFEST}" -i 0

# 3. The output GeoJSON carries each zone's original properties + zs_mean/zs_min/... columns.
loam status --manifest "${STATS_MANIFEST}"
