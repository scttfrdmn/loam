#!/usr/bin/env bash
# Example: run loam shards on cheap arm64/Graviton spot boxes using the loam container image.
#
# The geospatial stack (GDAL/rasterio) is ~1.9x cheaper per core on Graviton (c7g/m8g), but often
# won't pip-install on arm64. The loam image (built FROM aarch.science's verified conda-forge
# earth-observation base) sidesteps that — the stack is already assembled. See the Dockerfile.
set -euo pipefail

IMAGE="${LOAM_IMAGE:-ghcr.io/scttfrdmn/loam:latest}"   # or a pinned :vX.Y.Z tag
MANIFEST="s3://my-bucket/h2/manifest.json"

# 1. (once) build + push the arm64 image — do this on an arm64 host (Graviton or Apple Silicon)
#    so conda-forge resolves native aarch64 with no emulation:
#      docker build -t "$IMAGE" .
#      docker push "$IMAGE"

# 2. See how you'd fan the shards to Graviton spot boxes with the image (prints; does not launch):
loam dispatch --manifest "$MANIFEST" --runner spawn --instance m8g.4xlarge

# 3. Each printed spawn command runs a shard from the image on a Graviton box, e.g.:
#      spawn launch loam-00000 --instance-type m8g.4xlarge --spot \
#        --image "$IMAGE" --on-complete terminate --iam-policy s3:ReadWrite \
#        --command 'loam run-shard --manifest '"$MANIFEST"' -i 0'
#
# loam stays execution-agnostic: the image just carries the assembled stack; spawn provides the box.
