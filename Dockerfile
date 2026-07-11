# loam on arm64/Graviton — a thin layer over the aarch.science earth-observation base.
#
# Why this exists: the geospatial stack (GDAL/PROJ/GEOS/rasterio) is ~1.9x cheaper per core on
# Graviton (c7g/m8g) but often won't assemble on arm64 via pip — PyPI's arm64 wheel coverage for
# native-lib science is fragile (the fieldwork c7g failure). aarch.science already solved that:
# quay.io/aarchsci/earth-observation is a verified, signed, native-arm64 conda-forge image with
# rasterio + pystac-client + numpy (+ the xarray/stackstac EO layer). loam is a founding consumer —
# we add only the two pure-Python bits the base lacks: boto3 (S3 state) and loam-geo itself.
#
# Build NATIVELY on an arm64 host (Graviton, or Apple Silicon) for a real linux/arm64 image, no
# emulation:
#   docker build -t ghcr.io/scttfrdmn/loam:<tag> .
#   docker run --rm ghcr.io/scttfrdmn/loam:<tag> loam --version
# Then run a shard on a Graviton box (see examples/graviton_spawn.sh):
#   spawn launch --instance-type m8g.4xlarge --image ghcr.io/scttfrdmn/loam:<tag> \
#     --command 'loam run-shard --manifest s3://... -i 0'

# Pinned by digest, not :latest, so a rebuild is reproducible and a base change (which could shift
# GDAL/rasterio under loam) is a deliberate, reviewed bump — Dependabot's docker ecosystem proposes
# it. The base is cosign-signed; verify with: cosign verify quay.io/aarchsci/earth-observation@<digest>
FROM quay.io/aarchsci/earth-observation@sha256:97dfd75f252e9e9ece14457c3cc93d13f3d311afd0dded60332408e182519e38

# The base's conda-forge env is at /opt/conda (CONDA_PREFIX) but only activated via micromamba's
# entrypoint, not PATH. The image runs as the non-root `mambauser` and /opt/conda isn't user-
# writable, so pip installs into ~/.local (a --user install). Put BOTH the conda env and the user
# bin on PATH: /opt/conda/bin for python + the base geo stack, ~/.local/bin for loam's console
# script. This also makes `docker run … loam …` resolve without an activation wrapper.
ENV PATH=/opt/conda/bin:/home/mambauser/.local/bin:$PATH

# Add loam's two pure-Python deps not in the EO base — both are wheels, so no arm64 native-build
# risk. Pin loam-geo so the image tag maps to a known release; override with --build-arg for a bump.
ARG LOAM_VERSION=0.8.0
RUN pip install --no-cache-dir --user "loam-geo[vector,viz]==${LOAM_VERSION}" boto3

# Sanity: fail the build if loam can't import/run in the assembled env.
RUN loam --version && loam indices >/dev/null

# loam is execution-agnostic — no entrypoint that launches work. A runner (spawn/nf-spawn/…) sets
# the command; default to a shell so `docker run ... loam run-shard ...` works and the box is
# inspectable.
CMD ["bash"]
