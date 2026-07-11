# loam

[![CI](https://github.com/scttfrdmn/loam/actions/workflows/ci.yml/badge.svg)](https://github.com/scttfrdmn/loam/actions/workflows/ci.yml)

**An open, execution-agnostic library of geospatial operations — cloud-mask and band-math over
Sentinel-2 — that replaces the parts of Amazon SageMaker Geospatial which closed to new
customers on 2026-07-30. Runs anywhere; pairs naturally with the
[spore.host](https://github.com/spore-host) substrate (`spawn` / `truffle` / `lagotto`).**

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
pip install loam-geo            # distribution name (import name stays `loam`)
pip install 'loam-geo[vector]'  # + offline reverse-geocode (the vector-enrichment op)
```

The PyPI distribution is **`loam-geo`** (the bare `loam` name was taken by an unrelated
project); the import name is unchanged — `import loam` / the `loam` CLI.

### Develop

loam standardizes on [uv](https://docs.astral.sh/uv/). The pinned `uv.lock` is what CI
installs, so a local checkout runs the exact resolved dependency set:

```bash
uv sync --extra dev              # create .venv + install loam and dev tools from the lock
uv run ruff check loam/          # lint
uv run mypy loam/                # type-check
uv run pytest -q                 # tests (fully offline — no network/AWS)
uv run loam indices              # run the CLI
```

`uv.lock` is committed and CI uses `uv sync --locked`; regenerate it with `uv lock` after
changing dependencies in `pyproject.toml`.

The default suite is hermetic (no network). Opt-in **live** tests exercise the real Earth
Search STAC API and read Sentinel-2 COGs over `/vsicurl` (no AWS credentials needed):

```bash
LOAM_LIVE_TESTS=1 uv run pytest -m integration   # hits the network; skipped by default
```

## Use

```bash
loam indices                # the band-math catalog (NDVI, EVI, SAVI, NDWI, NDMI, NDRE, …)
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

For the full picture — the manifest mental model, every op's flags, cross-op recipes (e.g.
band-math → zonal-stats), and choosing a runner — see the **[guide](docs/GUIDE.md)**.

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

**v0.2.0 — Parity ops.** STAC search (Earth Search), cloud-mask, band-math (14 indices + custom
equations), resample/reproject, temporal composites/geomosaics, and reverse-geocode — over
`/vsicurl` COG reads, with an S3 manifest + shard/checkpoint protocol, per-shard compute-shape
estimates, an `loam status --detail` job ledger, and `spawn`/local dispatch. loam now covers the
operations half of SageMaker Geospatial's EOJ **and** VEJ.

**Coming from SageMaker Geospatial?** See **[docs/MIGRATION.md](docs/MIGRATION.md)** — a hands-on
EOJ-config → loam-CLI mapping with before/after code. And **[docs/PARITY.md](docs/PARITY.md)** for
the full parity matrix and what loam deliberately does differently: **every SageMaker Geospatial
operation is now covered** — cloud-mask, band-math, temporal composites, resample, zonal-stats,
reverse-geocode, and map-match. Remaining roadmap is product surface, not ops: an arm64/Graviton
container and a `titiler`/`leafmap` viewer.

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for dev setup (uv), tests, and the checks CI runs, and
**[docs/DESIGN.md](docs/DESIGN.md)** for why loam is execution-agnostic — the one contract to
preserve when extending it (enforced by `tests/test_contract.py`).

## License

Apache-2.0.
