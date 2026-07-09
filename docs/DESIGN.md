# loam — design

This is the in-repo design of record. It exists so you can change loam **without breaking the
one property that makes it composable**: execution-agnosticism. Read it before touching
`loam/`. (Derived from the founding write-up, `fieldwork/docs/loam-design.md`.)

## What loam is

loam is an **open, execution-agnostic library of geospatial operations** — cloud-mask and
band-math over Sentinel-2 — plus a manifest/shard CLI. It replaces the *operations* half of
Amazon SageMaker Geospatial's Earth Observation Jobs (EOJ), which closed to new customers on
2026-07-30.

loam is a **third kind** of spore.host repo:
- **Tools** (`spawn`, `truffle`, `lagotto`, `cohort`) — the substrate; verbs against EC2.
- **`-spawn` adapters** (`nf-spawn`, `cwl-spawn`, `miniwdl-spawn`) — bridge an existing engine's
  work onto the substrate.
- **loam** — neither. It is **the work itself**: the first *native workload* the substrate
  carries. spawn/nf-spawn are *how* it runs; loam is *what* runs.

## Why SageMaker Geospatial was replaceable

It fused two things that never needed to be one:

1. **A catalog of operations** — cloud-mask, band-math, composite, resample, geomosaic,
   vector-enrichment. Pure domain content. loam keeps this. The band-math equations are literally
   seven one-line formulas (`loam/indices.py`).
2. **A managed executor** — provision a box, run the op, export to S3, tear down.

Decomposed against today's OSS world, there is **no irreplaceable core**: Raster Data
Collections ≈ a STAC catalog (Earth Search indexes ESA's free Sentinel-2, which AWS never
owned); cloud-mask ≈ apply the SCL band; band-math ≈ numpy over named bands; composites ≈
`stackstac`/`xarray`; export ≈ `aws s3 cp`. The one genuinely hard part — get the right box and
run it — is what the executor was **worst** at.

### The correction that sharpens the thesis

SageMaker Geospatial did **not** "provision the right instance." That was its weakest part:
- On the EOJ path the instance was **opaque** — no type choice, no SSH, `IN_PROGRESS`-only
  status, no progress %, no scene count.
- On the Processing path (`ml.g5.*`) it was actively bad at *getting* one: recurrent
  `CapacityError` for hours, a **24h `MaxRuntimeInSeconds` wall** that killed heavy jobs mid-run,
  no spot, no retry.

So the executor half is a **liability**, not a convenience. "Pick the right instance" is a solved
problem in this org — but by the substrate, not a monolith: `truffle` (pick), `lagotto` (get one
when scarce), `spawn` (run it, spot-safe, uncapped, observable). loam keeps the operations,
throws away the executor, and lets the ops ride that substrate — better on the exact axis the
managed service was worst.

## The load-bearing decision: EXECUTION-AGNOSTIC

**loam describes and computes work. It never provisions, terminates, or knows what EC2 is.** The
dependency arrow points one way — loam imports nothing from spawn/lagotto; the runners import
nothing from loam:

```
loam        (pure content: STAC search, cloud_mask, band_math)
  ▲   ▲   ▲
  │   │   └── nf-spawn step   (Nextflow process wraps a loam op → ephemeral EC2)
  │   └────── cwl-spawn step  (CWL tool wraps a loam op)
  └────────── spawn launch    (plain: `loam run-shard ...` as the box's command)
```

The runners already know how to run an arbitrary command on a box. loam only has to *be* a
well-behaved command. That's the whole contract — one command satisfying an interface all runners
already speak, instead of three bespoke integrations.

### The three properties agnosticism forces (the actual spec)

1. **Work is data, not a running job.** `loam plan` searches a STAC catalog and writes a
   **manifest** — scenes, grouped into shards, plus an operation. No pixels are read at plan time.
   (`loam/plan.py`, `loam/manifest.py`, `loam/catalog.py`.)
2. **State lives in S3, never in loam.** No job ARNs, no in-memory status, no control plane. A
   shard is **done** iff its checkpoint object exists in S3. Progress is an `ls` — which also
   fixes the EOJ opaque-status wart for free. (`loam/state.py`.)
3. **A shard is one idempotent command.** `loam run-shard --manifest <uri> -i N` is re-runnable,
   side-effect-free on retry (spot-safe: re-running a reclaimed shard is a no-op if its checkpoint
   exists), and ignorant of its neighbors. It writes the checkpoint **last**, after outputs are
   durable (delete-after-durable), so "checkpoint exists" strictly implies "output is safe".
   (`loam/run.py`.)

Given those three, loam composes with **every** spore.host runner for free — today and with
runners that don't exist yet.

### `dispatch` is the seam — it PRINTS, never launches

`loam dispatch` emits the `spawn launch … -i N` lines (one box per shard) or a laptop `for` loop
and **stops**. You, or an outer orchestrator, run them. This is what keeps loam agnostic: it shows
you the command; it does not call spawn. (`loam/cli.py::_cmd_dispatch`.)

## What this means for contributors (the rules)

These are enforced mechanically by `tests/test_contract.py` — but understand *why*:

- **`loam/` imports nothing from `spawn`/`lagotto`/`truffle`/`cohort`/adapters.** If a task seems
  to need it, you're putting executor logic in the wrong repo.
- **loam never calls `ec2 run-instances` / terminate / `organizations create-account`**, shells
  out to provision, or launches anything. `dispatch` prints.
- **No control plane / no job state in memory.** Completion is derived from S3 objects, full stop.
- **`plan` reads no pixels.** Only `run-shard` touches COGs (via `/vsicurl`).

## Scope tiers

- **Tier 1 — MVP (shipped, v0.1):** STAC search (Earth Search) + cloud-mask + band-math +
  `/vsicurl` COG read + georeferenced GeoTIFF write + manifest/run-shard. Runs in ANY region and
  a FRESH account → kills both the us-west-2 lock and the 2026-07-30 cliff.
- **Tier 2 — parity:** temporal composites/geomosaics (`stackstac`+`xarray`), resample, more
  indices, VEJ-style reverse-geocode; per-shard **compute-shape** estimation for truffle (#17).
- **Tier 3 — product:** `titiler`/`leafmap` viewer, declarative multi-op pipeline spec, arm64
  packaging for Graviton prep.

See [PARITY.md](PARITY.md) for how loam maps against SageMaker Geospatial operation-by-operation.
