# Contributing to loam

Thanks for helping! loam is small on purpose. Before you change code, read
**[docs/DESIGN.md](docs/DESIGN.md)** — loam is *execution-agnostic*, and that property is easy to
break by accident. The one rule below is load-bearing.

## The one rule

**loam describes and computes work; it never provisions compute.** Concretely:

- `loam/` imports **nothing** from the substrate — `spawn`, `lagotto`, `truffle`, `cohort`, or the
  `-spawn` adapters.
- loam **never launches or terminates anything** (no `ec2 run-instances`, no shell-out to
  provision). `loam dispatch` **prints** runner commands; it does not execute them.
- **State lives in S3, not in loam.** A shard is done iff its checkpoint object exists — no job
  ARNs, no in-memory status, no control plane.
- `loam plan` reads **no pixels**; only `loam run-shard` touches COGs.

These are enforced by `tests/test_contract.py`, which fails the build on a violation. If a change
seems to need executor logic, it belongs in a runner (`spawn`/`nf-spawn`/…), not here.

## Dev setup

loam standardizes on [uv](https://docs.astral.sh/uv/). The committed `uv.lock` is what CI
installs, so a local checkout runs the exact resolved dependency set:

```bash
uv sync --extra dev          # create .venv and install loam + dev tools from the lock
```

(No uv? `pipx install uv` or see the uv docs. A plain `pip install -e '.[dev]'` also works but
won't match the lock.)

## Checks — run all three before opening a PR

```bash
uv run ruff check loam/      # lint
uv run mypy loam/            # type-check (targets 3.12; see pyproject)
uv run pytest -q             # tests — fully offline, no network or AWS
```

CI runs exactly these across Python 3.10 / 3.11 / 3.12. The default test suite is **hermetic**:
`loam/state.py` treats non-`s3://` URIs as local files, so shards, checkpoints, and status all run
against a `tmp_path` with zero cloud.

### Live (integration) tests

Opt-in tests exercise the real Earth Search STAC API and read Sentinel-2 COGs over `/vsicurl`
(no AWS credentials needed). They are skipped by default:

```bash
LOAM_LIVE_TESTS=1 uv run pytest -m integration
```

If you change `loam/catalog.py` (search, asset→band mapping) or `loam/ops.py` (COG reads), run
these — they cover the one seam the offline suite can't.

## Changing dependencies

Edit `pyproject.toml`, then regenerate the lock and commit it (CI uses `uv sync --locked` and
fails on a stale lock):

```bash
uv lock
```

## Conventions

- Match the surrounding style: dense, purposeful docstrings that explain *why*, not just *what*.
- Add a `CHANGELOG.md` entry under `[Unreleased]` for anything user-visible.
- Keep `loam/` numpy/rasterio-focused and substrate-free; put executor concerns in a runner.

## Releasing (maintainers)

Releases are automated by `.github/workflows/release.yml`. Bump `version` in `pyproject.toml`,
promote `CHANGELOG.md`'s `[Unreleased]` to a dated `[x.y.z]` section, then:

```bash
git tag vX.Y.Z && git push origin vX.Y.Z
```

The workflow verifies the tag matches the version, builds with `uv build`, smoke-tests the wheel,
publishes to PyPI via **Trusted Publishing (OIDC)** as `loam-geo` (import name stays `loam`), and
cuts a GitHub Release.
