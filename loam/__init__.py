"""loam — geospatial operations for the spore.host substrate.

An execution-agnostic library + CLI that provides the operations SageMaker Geospatial's Earth
Observation Jobs offered (cloud-mask, band-math over Sentinel-2), decoupled from any executor.
loam describes and computes work; it never provisions compute. A runner (spawn, nf-spawn,
cwl-spawn, or a laptop loop) schedules ``loam run-shard`` over a manifest.

See docs/DESIGN or the founding write-up (fieldwork/docs/loam-design.md).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Single source of truth is pyproject.toml's version — read it from the installed distribution
# metadata so the CLI/API can never drift from what was actually built/published. The fallback
# covers running from a source tree that was never installed (e.g. some CI/test invocations).
try:
    __version__ = _pkg_version("loam-geo")
except PackageNotFoundError:  # pragma: no cover - source tree without an installed dist
    __version__ = "0+unknown"

from .indices import INDICES, IndexDef, parse_spec, resolve  # noqa: E402
from .manifest import Manifest, Scene, Shard  # noqa: E402

__all__ = [
    "__version__",
    "INDICES",
    "IndexDef",
    "parse_spec",
    "resolve",
    "Manifest",
    "Scene",
    "Shard",
]
