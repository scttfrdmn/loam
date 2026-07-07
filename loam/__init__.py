"""loam — geospatial operations for the spore.host substrate.

An execution-agnostic library + CLI that provides the operations SageMaker Geospatial's Earth
Observation Jobs offered (cloud-mask, band-math over Sentinel-2), decoupled from any executor.
loam describes and computes work; it never provisions compute. A runner (spawn, nf-spawn,
cwl-spawn, or a laptop loop) schedules ``loam run-shard`` over a manifest.

See docs/DESIGN or the founding write-up (fieldwork/docs/loam-design.md).
"""

from __future__ import annotations

__version__ = "0.1.0"

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
