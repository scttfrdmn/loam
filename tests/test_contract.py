"""Mechanical guard for loam's execution-agnostic contract.

loam describes and computes work; it NEVER provisions compute. The load-bearing rule is that
`loam/` imports nothing from the substrate (spawn/lagotto/truffle/cohort) and never launches EC2
— see docs/DESIGN.md. That's easy to violate by accident (a helpful `import spawn`, an
`aws ec2 run-instances` shell-out in `dispatch`), so this test fails the build if it happens.

We scan the *source* AST rather than importing, so lazily-imported modules (loam imports rasterio
/ pystac_client / boto3 inside functions) are caught too — a runtime import check would miss them.
"""

from __future__ import annotations

import ast
from pathlib import Path

_LOAM = Path(__file__).resolve().parent.parent / "loam"

# Substrate packages loam must never depend on — importing any couples the work to an executor.
_FORBIDDEN_ROOTS = {"spawn", "lagotto", "truffle", "cohort", "nf_spawn", "cwl_spawn"}


def _module_roots(path: Path) -> set[str]:
    """Top-level package of every import in a source file (via AST, so lazy imports count)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # skip relative (from .x import y) imports
                roots.add(node.module.split(".")[0])
    return roots


def test_loam_imports_nothing_from_the_substrate():
    """No module under loam/ may import spawn/lagotto/truffle/etc. (the agnostic contract)."""
    offenders: dict[str, set[str]] = {}
    for py in _LOAM.rglob("*.py"):
        bad = _module_roots(py) & _FORBIDDEN_ROOTS
        if bad:
            offenders[str(py.relative_to(_LOAM.parent))] = bad
    assert not offenders, (
        "loam must stay execution-agnostic — these files import the substrate: "
        f"{offenders}. loam describes/computes work; runners provision compute. See docs/DESIGN.md."
    )


def test_loam_never_launches_ec2():
    """No `ec2` / `run_instances` / `run-instances` string in loam source (dispatch PRINTS only)."""
    needles = ("run_instances", "run-instances", "terminate_instances", "create_account")
    offenders: dict[str, list[str]] = {}
    for py in _LOAM.rglob("*.py"):
        text = py.read_text()
        hits = [n for n in needles if n in text]
        if hits:
            offenders[str(py.relative_to(_LOAM.parent))] = hits
    assert not offenders, (
        f"loam must never provision/terminate compute — found launch calls: {offenders}. "
        "`loam dispatch` PRINTS runner commands; it does not execute them. See docs/DESIGN.md."
    )
