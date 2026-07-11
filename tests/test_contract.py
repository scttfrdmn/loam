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
import re
from pathlib import Path

_LOAM = Path(__file__).resolve().parent.parent / "loam"
_REPO = _LOAM.parent

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


def test_guide_op_table_matches_cli():
    """docs/GUIDE.md's op-reference table must list exactly the CLI's --op choices (anti-drift).

    The table is the one drift-prone part of the guide; this pins it to the real parser so a new
    op (or a rename/removal) forces the doc to update or the build fails.
    """
    from loam.cli import build_parser

    # Extract the plan --op choices from the argparse action.
    parser = build_parser()
    plan_action = next(
        a for sub in parser._subparsers._group_actions for name, sub_p in sub.choices.items()
        if name == "plan" for a in sub_p._actions if a.dest == "op"
    )
    cli_ops = set(plan_action.choices)

    # Ops named in the GUIDE's "Operations reference" section only — scope to that section so the
    # runner table (local/spawn/lagotto) further down doesn't get scooped up.
    guide = (_REPO / "docs" / "GUIDE.md").read_text()
    section = guide.split("## Operations reference", 1)[1].split("\n## ", 1)[0]
    # Op cells look like `| `band-math` |`; the header cell `| `--op` |` is excluded by the
    # leading-letter requirement.
    table_ops = set(re.findall(r"^\| `([a-z][a-z-]+)` \|", section, re.MULTILINE))

    assert table_ops == cli_ops, (
        f"docs/GUIDE.md op table is out of sync with the CLI. "
        f"only in CLI: {cli_ops - table_ops}; only in GUIDE: {table_ops - cli_ops}"
    )
