"""Spectral index definitions — the band-math catalog.

Ported verbatim from the reference implementation
(fieldwork/spawn-sagemaker/internal/sagemaker/eoj.go, ``IndexAlias``), which itself
mirrored the equations SageMaker Geospatial's BandMath EOJ accepted. These are the whole
"feature" — per-pixel arithmetic over named Sentinel-2 bands. There is no magic here, which
is precisely the point: the operations were never the hard part.

Band names follow the Sentinel-2 L2A convention used by Earth Search / the EOJ API:
    coastal, blue, green, red, rededge1-3, nir, nir08, nir09, swir16, swir22, aot, wvp, scl
Use ``swir16`` (1610nm) or ``swir22`` (2190nm) — the generic alias ``swir`` is invalid.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class IndexDef:
    """A named spectral index: its equation and a human description."""

    name: str
    equation: str
    description: str


# The catalog. Keyed by the short uppercase name a user passes to ``--indices``.
INDICES: dict[str, IndexDef] = {
    "NDVI": IndexDef("NDVI", "(nir - red) / (nir + red)", "Normalized Difference Vegetation Index"),
    "BSI": IndexDef("BSI", "(swir16 - nir) / (swir16 + nir)", "Bare Soil Index"),
    "EVI": IndexDef(
        "EVI",
        "2.5 * (nir - red) / (nir + 6.0*red - 7.5*blue + 1.0)",
        "Enhanced Vegetation Index",
    ),
    "MNDWI": IndexDef(
        "MNDWI",
        "(green - swir16) / (green + swir16)",
        "Modified Normalized Difference Water Index",
    ),
    "NDBI": IndexDef("NDBI", "(swir16 - nir) / (swir16 + nir)", "Normalized Difference Built-up Index"),
    "NBR": IndexDef("NBR", "(nir - swir22) / (nir + swir22)", "Normalized Burn Ratio"),
    "NDSI": IndexDef("NDSI", "(green - swir16) / (green + swir16)", "Normalized Difference Snow Index"),
    # Additional well-known indices (curated so users get them by name; equation sources noted).
    # NDWI: McFeeters 1996, open-water delineation (distinct from Gao's NDMI below).
    "NDWI": IndexDef("NDWI", "(green - nir) / (green + nir)", "Normalized Difference Water Index (McFeeters)"),
    # SAVI: Huete 1988, soil-adjusted (L=0.5 canopy constant baked in).
    "SAVI": IndexDef("SAVI", "1.5 * (nir - red) / (nir + red + 0.5)", "Soil-Adjusted Vegetation Index (L=0.5)"),
    # GNDVI: Gitelson 1996, green-based, sensitive to chlorophyll.
    "GNDVI": IndexDef("GNDVI", "(nir - green) / (nir + green)", "Green Normalized Difference Vegetation Index"),
    # NDMI: Gao 1996, vegetation/canopy moisture (nir vs swir16).
    "NDMI": IndexDef("NDMI", "(nir - swir16) / (nir + swir16)", "Normalized Difference Moisture Index"),
    # NDRE: Barnes 2000, red-edge chlorophyll (needs the rededge1 band).
    "NDRE": IndexDef("NDRE", "(nir - rededge1) / (nir + rededge1)", "Normalized Difference Red-Edge Index"),
    # ARVI: Kaufman & Tanre 1992, atmospherically resistant (blue corrects red).
    "ARVI": IndexDef(
        "ARVI",
        "(nir - (2.0*red - blue)) / (nir + (2.0*red - blue))",
        "Atmospherically Resistant Vegetation Index",
    ),
}

# Bands that appear in any equation above — the set a scene must resolve hrefs for.
# Parsed once so ``ops`` knows which assets to read without re-deriving per shard.
_BAND_TOKENS = {
    "coastal", "blue", "green", "red", "rededge1", "rededge2", "rededge3",
    "nir", "nir08", "nir09", "swir16", "swir22", "aot", "wvp", "scl",
}


def resolve(name: str) -> IndexDef:
    """Look up an index by name (case-insensitive). Raises KeyError with a helpful list."""
    key = name.strip().upper()
    if key not in INDICES:
        raise KeyError(
            f"unknown index {name!r}; known: {', '.join(sorted(INDICES))}. "
            "Pass a custom one as NAME=equation (e.g. NDWI=(green-nir)/(green+nir))."
        )
    return INDICES[key]


def parse_spec(spec: str) -> IndexDef:
    """Parse an index spec: either a known name (``NDVI``) or ``NAME=equation``.

    The ``NAME=equation`` form lets a user supply an index not in the catalog without a
    code change — mirrors the reference impl's ``resolveBandMathOps`` custom-equation path.
    """
    if "=" in spec:
        name, _, equation = spec.partition("=")
        name, equation = name.strip(), equation.strip()
        if not name or not equation:
            raise ValueError(f"malformed index spec {spec!r}; expected NAME=equation")
        validate_equation(equation)  # reject a hostile custom equation at plan time, not run time
        return IndexDef(name.upper(), equation, "custom")
    return resolve(spec)


def bands_in(equation: str) -> set[str]:
    """Return the set of Sentinel-2 band names referenced in an equation string."""
    import re

    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", equation))
    return {t for t in tokens if t in _BAND_TOKENS}


# ── Safe equation evaluator ──────────────────────────────────────────────────
# band_math must evaluate an equation string over numpy band arrays. The 7 catalog equations
# are constants we authored, but a user's ``NAME=equation`` spec — or an equation string in a
# manifest of unknown origin — reaches the same code path. Python's ``eval`` (even with empty
# ``__builtins__``) is not a hard sandbox: dunder traversal escapes are well documented. So we
# parse the equation to an AST and walk a strict allowlist, executing ONLY numeric literals,
# band-name lookups, and ``+ - * / **`` / unary ±. Anything else raises ValueError and never
# runs. This module stays numpy-free: operators dispatch to whatever objects ``env`` holds, so
# the same numpy ops run under band_math's ``np.errstate`` context (identical NaN/inf handling).

_BINOPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_node(node: ast.AST) -> None:
    """Recursively assert a node is within the allowlist. Raises ValueError otherwise.

    Purely structural — does NO arithmetic (so validating ``(nir-red)/(nir+red)`` can't raise a
    spurious ZeroDivisionError). ``safe_eval`` re-walks the same shape to actually compute.
    """
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise ValueError(f"operator {type(node.op).__name__} not allowed in an equation")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise ValueError(f"unary {type(node.op).__name__} not allowed in an equation")
        _check_node(node.operand)
    elif isinstance(node, ast.Constant):
        # Exact type check, not isinstance: isinstance(True, int) is True, and we must exclude
        # bool/complex/str/bytes/None. Only real numeric literals are legal.
        if type(node.value) not in (int, float):
            raise ValueError(f"only numeric literals allowed, got {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id not in _BAND_TOKENS:
            raise ValueError(
                f"unknown band {node.id!r}; known: {', '.join(sorted(_BAND_TOKENS))}"
            )
    else:
        raise ValueError(f"{type(node).__name__} is not allowed in an equation")


def _parse(equation: str) -> ast.Expression:
    """Parse an equation to an AST, turning SyntaxError into ValueError (the caller's contract)."""
    try:
        return ast.parse(equation, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid equation {equation!r}: {e}") from e


def validate_equation(equation: str) -> None:
    """Raise ValueError unless ``equation`` uses only the allowed grammar (no evaluation)."""
    _check_node(_parse(equation))


def safe_eval(equation: str, env: dict[str, Any]) -> Any:
    """Evaluate ``equation`` over ``env`` (band name -> array) via the allowlist. No ``eval``.

    Self-validates every node before dispatching, so it is safe even when reached without
    ``parse_spec`` (e.g. an ``IndexDef`` built directly, or a catalog equation).
    """
    tree = _parse(equation)
    _check_node(tree)  # defense-in-depth: never dispatch a node the allowlist would reject

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return _UNARYOPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Constant):
            # _check_node already guaranteed a real numeric literal; float() neutralizes
            # giant-int ** DoS and is a no-op for float literals.
            return float(node.value)  # type: ignore[arg-type]
        if isinstance(node, ast.Name):
            return env[node.id]
        raise ValueError(f"{type(node).__name__} is not allowed in an equation")  # unreachable

    return _eval(tree)
