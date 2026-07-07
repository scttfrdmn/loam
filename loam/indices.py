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

from dataclasses import dataclass


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
        return IndexDef(name.upper(), equation, "custom")
    return resolve(spec)


def bands_in(equation: str) -> set[str]:
    """Return the set of Sentinel-2 band names referenced in an equation string."""
    import re

    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", equation))
    return {t for t in tokens if t in _BAND_TOKENS}
