"""SageMaker Geospatial EOJ-shaped compatibility shim — a MIGRATION ON-RAMP, not the main API.

If you have existing code calling ``boto3.client("sagemaker-geospatial").start_earth_observation_job``,
this lets it run on loam with minimal edits: point it at ``loam.compat.sagemaker`` instead. It maps
the subset of the EOJ config loam covers (RasterDataCollectionQuery + AOI + time + cloud-cover
filter, and a BandMath or CloudMasking JobConfig) to a loam ``Manifest``, and returns a fake "job"
whose status reads from loam's S3 ledger — so the opaque ``IN_PROGRESS`` you used to poll is now a
real progress count underneath.

**This is a convenience for porting, NOT the recommended interface.** New code should use loam's
native, cleaner API — ``loam.plan.build_manifest`` → ``loam.run.run_shard`` → ``loam.plan.status``,
or the ``loam`` CLI. The SM EOJ shape is one of the weaker parts loam deliberately left behind
(opaque status, band-math can't chain from cloud-mask, ARNs everywhere). See docs/MIGRATION.md.

Note: the shim builds the manifest (the "plan" half of an EOJ). It does NOT launch compute —
loam is execution-agnostic; you still dispatch shards to a runner (spawn/lagotto/local/Nextflow).
``start_earth_observation_job`` here means "plan the work and write the manifest", and the job
object's status reflects shards a runner completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import plan
from ..manifest import Manifest


def _aoi_from_geometry(geom: dict[str, Any]) -> list[float]:
    """Extract a [W, S, E, N] bbox from an EOJ AreaOfInterestGeometry (Polygon or MultiPolygon)."""
    if "PolygonGeometry" in geom:
        rings = geom["PolygonGeometry"]["Coordinates"]  # [[ [lon,lat], ... ]]
        coords = rings[0]
    elif "MultiPolygonGeometry" in geom:
        coords = geom["MultiPolygonGeometry"]["Coordinates"][0][0]  # first ring of first polygon
    else:
        raise ValueError("AreaOfInterestGeometry must be a PolygonGeometry or MultiPolygonGeometry")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _build_manifest_from_eoj(
    input_config: dict, job_config: dict, output_config: dict, manifest_uri: str,
) -> Manifest:
    """Translate an EOJ InputConfig + JobConfig + OutputConfig into a loam Manifest (no compute)."""
    query = input_config["RasterDataCollectionQuery"]
    aoi = _aoi_from_geometry(query["AreaOfInterest"]["AreaOfInterestGeometry"])
    trf = query.get("TimeRangeFilter", {})
    start, end = trf.get("StartTime"), trf.get("EndTime")

    # Optional eo:cloud_cover upper bound from PropertyFilters.
    max_cloud = None
    for pf in query.get("PropertyFilters", {}).get("Properties", []):
        cc = pf.get("Property", {}).get("EoCloudCover")
        if cc and "UpperBound" in cc:
            max_cloud = float(cc["UpperBound"])

    # The RasterDataCollectionArn ends in the collection id; loam resolves the alias itself, so pass
    # a known alias when we recognize it, else the arn's tail.
    arn = query.get("RasterDataCollectionArn", "")
    collection = "sentinel-2" if "sentinel-2" in arn.lower() or not arn else arn.rsplit("/", 1)[-1]

    if "CloudMaskingConfig" in job_config:
        op, indices = "cloud-mask", None
    elif "BandMathConfig" in job_config:
        ops = job_config["BandMathConfig"]["CustomIndices"]["Operations"]
        # loam accepts NAME=equation custom specs verbatim (or bare names it knows).
        indices = [f"{o['Name']}={o['Equation']}" if o.get("Equation") else o["Name"] for o in ops]
        op = "band-math"
    else:
        raise ValueError("JobConfig must be a CloudMaskingConfig or BandMathConfig (shim subset)")

    output_uri = output_config["S3Data"]["S3Uri"]
    manifest = plan.build_manifest(
        op=op, collection=collection, aoi=aoi, start=start, end=end,
        indices=indices, max_cloud=max_cloud, output_uri=output_uri,
    )
    plan.write_manifest(manifest, manifest_uri)
    return manifest


@dataclass
class EarthObservationJob:
    """A fake EOJ handle — the manifest is the real work; status comes from loam's S3 ledger.

    Mirrors just enough of the SM ``start_earth_observation_job`` return + ``get_..._job`` status
    for a migrating script. ``Arn`` is a stand-in (the manifest URI); there is no managed job.
    """

    manifest_uri: str
    output_uri: str
    _region: str | None = None

    @property
    def Arn(self) -> str:  # noqa: N802 - mirrors the boto3 field name
        return self.manifest_uri

    def get_status(self) -> dict[str, Any]:
        """Return an SM-shaped status dict, derived from loam's S3 ledger (real progress, no ARN poll).

        ``Status`` is ``COMPLETED`` / ``IN_PROGRESS`` — but unlike SM, it carries real shard counts.
        """
        s = plan.status(self.manifest_uri, region=self._region)
        return {
            "Status": "COMPLETED" if s["complete"] else "IN_PROGRESS",
            # SM never gave you these; the shim exposes loam's real progress underneath.
            "ShardsTotal": s["shards_total"],
            "ShardsCompleted": s["shards_done"],
            "ShardsRemaining": s["shards_remaining"],
        }


def start_earth_observation_job(
    *, Name: str, InputConfig: dict, JobConfig: dict, OutputConfig: dict,  # noqa: N803 - SM shape
    manifest_uri: str | None = None, region: str | None = None,
) -> EarthObservationJob:
    """SM-shaped entrypoint: build a loam manifest from an EOJ config and return a job handle.

    Keyword names mirror ``boto3.client("sagemaker-geospatial").start_earth_observation_job``. Unlike
    the managed service this does not launch compute — it plans the work (writes the manifest); you
    then dispatch shards to a runner (see docs/MIGRATION.md). ``manifest_uri`` defaults to
    ``<OutputConfig S3Uri>/manifest.json``.
    """
    output_uri = OutputConfig["S3Data"]["S3Uri"]
    if manifest_uri is None:
        manifest_uri = output_uri.rstrip("/") + "/manifest.json"
    _build_manifest_from_eoj(InputConfig, JobConfig, OutputConfig, manifest_uri)
    return EarthObservationJob(manifest_uri=manifest_uri, output_uri=output_uri, _region=region)
