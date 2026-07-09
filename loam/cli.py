"""loam CLI — a thin wrapper over the library.

Verbs mirror the spore.host house style (truffle's read verbs; a single work atom):

    loam indices                      list the band-math catalog
    loam collections                  list known STAC collections
    loam plan      --op ... --aoi ... build a manifest (search + shard), write to S3/local
    loam run-shard --manifest U -i N  run ONE shard (the executor-agnostic atom)
    loam status    --manifest U       progress, derived from S3
    loam dispatch  --manifest U       print the runner commands (spawn/nf-spawn) — never runs them

The CLI provisions nothing. ``dispatch`` only PRINTS how to hand shards to a runner, keeping
loam execution-agnostic: it shows you the spawn command, it does not call spawn.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _aoi(s: str) -> list[float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--aoi must be W,S,E,N (4 comma-separated floats)")
    return parts


def _cmd_indices(args: argparse.Namespace) -> int:
    from .indices import INDICES

    for name, d in sorted(INDICES.items()):
        print(f"{name:6}  {d.equation:52}  {d.description}")
    return 0


def _cmd_collections(args: argparse.Namespace) -> int:
    from .catalog import COLLECTIONS

    for alias in sorted(set(COLLECTIONS)):
        print(f"{alias:18} -> {COLLECTIONS[alias]}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .plan import build_manifest, write_manifest

    manifest = build_manifest(
        op=args.op,
        collection=args.collection,
        aoi=args.aoi,
        start=args.start,
        end=args.end,
        output_uri=args.output,
        indices=args.indices.split(",") if args.indices else None,
        bands=args.bands.split(",") if args.bands else None,
        dst_crs=args.dst_crs,
        dst_res=args.dst_res,
        resampling=args.resampling,
        reducer=args.reducer,
        max_cloud=args.max_cloud,
        shard_size=args.shard_size,
        limit=args.limit,
        fmt=args.format,
        target_res=args.target_res,
        stac_url=args.stac_url,
    )
    write_manifest(manifest, args.manifest, region=args.region)
    print(
        f"planned: {len(manifest.scenes)} scenes -> {len(manifest.shards)} shards "
        f"(shard_size={args.shard_size}) op={args.op}"
    )
    print(f"manifest: {args.manifest}")
    print(f"outputs:  {args.output}")

    # Compute-shape footer — an estimate of per-shard demand for right-sizing a box (loam
    # describes; it never provisions). Aggregate from the per-shard shapes the plan attached.
    from .shape import human_bytes

    shapes = [sh.shape for sh in manifest.shards if sh.shape]
    if shapes:
        peak = max(s["peak_rss_bytes"] for s in shapes)
        max_read = max(s["approx_bytes_read"] for s in shapes)
        max_secs = max(s["est_seconds"] for s in shapes)
        print(
            f"est/shard (max): ~{human_bytes(max_read)} read · ~{human_bytes(peak)} peak RAM · "
            f"~{max_secs:.0f}s  (order-of-magnitude; feed truffle, not an SLA)"
        )
    print(f"\nnext: hand shards 0..{len(manifest.shards) - 1} to any runner:")
    print(f"  loam dispatch --manifest {args.manifest}")
    return 0


def _cmd_run_shard(args: argparse.Namespace) -> int:
    from .run import run_shard

    summary = run_shard(args.manifest, args.index, region=args.region, force=args.force)
    print(json.dumps(summary))
    return 0 if summary.get("status") in ("done", "skipped") else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from .plan import status

    s = status(args.manifest, region=args.region, detail=args.detail)
    print(json.dumps(s, indent=2))
    return 0


def _cmd_dispatch(args: argparse.Namespace) -> int:
    """Print (do NOT run) the commands a runner would use to process every shard.

    This is the seam that keeps loam agnostic: it emits the spawn / nf-spawn / for-loop
    invocation for each shard and stops. The user (or an outer orchestrator) runs them.
    """
    from .manifest import Manifest
    from . import state

    manifest = Manifest.from_json(state.get_text(args.manifest, region=args.region))
    n = len(manifest.shards)

    if args.runner == "local":
        print(f"# {n} shards — bare loop (laptop / single box):")
        print(f"for i in $(seq 0 {n - 1}); do loam run-shard --manifest {args.manifest} -i $i; done")
    elif args.runner == "spawn":
        from .shape import human_bytes

        print(f"# {n} shards — one spawn box per shard (fan out; scale-out beats one big box):")
        print("# each is idempotent + spot-safe; --on-complete terminate (spawn#262).")
        for sh in manifest.shards:
            i = sh.index
            if sh.shape:  # a suggestion for right-sizing — loam describes, truffle decides
                print(
                    f"# shard {i:05d}: ~{human_bytes(sh.shape['peak_rss_bytes'])} peak RAM, "
                    f"~{human_bytes(sh.shape['approx_bytes_read'])} read, "
                    f"~{sh.shape['est_seconds']:.0f}s (est)"
                )
            print(
                f"spawn launch loam-{i:05d} --instance-type {args.instance} --spot "
                f"--on-complete terminate --iam-policy s3:ReadWrite "
                f"--command 'loam run-shard --manifest {args.manifest} -i {i}'"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loam", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="version", version=f"loam {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("indices", help="list the band-math catalog").set_defaults(func=_cmd_indices)
    sub.add_parser("collections", help="list known STAC collections").set_defaults(
        func=_cmd_collections
    )

    pp = sub.add_parser("plan", help="search + shard into a manifest")
    pp.add_argument("--op", required=True,
                    choices=["band-math", "cloud-mask", "resample", "temporal-composite"])
    pp.add_argument("--collection", default="sentinel-2")
    pp.add_argument("--aoi", type=_aoi, required=True, help="W,S,E,N (WGS84)")
    pp.add_argument("--start", required=True, help="YYYY-MM-DD or RFC3339")
    pp.add_argument("--end", required=True)
    pp.add_argument("--indices", help="comma list, e.g. NDVI,BSI (band-math)")
    pp.add_argument("--bands", help="comma list of bands to reproject, e.g. red,nir (resample)")
    pp.add_argument("--dst-crs", dest="dst_crs", help="target CRS, e.g. EPSG:4326 (resample)")
    pp.add_argument("--dst-res", dest="dst_res", type=float, default=None,
                    help="target pixel size in dst-crs units (resample; omit to preserve count)")
    pp.add_argument("--resampling", default="bilinear",
                    help="resampling method: nearest|bilinear|cubic|average|… (resample)")
    pp.add_argument("--reducer", choices=["median", "mean", "max"], default="median",
                    help="time-reduction for temporal-composite (default median)")
    pp.add_argument("--max-cloud", type=float, default=None)
    pp.add_argument("--shard-size", type=int, default=50)
    pp.add_argument("--limit", type=int, default=None)
    pp.add_argument("--format", choices=["cog", "gtiff", "npy"], default="cog",
                    help="output raster format (default cog: georeferenced Cloud-Optimized GeoTIFF)")
    res = pp.add_mutually_exclusive_group()
    res.add_argument("--target-res", type=float, default=100.0, dest="target_res",
                     help="resolution in metres to read (default 100; overview-based, fewer bytes)")
    res.add_argument("--full-res", action="store_const", const=None, dest="target_res",
                     help="read native full resolution (e.g. Sentinel-2 10m) — for small features")
    pp.add_argument("--output", required=True, help="s3://bucket/prefix/ for shard outputs")
    pp.add_argument("--manifest", required=True, help="s3://... where to write the manifest")
    pp.add_argument("--stac-url", default="https://earth-search.aws.element84.com/v1")
    pp.add_argument("--region", default=None)
    pp.set_defaults(func=_cmd_plan)

    pr = sub.add_parser("run-shard", help="run ONE shard (the runner atom)")
    pr.add_argument("--manifest", required=True)
    pr.add_argument("-i", "--index", type=int, required=True)
    pr.add_argument("--force", action="store_true", help="re-run even if checkpoint exists")
    pr.add_argument("--region", default=None)
    pr.set_defaults(func=_cmd_run_shard)

    ps = sub.add_parser("status", help="progress from S3 (no control plane)")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--detail", action="store_true",
                    help="aggregate a job ledger (bytes/seconds/failures + per-shard rows) from S3")
    ps.add_argument("--region", default=None)
    ps.set_defaults(func=_cmd_status)

    pd = sub.add_parser("dispatch", help="print runner commands for every shard (does not run)")
    pd.add_argument("--manifest", required=True)
    pd.add_argument("--runner", choices=["local", "spawn"], default="spawn")
    pd.add_argument("--instance", default="m8g.4xlarge")
    pd.add_argument("--region", default=None)
    pd.set_defaults(func=_cmd_dispatch)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
