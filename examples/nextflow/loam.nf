#!/usr/bin/env nextflow
// Run a loam manifest as a Nextflow pipeline on ephemeral EC2 via spore-host/nf-spawn.
//
// The whole point: `loam run-shard --manifest <uri> -i N` is one idempotent command, so a
// pipeline that fans it over shards 0..N-1 Just Works — loam needs ZERO changes and imports
// nothing from Nextflow or nf-spawn. This file is the proof.
//
// Usage:
//   nextflow run loam.nf -profile nf-spawn --manifest s3://bucket/prefix/manifest.json --n 12
//
// `--n` is the shard count (from `loam plan`'s output, or `loam status --manifest <uri>` →
// `.shards_total`). Kept explicit so the pipeline needs no loam call to enumerate work.

nextflow.enable.dsl = 2

params.manifest = null            // s3:// URI of the manifest written by `loam plan`
params.n        = null            // number of shards (0..n-1)

process runShard {
    // nf-spawn maps this to an ephemeral EC2 box; the container/AMI must have `loam` on PATH
    // (pip install loam-geo) and S3 read/write for the manifest + outputs.
    tag "shard-${idx}"

    input:
    val idx

    script:
    """
    loam run-shard --manifest ${params.manifest} -i ${idx}
    """
}

workflow {
    if( !params.manifest || params.n == null )
        error "pass --manifest <uri> and --n <shard-count> (see `loam status`)"

    // One task per shard. Each is idempotent + spot-safe: a reclaimed shard is simply re-run
    // (its checkpoint makes a completed shard a no-op), so the pull model needs no bookkeeping.
    Channel.of( 0 .. (params.n as int) - 1 ) | runShard
}
