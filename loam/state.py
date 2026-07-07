"""State lives in S3, not in loam.

The execution-agnostic contract's second pillar: there is no control plane and no job
status held in memory. A shard is *done* iff its checkpoint object exists in S3. Progress is
therefore just an ``ls`` — which also fixes SageMaker Geospatial's worst UX wart (an
opaque ``IN_PROGRESS`` with no percentage): ``loam status`` counts objects.

This module wraps the tiny bit of S3 I/O loam needs (get/put text, read/write bytes, exists,
list). It shells nothing out; it uses boto3 so it works from a laptop or an instance role.

URIs are ``s3://bucket/key`` or plain local paths (for tests / laptop runs) — both supported
so a shard can run with zero AWS in a unit test.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def _split(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _client(region: str | None = None):
    import boto3

    return boto3.client("s3", region_name=region or os.environ.get("AWS_DEFAULT_REGION"))


def join(uri: str, *parts: str) -> str:
    """Join a base URI/path with additional path segments."""
    base = uri.rstrip("/")
    tail = "/".join(p.strip("/") for p in parts)
    return f"{base}/{tail}"


def exists(uri: str, *, region: str | None = None) -> bool:
    if not is_s3(uri):
        return Path(uri).exists()
    bucket, key = _split(uri)
    from botocore.exceptions import ClientError

    try:
        _client(region).head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def put_text(uri: str, text: str, *, region: str | None = None) -> None:
    put_bytes(uri, text.encode("utf-8"), region=region)


def get_text(uri: str, *, region: str | None = None) -> str:
    return get_bytes(uri, region=region).decode("utf-8")


def put_bytes(uri: str, data: bytes, *, region: str | None = None) -> None:
    if not is_s3(uri):
        p = Path(uri)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return
    bucket, key = _split(uri)
    _client(region).put_object(Bucket=bucket, Key=key, Body=data)


def get_bytes(uri: str, *, region: str | None = None) -> bytes:
    if not is_s3(uri):
        return Path(uri).read_bytes()
    bucket, key = _split(uri)
    return _client(region).get_object(Bucket=bucket, Key=key)["Body"].read()


def list_keys(prefix_uri: str, *, region: str | None = None) -> list[str]:
    """List object URIs under a prefix (recursively). Works for local dirs too."""
    if not is_s3(prefix_uri):
        base = Path(prefix_uri)
        if not base.exists():
            return []
        return [str(p) for p in base.rglob("*") if p.is_file()]
    bucket, key = _split(prefix_uri)
    if key and not key.endswith("/"):
        key += "/"
    out: list[str] = []
    paginator = _client(region).get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
            out.append(f"s3://{bucket}/{obj['Key']}")
    return out


# ── Shard checkpoint convention ──────────────────────────────────────────────
# A shard's completion marker. run-shard writes this LAST, only after its real output is
# durable — so "checkpoint exists" strictly implies "output is safe" (delete-after-durable).

def checkpoint_uri(output_uri: str, shard_index: int) -> str:
    return join(output_uri, "_checkpoints", f"shard-{shard_index:05d}.done")


def output_uri_for(output_uri: str, shard_index: int, filename: str) -> str:
    return join(output_uri, f"shard={shard_index:05d}", filename)


def shard_done(output_uri: str, shard_index: int, *, region: str | None = None) -> bool:
    return exists(checkpoint_uri(output_uri, shard_index), region=region)
