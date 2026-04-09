from __future__ import annotations

import mimetypes

import boto3
from botocore.client import Config

from app.core.config import settings


def _require_r2():
    if not settings.r2_bucket or not settings.r2_access_key_id or not settings.r2_secret_access_key:
        raise RuntimeError("R2 is not configured. Set R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY.")


def make_r2_client():
    _require_r2()
    return boto3.client(
        "s3",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        endpoint_url=settings.r2_endpoint_url,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def mime_to_extension(mime_type: str) -> str:
    ext = mimetypes.guess_extension(mime_type.split(";")[0].strip())
    return ext or ".webm"


def build_audio_object_key(*, session_id: str, turn_index: int, chunk_index: int, ext: str) -> str:
    # Keep keys compact and deterministic for idempotency.
    return f"audio/sessions/{session_id}/turns/{turn_index}/chunks/{chunk_index}{ext}"


def presign_put_url(*, object_key: str, content_type: str, expires_seconds: int = 3600) -> str:
    client = make_r2_client()
    return client.generate_presigned_url(
        ClientMethod="put_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key, "ContentType": content_type},
        ExpiresIn=expires_seconds,
    )


def upload_bytes(*, object_key: str, content_type: str, data: bytes) -> None:
    """
    Upload bytes directly to R2 using the S3-compatible endpoint.
    """
    client = make_r2_client()
    client.put_object(Bucket=settings.r2_bucket, Key=object_key, Body=data, ContentType=content_type)


def presign_get_url(*, object_key: str, expires_seconds: int = 3600) -> str:
    client = make_r2_client()
    return client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": settings.r2_bucket, "Key": object_key},
        ExpiresIn=expires_seconds,
    )


