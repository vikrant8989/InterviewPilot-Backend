"""
Whisper transcription worker (placeholder).

Production:
- Redis Queue job consumes R2 audio chunk metadata
- Downloads chunk from R2 to local temp
- Runs Whisper transcription
- Writes transcript to Postgres and notifies session via WS (or via API event bus)

Free-first MVP:
- Keep as skeleton; wire job enqueue in WS/API once you add R2 presign + upload.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass

import httpx
from sqlalchemy import select

from app.core.config import settings
from app.core.r2_storage import make_r2_client
from app.db.session import SessionLocal
from app.db.models import InterviewSession, TranscriptionChunk
from app.ws.session_manager import session_manager


@dataclass(frozen=True)
class TranscriptionJobInput:
    transcription_chunk_id: str
    session_id: str
    turn_index: int


def transcription_job_runner(transcription_chunk_id: str, session_id: str, turn_index: int):
    """
    RQ runs sync functions. We bridge into async with `asyncio.run`.
    """
    asyncio.run(transcribe_and_maybe_finalize(transcription_chunk_id, session_id, turn_index))


async def transcribe_and_maybe_finalize(transcription_chunk_id: str, session_id: str, turn_index: int):
    # Load chunk metadata.
    async with SessionLocal() as db:
        chunk: TranscriptionChunk | None = (
            await db.get(TranscriptionChunk, transcription_chunk_id)
        )
        if not chunk:
            return

        session = (
            await db.execute(select(InterviewSession).where(InterviewSession.id == session_id))
        ).scalar_one_or_none()
        user_id = session.user_id if session else None
        if not user_id:
            return

        if chunk.transcript_text:
            # Idempotency: already transcribed.
            return

        # Download audio from R2.
        # Note: we download bytes to temp file for both OpenAI and local whisper.
        s3 = make_r2_client()
        obj = s3.get_object(Bucket=settings.r2_bucket, Key=chunk.r2_audio_key)
        audio_bytes: bytes = obj["Body"].read()

    # Write temp file outside DB session.
    fd, tmp_path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        transcript_text, language = await transcribe_audio(tmp_path=tmp_path, mime_type=chunk.mime_type)

        # Persist transcript back to DB.
        async with SessionLocal() as db:
            chunk_row: TranscriptionChunk | None = await db.get(TranscriptionChunk, transcription_chunk_id)
            if not chunk_row:
                return
            chunk_row.transcript_text = transcript_text
            chunk_row.language = language
            await db.commit()

        # Notify client of transcript for this chunk (best-effort).
        try:
            await session_manager.send_json(
                session_id=session_id,
                user_id=user_id,
                message={
                    "event": "transcript_ready",
                    "payload": {
                        "turnIndex": turn_index,
                        "chunkIndex": chunk.chunk_index,
                        "transcriptText": transcript_text,
                        "language": language,
                    },
                },
            )
        except Exception:
            pass

        if chunk.is_final_chunk:
            # Notify API server to advance the interview after final chunk.
            await finalize_turn_with_api(session_id=session_id, turn_index=turn_index)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


async def transcribe_audio(*, tmp_path: str, mime_type: str | None) -> tuple[str, str | None]:
    provider = settings.whisper_provider
    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but WHISPER_PROVIDER=openai")
        return await transcribe_with_openai(tmp_path=tmp_path)

    if provider == "local":
        # Local whisper (open-source) - requires dependency `openai-whisper`.
        try:
            import whisper  # type: ignore
        except Exception as e:
            raise RuntimeError("Local whisper provider requires `openai-whisper` installed.") from e

        model = whisper.load_model("small")
        result = model.transcribe(tmp_path)
        return (result.get("text") or "").strip(), result.get("language")

    raise RuntimeError(f"Unsupported whisper_provider: {provider}")


async def transcribe_with_openai(*, tmp_path: str) -> tuple[str, str | None]:
    # Uses OpenAI Whisper API (model: whisper-1).
    base_url = settings.openai_base_url or "https://api.openai.com/v1"
    url = f"{base_url.rstrip('/')}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    data = {"model": settings.openai_transcription_model}

    async with httpx.AsyncClient(timeout=120) as client:
        with open(tmp_path, "rb") as f:
            files = {"file": ("audio.webm", f, "audio/webm")}
            res = await client.post(url, headers=headers, data=data, files=files)
        res.raise_for_status()
        j = res.json()
        return (j.get("text") or "").strip(), None


async def finalize_turn_with_api(*, session_id: str, turn_index: int):
    if not settings.internal_api_url or not settings.internal_secret:
        raise RuntimeError("INTERNAL_API_URL/INTERNAL_SECRET must be set for worker callback")

    url = f"{settings.internal_api_url.rstrip('/')}/api/internal/transcription/finalize"
    headers = {"X-Internal-Secret": settings.internal_secret}
    payload = {"sessionId": session_id, "turnIndex": turn_index}

    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(10):
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            j = res.json()
            if j.get("ok") is True:
                return
            if j.get("reason") == "transcripts_not_ready":
                await asyncio.sleep(2)
                continue
            # Unknown/invalid response; don't loop forever.
            return

