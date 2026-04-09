from __future__ import annotations

import io
import tempfile

from app.core.config import settings
from app.core.r2_storage import presign_get_url, upload_bytes


def _tts_object_key(*, session_id: str, turn_index: int) -> str:
    return f"audio/tts/sessions/{session_id}/turns/{turn_index}.mp3"


async def generate_question_tts_audio_url(*, session_id: str, turn_index: int, text: str) -> str | None:
    """
    Free-first TTS implementation.

    - Generates MP3 via gTTS
    - Uploads MP3 bytes to Cloudflare R2
    - Returns a presigned GET URL for the frontend player

    Note: gTTS uses external network; keep question length reasonable.
    """
    if not text or not text.strip():
        return None
    if settings.tts_provider.lower() != "gtts":
        return None

    # gTTS writes to a file; we read file bytes and upload to R2.
    from gtts import gTTS  # local import so backend can start without TTS deps failing at import time

    tts = gTTS(text=text, lang=settings.tts_lang)
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        tts.write_to_fp(f)
        f.seek(0)
        data = f.read()

    object_key = _tts_object_key(session_id=session_id, turn_index=turn_index)
    upload_bytes(object_key=object_key, content_type="audio/mpeg", data=data)
    return presign_get_url(object_key=object_key, expires_seconds=settings.tts_audio_expires_seconds)

