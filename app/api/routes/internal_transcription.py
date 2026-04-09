from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.db.models import InterviewSession, SessionTurn, TranscriptionChunk
from app.services.interview_service import InterviewService

router = APIRouter()
service = InterviewService()


class FinalizeTranscriptionRequest(BaseModel):
    sessionId: str
    turnIndex: int


@router.post("/transcription/finalize")
async def finalize_transcription(
    payload: FinalizeTranscriptionRequest,
    internal_secret: str | None = Header(default=None, alias="X-Internal-Secret"),
    db: AsyncSession = Depends(get_db),
):
    if not internal_secret or settings.internal_secret is None or internal_secret != settings.internal_secret:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    session: InterviewSession | None = (
        await db.execute(select(InterviewSession).where(InterviewSession.id == payload.sessionId))
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    turn: SessionTurn | None = (
        await db.execute(
            select(SessionTurn).where(
                SessionTurn.session_id == payload.sessionId,
                SessionTurn.turn_index == payload.turnIndex,
            )
        )
    ).scalar_one_or_none()
    if not turn:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Turn not found")

    # Idempotency: if already evaluated, ignore.
    if turn.evaluation_status != "pending":
        return {"ok": True, "reason": "already_processed"}

    # Load transcript chunks for this turn and build answer text.
    chunks = (
        await db.execute(
            select(TranscriptionChunk)
            .where(TranscriptionChunk.session_turn_id == turn.id)
            .order_by(TranscriptionChunk.chunk_index.asc())
        )
    ).scalars().all()

    # Wait until all chunks for this turn have been transcribed.
    if any(c.transcript_text is None for c in chunks):
        return {"ok": False, "reason": "transcripts_not_ready"}

    transcript_parts = [c.transcript_text or "" for c in chunks]
    answer_text = "\n".join([p for p in transcript_parts if p.strip()]).strip()
    if not answer_text:
        return {"ok": False, "reason": "transcripts_not_ready"}

    # Advance interview and notify client(s) via WS manager.
    await service.advance_after_answer(session_id=payload.sessionId, user_id=session.user_id, turn_index=payload.turnIndex, answer_text=answer_text)

    return {"ok": True}

