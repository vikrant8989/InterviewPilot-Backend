from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.r2_storage import build_audio_object_key, mime_to_extension, presign_put_url
from app.core.security import extract_bearer_token, verify_jwt
from app.db.session import get_db
from app.db.models import InterviewSession

router = APIRouter()


class PresignUploadRequest(BaseModel):
    sessionId: str
    turnIndex: int
    chunkIndex: int
    mimeType: str


class PresignUploadResponse(BaseModel):
    uploadUrl: str
    r2AudioKey: str


@router.post("/presign", response_model=PresignUploadResponse)
async def presign_audio_upload(
    payload: PresignUploadRequest,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
):
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization token")
    auth = verify_jwt(token)
    user_id = auth["user_id"]

    session = (
        await db.execute(select(InterviewSession).where(InterviewSession.id == payload.sessionId, InterviewSession.user_id == user_id))
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    ext = mime_to_extension(payload.mimeType)
    object_key = build_audio_object_key(
        session_id=payload.sessionId,
        turn_index=payload.turnIndex,
        chunk_index=payload.chunkIndex,
        ext=ext,
    )

    try:
        upload_url = presign_put_url(object_key=object_key, content_type=payload.mimeType)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"R2 presign failed: {e}")

    return PresignUploadResponse(uploadUrl=upload_url, r2AudioKey=object_key)

