from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import extract_bearer_token, verify_jwt
from app.db.models import InterviewReport, InterviewSession
from app.db.session import get_db

router = APIRouter()


class HistoryItem(BaseModel):
    sessionId: str
    company: str
    targetRole: str
    mode: str | None = None
    difficulty: str | None = None
    date: str | None = None
    startedAt: str | None = None
    endedAt: str | None = None
    overallScore: float | None = None
    status: str | None = None


class HistoryReportResponse(BaseModel):
    sessionId: str
    overallScore: float
    skillBreakdownJson: dict
    keyStrengths: list[str]
    areasForImprovement: list[str]
    recommendations: list[str]
    reportDownloadUrl: str | None = None


@router.get("")
async def list_history(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization token")
    auth = verify_jwt(token)
    user_id = auth["user_id"]

    sessions = (
        await db.execute(
            select(InterviewSession)
            .where(InterviewSession.user_id == user_id)
            .order_by(InterviewSession.created_at.desc())
            .limit(50)
        )
    ).scalars().all()

    items: list[HistoryItem] = []
    for s in sessions:
        report = (
            await db.execute(select(InterviewReport).where(InterviewReport.session_id == s.id))
        ).scalar_one_or_none()

        items.append(
            HistoryItem(
                sessionId=s.id,
                company=s.company,
                targetRole=s.target_role,
                mode=s.interview_mode,
                difficulty=s.difficulty_current,
                date=(s.ended_at or s.created_at).isoformat() if (s.ended_at or s.created_at) else None,
                startedAt=s.started_at.isoformat() if s.started_at else None,
                endedAt=s.ended_at.isoformat() if s.ended_at else None,
                overallScore=report.overall_score if report else None,
                status=s.status,
            )
        )

    return {"items": items}


@router.get("/{sessionId}")
async def get_history(
    sessionId: str,
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization token")
    auth = verify_jwt(token)
    user_id = auth["user_id"]

    session = (
        await db.execute(
            select(InterviewSession).where(InterviewSession.id == sessionId, InterviewSession.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    report = (
        await db.execute(select(InterviewReport).where(InterviewReport.session_id == sessionId))
    ).scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not ready")

    return HistoryReportResponse(
        sessionId=report.session_id,
        overallScore=report.overall_score,
        skillBreakdownJson=report.skill_breakdown_json or {},
        keyStrengths=report.key_strengths or [],
        areasForImprovement=report.areas_for_improvement or [],
        recommendations=report.recommendations or [],
        reportDownloadUrl=report.report_download_url,
    )

