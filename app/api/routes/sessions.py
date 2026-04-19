from datetime import datetime
import asyncio

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import extract_bearer_token, verify_jwt
from app.db.session import get_db
from app.db.models import InterviewSession, SessionTurn, User, Evaluation, TranscriptionChunk, ProctorEvent, InterviewReport
from app.services.interview_service import InterviewService
from app.services.tts_service import generate_question_tts_audio_url
from app.services.langgraph_agent_service import interview_graph

router = APIRouter()
service = InterviewService()


class CreateSessionRequest(BaseModel):
    company: str
    targetRole: str
    difficulty: str  # Easy/Medium/Hard
    interviewMode: str  # text/voice/video
    multiAgent: bool = True
    maxTurns: int = 10


class CreateSessionResponse(BaseModel):
    sessionId: str


class StartSessionResponse(BaseModel):
    sessionId: str
    firstQuestion: dict
    maxTurns: int


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    payload: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
):
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization token")
    auth = verify_jwt(token)
    user_id = auth["user_id"]

    existing_user = await db.execute(select(User).where(User.id == user_id))
    if not existing_user.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    max_turns = payload.maxTurns if payload.maxTurns >= 3 else 10
    session = InterviewSession(
        user_id=user_id,
        company=payload.company,
        target_role=payload.targetRole,
        difficulty_start=payload.difficulty,
        difficulty_current=payload.difficulty,
        interview_mode=payload.interviewMode,
        status="CREATED",
        max_turns=max_turns,
        multi_agent=payload.multiAgent,
        created_at=datetime.utcnow(),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return CreateSessionResponse(sessionId=session.id)


@router.post("/{sessionId}/start", response_model=StartSessionResponse)
async def start_session(
    sessionId: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
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

    session.status = "IN_PROGRESS"
    session.started_at = datetime.utcnow()
    await db.commit()

    # Generate first question with timeout protection
    try:
        first = await asyncio.wait_for(
            service.generate_next_question(session=session, turn_index=0, user_answer_text=None, db=db),
            timeout=45.0  # 45 second timeout
        )
    except asyncio.TimeoutError:
        # Fallback question if generation times out
        first = {
            "question_text": f"Tell me about your experience with {session.target_role} roles.",
            "question_type": "behavioral",
            "question_audio_url": None,
            "difficulty_next": session.difficulty_current or "Medium",
            "follow_up_text": None,
            "interviewer_type": "HR",
        }
    except Exception:
        # Fallback on any error
        first = {
            "question_text": f"Tell me about your experience with {session.target_role} roles.",
            "question_type": "behavioral",
            "question_audio_url": None,
            "difficulty_next": session.difficulty_current or "Medium",
            "follow_up_text": None,
            "interviewer_type": "HR",
        }
    
    if first.get("difficulty_next"):
        session.difficulty_current = first["difficulty_next"]
        await db.commit()

    question_audio_url = None
    if session.interview_mode in ("voice", "video"):
        try:
            question_audio_url = await generate_question_tts_audio_url(
                session_id=session.id,
                turn_index=0,
                text=first["question_text"],
            )
        except Exception:
            question_audio_url = None
    
    turn = SessionTurn(
        session_id=session.id,
        turn_index=0,
        question_type=first.get("question_type", "behavioral"),
        question_text=first.get("question_text", "Tell me about your experience."),
        question_audio_url=question_audio_url,
        evaluation_status="pending",
        created_at=datetime.utcnow(),
    )
    db.add(turn)
    await db.commit()

    return StartSessionResponse(sessionId=session.id, firstQuestion=first, maxTurns=session.max_turns)


@router.post("/{sessionId}/end")
async def end_session(
    sessionId: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
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

    session.status = "COMPLETED"
    session.ended_at = datetime.utcnow()
    await db.commit()
    report_id, overall_score = await service._compile_and_store_report(session_id=sessionId, user_id=user_id)
    return {"reportId": report_id, "overallScore": overall_score}


@router.get("/{sessionId}/turns")
async def get_session_turns(
    sessionId: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
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

    turns_result = await db.execute(
        select(SessionTurn).where(SessionTurn.session_id == sessionId).order_by(SessionTurn.turn_index)
    )
    turns = turns_result.scalars().all()

    turn_data = []
    for turn in turns:
        turn_data.append({
            "turnIndex": turn.turn_index,
            "questionText": turn.question_text,
            "questionType": turn.question_type,
            "questionAudioUrl": turn.question_audio_url,
            "userAnswerText": turn.user_answer_text,
            "evaluationStatus": turn.evaluation_status,
        })

    return {"sessionId": sessionId, "turns": turn_data, "maxTurns": session.max_turns, "status": session.status}


@router.delete("/{sessionId}")
async def delete_session(
    sessionId: str,
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(default=None),
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

    # Get all session turn IDs for this session
    turns_result = await db.execute(
        select(SessionTurn.id).where(SessionTurn.session_id == sessionId)
    )
    turn_ids = [row[0] for row in turns_result.all()]

    # Delete in correct order to handle foreign key constraints:
    # 1. Evaluations (references session_turns)
    if turn_ids:
        await db.execute(delete(Evaluation).where(Evaluation.session_turn_id.in_(turn_ids)))
    
    # 2. TranscriptionChunks (references session_turns)
    if turn_ids:
        await db.execute(delete(TranscriptionChunk).where(TranscriptionChunk.session_turn_id.in_(turn_ids)))
    
    # 3. SessionTurns (references interview_sessions)
    await db.execute(delete(SessionTurn).where(SessionTurn.session_id == sessionId))
    
    # 4. ProctorEvents (references interview_sessions)
    await db.execute(delete(ProctorEvent).where(ProctorEvent.session_id == sessionId))
    
    # 5. InterviewReports (references interview_sessions)
    await db.execute(delete(InterviewReport).where(InterviewReport.session_id == sessionId))
    
    # 6. Finally delete the session
    await db.execute(delete(InterviewSession).where(InterviewSession.id == sessionId))
    
    await db.commit()
    return {"message": "Session deleted successfully"}

