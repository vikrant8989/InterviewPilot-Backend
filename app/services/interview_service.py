from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import WebSocket
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Evaluation, InterviewReport, InterviewSession, ProctorEvent, SessionTurn, TranscriptionChunk
from app.db.session import SessionLocal
from app.services.langgraph_agent_service import interview_graph
from app.services.persona_service import load_persona
from app.services.evaluation_service import evaluate_answer
from app.ws.session_manager import session_manager
from app.core.queue import make_transcription_queue
from app.workers.transcription_worker import transcription_job_runner
from app.services.tts_service import generate_question_tts_audio_url


@dataclass
class ClientCapabilities:
    mode: str | None = None  # text/voice/video


class InterviewService:
    """
    Orchestrates the live interview:
    - loads persona config (company-specific behavior)
    - retrieves RAG context
    - generates next question + optional follow-up
    - evaluates answers (hybrid rule-based + LLM stub)
    """

    async def on_join(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        payload: dict[str, Any],
    ):
        _ = payload
        await websocket.send_json({"event": "joined", "payload": {"sessionId": session_id, "userId": user_id}})

        # Load latest pending turn for this session and push question to client.
        async with SessionLocal() as db:
            session: InterviewSession | None = (
                await db.execute(select(InterviewSession).where(InterviewSession.id == session_id, InterviewSession.user_id == user_id))
            ).scalar_one_or_none()
            if not session:
                await websocket.send_json(
                    {"event": "error", "payload": {"code": "SESSION_NOT_FOUND", "message": "Invalid session"}}
                )
                return

            turn: SessionTurn | None = (
                await db.execute(
                    select(SessionTurn)
                    .where(SessionTurn.session_id == session_id)
                    .where(SessionTurn.evaluation_status == "pending")
                    .order_by(SessionTurn.turn_index.desc())
                )
            ).scalars().first()

            if not turn:
                await websocket.send_json(
                    {
                        "event": "error",
                        "payload": {"code": "NO_PENDING_TURN", "message": "No pending question found"},
                    }
                )
                return

            await websocket.send_json(
                {
                    "event": "agent_question",
                    "payload": {
                        "turnIndex": turn.turn_index,
                        "question_text": turn.question_text,
                        "question_type": turn.question_type,
                        "question_audio_url": turn.question_audio_url,
                        "follow_up_text": None,
                        "difficulty_next": session.difficulty_current,
                        "interviewer_type": interview_graph.get_interviewer_type_for_turn(session, turn.turn_index),
                    },
                }
            )

    async def on_disconnect(self, *, session_id: str):
        # Production: mark session as ENDED/ABORTED if needed.
        _ = session_id

    async def generate_next_question(
        self,
        *,
        session: InterviewSession,
        turn_index: int,
        user_answer_text: str | None,
        db: AsyncSession,
    ) -> dict:
        # Decide next question using dynamic agent with timeout protection.
        try:
            next_payload = await asyncio.wait_for(
                interview_graph.run_interview_turn(
                    session=session,
                    turn_index=turn_index,
                    user_answer_text=user_answer_text,
                ),
                timeout=45.0  # 45 second timeout
            )
        except asyncio.TimeoutError:
            # Fallback question if generation times out
            next_payload = {
                "question_text": f"Tell me about your experience with {session.target_role} roles.",
                "question_type": "behavioral",
                "question_audio_url": None,
                "difficulty_next": session.difficulty_current or "Medium",
                "follow_up_text": None,
                "interviewer_type": interview_graph.get_interviewer_type_for_turn(session, turn_index),
            }
        except Exception:
            # Fallback on any error
            next_payload = {
                "question_text": f"Tell me about your experience with {session.target_role} roles.",
                "question_type": "behavioral",
                "question_audio_url": None,
                "difficulty_next": session.difficulty_current or "Medium",
                "follow_up_text": None,
                "interviewer_type": interview_graph.get_interviewer_type_for_turn(session, turn_index),
            }
        return next_payload

    async def _compile_and_store_report(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> tuple[str, float]:
        """
        Compile a final feedback report from evaluations in DB.
        Returns: (report_id, overall_score)
        """
        async with SessionLocal() as db:
            session: InterviewSession | None = (
                await db.execute(
                    select(InterviewSession).where(
                        InterviewSession.id == session_id, InterviewSession.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if not session:
                return ("", 0.0)

            existing = (
                await db.execute(select(InterviewReport).where(InterviewReport.session_id == session_id))
            ).scalar_one_or_none()
            if existing:
                return (existing.id, existing.overall_score)

            done_turns = (
                await db.execute(
                    select(SessionTurn.id).where(
                        SessionTurn.session_id == session_id,
                        SessionTurn.evaluation_status == "done",
                    )
                )
            ).scalars().all()

            if not done_turns:
                # No answers yet; create an empty report.
                report = InterviewReport(
                    session_id=session_id,
                    overall_score=0.0,
                    skill_breakdown_json={},
                    key_strengths=[],
                    areas_for_improvement=[],
                    recommendations=[],
                    created_at=datetime.utcnow(),
                )
                db.add(report)
                await db.commit()
                await db.refresh(report)
                return (report.id, report.overall_score)

            evaluations = (
                await db.execute(
                    select(Evaluation).where(Evaluation.session_turn_id.in_(done_turns))
                )
            ).scalars().all()

            overall_scores: list[float] = []
            length_scores: list[float] = []
            keyword_scores: list[float] = []
            clarity_scores: list[float] = []
            correctness_scores: list[float] = []
            all_strengths: list[str] = []
            all_weaknesses: list[str] = []
            all_recommendations: list[str] = []

            proctor_types = (
                await db.execute(select(ProctorEvent.event_type).where(ProctorEvent.session_id == session_id))
            ).scalars().all()
            tab_switches = sum(1 for t in proctor_types if t == "tab_switch")
            focus_losts = sum(1 for t in proctor_types if t == "focus_lost")
            if tab_switches > 0:
                all_recommendations.append(f"Proctoring: tab switches detected ({tab_switches}). Maintain focus during interviews.")
            if focus_losts > 0:
                all_recommendations.append(f"Proctoring: focus lost detected ({focus_losts}). Reduce context switching.")

            for ev in evaluations:
                try:
                    overall_scores.append(float(ev.final_score_json.get("overall", 0.0)))
                except Exception:
                    overall_scores.append(0.0)
                try:
                    length_scores.append(float(ev.final_score_json.get("length_score", 0.0)))
                except Exception:
                    length_scores.append(0.0)
                try:
                    keyword_scores.append(float(ev.final_score_json.get("keyword_score", 0.0)))
                except Exception:
                    keyword_scores.append(0.0)
                try:
                    clarity_scores.append(float(ev.final_score_json.get("clarity", 0.0)))
                except Exception:
                    clarity_scores.append(0.0)
                try:
                    correctness_scores.append(float(ev.final_score_json.get("correctness", 0.0)))
                except Exception:
                    correctness_scores.append(0.0)
                all_strengths.extend(ev.strengths or [])
                all_weaknesses.extend(ev.weaknesses or [])
                all_recommendations.extend(ev.improvements or [])

            overall_score = round(sum(overall_scores) / max(len(overall_scores), 1), 2)

            def avg(xs: list[float]) -> float:
                return round(sum(xs) / max(len(xs), 1), 2)

            skill_breakdown_json = {
                "overall": overall_score,
                "length_score": avg(length_scores),
                "keyword_score": avg(keyword_scores),
                "clarity": avg(clarity_scores),
                "correctness": avg(correctness_scores),
            }

            report = InterviewReport(
                session_id=session_id,
                overall_score=overall_score,
                skill_breakdown_json=skill_breakdown_json,
                key_strengths=list(dict.fromkeys(all_strengths))[:8],
                areas_for_improvement=list(dict.fromkeys(all_weaknesses))[:8],
                recommendations=list(dict.fromkeys(all_recommendations))[:10],
                created_at=datetime.utcnow(),
            )

            db.add(report)
            await db.commit()
            await db.refresh(report)
            return (report.id, report.overall_score)

    async def advance_after_answer(
        self,
        *,
        session_id: str,
        user_id: str,
        turn_index: int,
        answer_text: str,
    ):
        print(f"[InterviewService] advance_after_answer called for session {session_id}, turn {turn_index}")
        evaluation_obj = None
        next_payload: dict | None = None
        next_question_audio_url: str | None = None
        should_end = False
        async with SessionLocal() as db:
            session: InterviewSession | None = (
                await db.execute(select(InterviewSession).where(InterviewSession.id == session_id, InterviewSession.user_id == user_id))
            ).scalar_one_or_none()
            if not session:
                return

            turn: SessionTurn | None = (
                await db.execute(
                    select(SessionTurn)
                    .where(SessionTurn.session_id == session_id, SessionTurn.turn_index == turn_index)
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if not turn:
                return

            # Idempotency: only advance if pending.
            if turn.evaluation_status != "pending":
                return

            turn.user_answer_text = answer_text

            answer_agent_type = interview_graph.get_interviewer_type_for_turn(session, turn_index)
            persona = load_persona(session.company, answer_agent_type)
            evaluation_obj = None
            try:
                print(f"[InterviewService] Calling evaluate_answer for turn {turn_index}")
                evaluation_obj = evaluate_answer(
                    persona=persona.__dict__,
                    question_text=turn.question_text,
                    answer_text=answer_text,
                )
                print(f"[InterviewService] Evaluation completed, score: {evaluation_obj.final_score_json.get('overall')}")
            except Exception as e:
                # Fallback evaluation if evaluation fails
                print(f"[InterviewService] Evaluation failed: {e}")
                evaluation_obj = None

            # Lightweight conversation memory to keep context-aware prompting.
            # Free-first MVP: store a truncated rolling window of recent answers.
            try:
                prev_mem = session.memory_summary or ""
                mem_piece = f"Turn {turn_index} answer: {answer_text[:600]}"
                session.memory_summary = (prev_mem + "\n" + mem_piece)[-8000:]
            except Exception:
                pass

            # Ensure evaluation_obj is not None
            if evaluation_obj is None:
                from app.services.evaluation_service import EvaluationResult
                evaluation_obj = EvaluationResult(
                    rule_score_json={"overall": 5.0},
                    llm_score_json={"overall": 5.0},
                    final_score_json={"overall": 5.0},
                    strengths=["Answer submitted"],
                    weaknesses=["Evaluation unavailable"],
                    improvements=["Continue with next question"],
                )

            evaluation_row = Evaluation(
                session_turn_id=turn.id,
                rule_score_json=evaluation_obj.rule_score_json,
                llm_score_json=evaluation_obj.llm_score_json,
                final_score_json=evaluation_obj.final_score_json,
                strengths=evaluation_obj.strengths,
                weaknesses=evaluation_obj.weaknesses,
                improvements=evaluation_obj.improvements,
                created_at=datetime.utcnow(),
            )
            turn.evaluation_status = "done"
            db.add(evaluation_row)

            await db.commit()

            # End interview if we've reached the configured number of turns.
            if turn_index + 1 >= int(session.max_turns or 10):
                session.status = "COMPLETED"
                session.ended_at = datetime.utcnow()
                await db.commit()
                should_end = True
            else:
                # Generate next question (adaptive) with timeout protection.
                try:
                    next_payload = await asyncio.wait_for(
                        interview_graph.run_interview_turn(
                            session=session,
                            turn_index=turn_index + 1,
                            user_answer_text=answer_text,
                            previous_question_text=turn.question_text,
                        ),
                        timeout=45.0  # 45 second timeout for question generation
                    )
                except asyncio.TimeoutError:
                    # Fallback question if generation times out
                    next_payload = {
                        "question_text": f"Tell me about your experience with {session.target_role} roles.",
                        "question_type": "behavioral",
                        "question_audio_url": None,
                        "difficulty_next": session.difficulty_current or "Medium",
                        "follow_up_text": None,
                        "interviewer_type": interview_graph.get_interviewer_type_for_turn(session, turn_index + 1),
                    }
                except Exception:
                    # Fallback on any error
                    next_payload = {
                        "question_text": f"Tell me about your experience with {session.target_role} roles.",
                        "question_type": "behavioral",
                        "question_audio_url": None,
                        "difficulty_next": session.difficulty_current or "Medium",
                        "follow_up_text": None,
                        "interviewer_type": interview_graph.get_interviewer_type_for_turn(session, turn_index + 1),
                    }

                difficulty_next = next_payload.get("difficulty_next")
                if difficulty_next:
                    session.difficulty_current = difficulty_next
                    await db.commit()

                next_turn = SessionTurn(
                    session_id=session.id,
                    turn_index=turn_index + 1,
                    question_type=next_payload.get("question_type"),
                    question_text=next_payload.get("question_text") or "Default question: Tell me about your experience.",
                    question_audio_url=next_question_audio_url,
                    evaluation_status="pending",
                    created_at=datetime.utcnow(),
                )
                db.add(next_turn)
                await db.commit()

                # Generate TTS audio (only for voice/video sessions).
                if session.interview_mode in ("voice", "video"):
                    try:
                        next_question_audio_url = await generate_question_tts_audio_url(
                            session_id=session.id,
                            turn_index=turn_index + 1,
                            text=next_payload["question_text"],
                        )
                        # Update DB row with generated audio URL.
                        next_turn.question_audio_url = next_question_audio_url
                        await db.commit()
                    except Exception:
                        next_question_audio_url = None

        # Notify client(s) via WS manager.
        print(f"[WS] Sending evaluation for turn {turn_index}, score: {evaluation_obj.final_score_json.get('overall')}")
        await session_manager.send_json(
            session_id=session_id,
            user_id=user_id,
            message={
                "event": "agent_evaluation_ready",
                "payload": {
                    "turnIndex": turn_index,
                    "finalScoreJson": evaluation_obj.final_score_json,
                    "strengths": evaluation_obj.strengths,
                    "weaknesses": evaluation_obj.weaknesses,
                    "improvements": evaluation_obj.improvements,
                },
            },
        )

        if should_end:
            report_id, overall_score = await self._compile_and_store_report(session_id=session_id, user_id=user_id)
            await session_manager.send_json(
                session_id=session_id,
                user_id=user_id,
                message={
                    "event": "session_ended",
                    "payload": {"reportId": report_id, "overallScore": overall_score, "reportDownloadUrl": None},
                },
            )
            return

        if next_payload:
            await session_manager.send_json(
                session_id=session_id,
                user_id=user_id,
                message={
                    "event": "agent_question",
                    "payload": {
                        "turnIndex": turn_index + 1,
                        "question_text": next_payload["question_text"],
                        "question_type": next_payload.get("question_type"),
                        "question_audio_url": next_question_audio_url,
                        "follow_up_text": next_payload.get("follow_up_text"),
                        "difficulty_next": next_payload.get("difficulty_next"),
                        "interviewer_type": next_payload.get("interviewer_type"),
                    },
                },
            )

    async def on_text_answer(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        payload: dict[str, Any],
    ):
        """
        payload:
        - turnIndex: int
        - answerText: str
        """
        turn_index = int(payload.get("turnIndex"))
        answer_text = (payload.get("answerText") or "").strip()
        await self.advance_after_answer(
            session_id=session_id,
            user_id=user_id,
            turn_index=turn_index,
            answer_text=answer_text,
        )

    async def on_proctor_event(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        payload: dict[str, Any],
    ):
        # Production: store proctor events (tab switch detection, focus lost) for the session.
        event_type = payload.get("type")
        at = payload.get("at")

        if event_type not in ("tab_switch", "focus_lost"):
            await websocket.send_json(
                {
                    "event": "error",
                    "payload": {"code": "INVALID_PROCTOR_EVENT", "message": "Unsupported proctor event type"},
                }
            )
            return

        async with SessionLocal() as db:
            # Ensure session belongs to user.
            session = (
                await db.execute(
                    select(InterviewSession).where(
                        InterviewSession.id == session_id, InterviewSession.user_id == user_id
                    )
                )
            ).scalar_one_or_none()
            if not session:
                await websocket.send_json(
                    {
                        "event": "error",
                        "payload": {"code": "SESSION_NOT_FOUND", "message": "Invalid session"},
                    }
                )
                return

            db.add(
                ProctorEvent(
                    session_id=session_id,
                    event_type=event_type,
                    at_epoch_ms=int(at // 1000) if isinstance(at, (int, float)) else None,  # Convert to seconds to avoid overflow
                    created_at=datetime.utcnow(),
                )
            )
            await db.commit()

        await websocket.send_json(
            {
                "event": "proctor_ack",
                "payload": {"sessionId": session_id, "userId": user_id, "eventType": event_type},
            }
        )

    async def on_audio_chunk_uploaded(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        payload: dict[str, Any],
    ):
        """
        payload:
        - turnIndex: int
        - chunkIndex: int
        - r2AudioKey: str
        - mimeType: str
        - isFinalChunk: boolean
        """
        turn_index = int(payload.get("turnIndex"))
        chunk_index = int(payload.get("chunkIndex"))
        r2_audio_key = payload.get("r2AudioKey")
        mime_type = payload.get("mimeType")
        is_final = bool(payload.get("isFinalChunk"))

        if not r2_audio_key:
            await websocket.send_json({"event": "error", "payload": {"code": "MISSING_R2_KEY", "message": "r2AudioKey missing"}})
            return

        async with SessionLocal() as db:
            session: InterviewSession | None = (
                await db.execute(select(InterviewSession).where(InterviewSession.id == session_id, InterviewSession.user_id == user_id))
            ).scalar_one_or_none()
            if not session:
                await websocket.send_json({"event": "error", "payload": {"code": "SESSION_NOT_FOUND", "message": "Invalid session"}})
                return

            turn: SessionTurn | None = (
                await db.execute(select(SessionTurn).where(SessionTurn.session_id == session_id, SessionTurn.turn_index == turn_index))
            ).scalar_one_or_none()
            if not turn:
                await websocket.send_json({"event": "error", "payload": {"code": "TURN_NOT_FOUND", "message": f"Turn {turn_index} not found"}})
                return

            # Persist chunk metadata so worker can update transcripts later.
            chunk_row = TranscriptionChunk(
                session_turn_id=turn.id,
                chunk_index=chunk_index,
                r2_audio_key=r2_audio_key,
                mime_type=mime_type,
                is_final_chunk=is_final,
                created_at=datetime.utcnow(),
            )
            db.add(chunk_row)
            await db.commit()
            await db.refresh(chunk_row)

        queue = make_transcription_queue()
        # Enqueue worker job. It will update DB and call internal finalize endpoint for last chunk.
        queue.enqueue(transcription_job_runner, chunk_row.id, session_id, turn_index)

        await websocket.send_json(
            {
                "event": "chunk_upload_ack",
                "payload": {"turnIndex": turn_index, "chunkIndex": chunk_index, "isFinalChunk": is_final},
            }
        )

