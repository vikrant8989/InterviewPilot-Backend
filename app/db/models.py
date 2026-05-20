import uuid

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, BigInteger, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def uuid4_str() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    auth_provider: Mapped[str] = mapped_column(String(50), nullable=False, default="local")

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)


class OAuthIdentity(Base):
    __tablename__ = "oauth_identities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)

    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # google
    provider_subject: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)

    user = relationship("User")


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)

    company: Mapped[str] = mapped_column(String(100), nullable=False)
    target_role: Mapped[str] = mapped_column(String(100), nullable=False)

    difficulty_start: Mapped[str] = mapped_column(String(20), nullable=False)
    difficulty_current: Mapped[str] = mapped_column(String(20), nullable=False)

    interview_mode: Mapped[str] = mapped_column(String(20), nullable=False)  # text/video
    status: Mapped[str] = mapped_column(String(30), nullable=False)  # CREATED/IN_PROGRESS/...

    max_turns: Mapped[int] = mapped_column(Integer, nullable=False, default=10)

    multi_agent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    started_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=True)
    memory_summary: Mapped[str | None] = mapped_column(Text(), nullable=True)

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)

    user = relationship("User")


class SessionTurn(Base):
    __tablename__ = "session_turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)

    question_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    question_text: Mapped[str] = mapped_column(Text(), nullable=False)
    question_audio_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    user_answer_text: Mapped[str | None] = mapped_column(Text(), nullable=True)
    user_answer_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    evaluation_status: Mapped[str] = mapped_column(String(30), nullable=False, default="pending")
    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    session_turn_id: Mapped[str] = mapped_column(ForeignKey("session_turns.id"), nullable=False)

    rule_score_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    llm_score_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    final_score_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    strengths: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    weaknesses: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    improvements: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)


class ProctorEvent(Base):
    __tablename__ = "proctor_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)  # tab_switch, focus_lost
    at_epoch_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)


class InterviewReport(Base):
    __tablename__ = "interview_reports"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid4_str)
    session_id: Mapped[str] = mapped_column(ForeignKey("interview_sessions.id"), unique=True, nullable=False)

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    skill_breakdown_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    key_strengths: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    areas_for_improvement: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    recommendations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    report_download_url: Mapped[str | None] = mapped_column(Text(), nullable=True)
    created_at: Mapped["DateTime"] = mapped_column(DateTime(timezone=True), nullable=False)

