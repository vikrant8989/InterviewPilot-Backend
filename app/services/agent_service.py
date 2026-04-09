from __future__ import annotations

import random
import asyncio
from typing import Any

from app.services.persona_service import load_persona
from app.services.evaluation_service import evaluate_answer
from app.core.config import settings
from openai import AsyncOpenAI


def get_interviewer_type_for_turn(session, turn_index: int) -> str:
    if not getattr(session, "multi_agent", True):
        return "TECH"
    try:
        max_turns = int(getattr(session, "max_turns", 10))
    except Exception:
        max_turns = 10
    if turn_index <= 1:
        return "HR"
    if turn_index >= max_turns - 2:
        return "MANAGER"
    return "TECH"


def _weighted_choice(weights: dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(max(0.0, float(w)) for _, w in items) or 1.0
    r = random.random() * total
    upto = 0.0
    for k, w in items:
        upto += max(0.0, float(w))
        if upto >= r:
            return k
    return items[-1][0]


# Pool of varied fallback questions by role and difficulty
_FALLBACK_QUESTIONS = {
    "SDE": {
        "Easy": [
            "Tell me about a project you're proud of.",
            "How do you approach debugging?",
            "What's your experience with version control?",
            "Describe your coding workflow.",
            "How do you stay updated with new technologies?",
        ],
        "Medium": [
            "Describe a challenging problem you solved.",
            "How do you ensure code quality?",
            "Explain your experience with system design.",
            "How do you handle technical disagreements?",
            "Describe your approach to code reviews.",
        ],
        "Hard": [
            "Design a scalable system for high traffic.",
            "How would you optimize a slow application?",
            "Describe your approach to handling system failures.",
            "How do you design for data consistency?",
            "Explain your microservices architecture experience.",
        ]
    },
    "Frontend": {
        "Easy": [
            "What frontend frameworks have you used?",
            "How do you ensure responsive design?",
            "Describe your CSS workflow.",
            "How do you handle cross-browser compatibility?",
            "What's your experience with state management?",
        ],
        "Medium": [
            "How do you optimize frontend performance?",
            "Describe state management libraries you've used.",
            "How do you handle complex UI interactions?",
            "Explain your approach to component architecture.",
            "How do you test frontend code?",
        ],
        "Hard": [
            "Design a frontend architecture for a large app.",
            "How would you implement real-time updates efficiently?",
            "Describe your frontend testing strategy.",
            "How do you handle frontend security?",
            "Explain your approach to accessibility.",
        ]
    },
    "Backend": {
        "Easy": [
            "What backend frameworks have you used?",
            "How do you handle database migrations?",
            "Describe your REST API experience.",
            "How do you implement authentication?",
            "What's your experience with caching?",
        ],
        "Medium": [
            "How do you design scalable APIs?",
            "Describe database optimization techniques.",
            "How do you handle API rate limiting?",
            "Explain your approach to database transactions.",
            "How do you implement logging and monitoring?",
        ],
        "Hard": [
            "Design a distributed system with high availability.",
            "How would you handle data consistency across services?",
            "Describe your microservices architecture.",
            "How do you handle database sharding?",
            "Explain your approach to system reliability.",
        ]
    }
}


def _get_fallback_question(target_role: str, difficulty: str) -> str:
    """Get a random fallback question from the pool."""
    role_questions = _FALLBACK_QUESTIONS.get(target_role, _FALLBACK_QUESTIONS["SDE"])
    difficulty_questions = role_questions.get(difficulty, role_questions["Medium"])
    question = random.choice(difficulty_questions)
    print(f"[Fallback Question] Using pool for {target_role} - {difficulty}: {question[:50]}...")
    return question


async def _generate_question_with_ai(
    *,
    company: str,
    target_role: str,
    difficulty: str,
    question_type: str,
    interviewer_type: str,
    previous_answer: str | None = None,
    previous_question: str | None = None,
) -> str:
    """Generate a question using OpenAI API with timeout protection."""
    try:
        if not settings.openai_api_key:
            print(f"[AI Question] No API key configured, using fallback")
            return _get_fallback_question(target_role, difficulty)
        
        client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        
        # Build context for the question
        context = f"""
You are an {interviewer_type} interviewer for {company}.
You are interviewing for a {target_role} position.
Current difficulty level: {difficulty}
Question type: {question_type}
"""
        
        if previous_answer and previous_question:
            context += f"""
Previous question: {previous_question}
Candidate's answer: {previous_answer}
Generate a follow-up question based on their answer.
"""
        else:
            context += f"""
Generate an opening interview question for this position.
"""
        
        prompt = f"{context}\n\nGenerate a single, clear interview question. Return only the question text, no explanation or additional text."
        
        print(f"[AI Question] Calling OpenAI API for {target_role} - {difficulty}")
        print(f"[AI Question] Prompt: {prompt[:200]}...")
        # Use asyncio.wait_for to add timeout protection (30 seconds)
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer. Generate clear, relevant interview questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.openai_temperature,
                max_tokens=150,
            ),
            timeout=30.0  # 30 second timeout
        )
        
        question_text = response.choices[0].message.content.strip()
        
        print(f"[AI Question] Raw response: '{question_text}'")
        print(f"[AI Question] Response length: {len(question_text)}")
        
        # Clean up the response if it has extra text
        if question_text.startswith('"') and question_text.endswith('"'):
            question_text = question_text[1:-1]
        
        # Fallback if response is empty or too short
        if not question_text or len(question_text) < 10:
            print(f"[AI Question] Response too short or empty, using fallback")
            return _get_fallback_question(target_role, difficulty)
        
        print(f"[AI Question] Cleaned response: '{question_text[:50]}...'")
        return question_text
        
    except asyncio.TimeoutError:
        print(f"[AI Question] API timeout, using fallback")
        return _get_fallback_question(target_role, difficulty)
    except Exception as e:
        print(f"[AI Question] API error: {e}, using fallback")
        return _get_fallback_question(target_role, difficulty)


async def dynamic_next_question(
    *,
    session,
    turn_index: int,
    user_answer_text: str | None,
    previous_question_text: str | None = None,
) -> dict:
    interviewer_type = get_interviewer_type_for_turn(session, turn_index)
    persona_next = load_persona(session.company, interviewer_type)
    
    if user_answer_text:
        answer_agent_type = get_interviewer_type_for_turn(session, max(turn_index - 1, 0))
        persona_answer = load_persona(session.company, answer_agent_type)
        evaluation_obj = evaluate_answer(
            persona=persona_answer.__dict__,
            question_text=previous_question_text or "",
            answer_text=user_answer_text,
        )
        overall = evaluation_obj.final_score_json.get("overall", 5.0)
        if overall >= persona_next.difficulty_thresholds["good"]:
            difficulty_next = "Hard"
        elif overall <= persona_next.difficulty_thresholds["weak"]:
            difficulty_next = "Easy"
        else:
            difficulty_next = "Medium"
    else:
        evaluation_obj = None
        difficulty_next = session.difficulty_current

    question_type = _weighted_choice(persona_next.question_type_weights)
    
    # Generate question using AI with fallback
    try:
        question_text = await _generate_question_with_ai(
            company=session.company,
            target_role=session.target_role,
            difficulty=difficulty_next,
            question_type=question_type,
            interviewer_type=interviewer_type,
            previous_answer=user_answer_text,
            previous_question=previous_question_text,
        )
    except Exception:
        # Ultimate fallback - use pool
        question_text = _get_fallback_question(session.target_role, difficulty_next)
    
    follow_up = None
    if evaluation_obj and evaluation_obj.final_score_json.get("overall", 5.0) < 6.0:
        follow_up = "Can you elaborate on your approach?"

    return {
        "question_text": question_text,
        "question_type": question_type or "conceptual",
        "question_audio_url": None,
        "difficulty_next": difficulty_next or "Medium",
        "follow_up_text": follow_up,
        "interviewer_type": interviewer_type,
    }
