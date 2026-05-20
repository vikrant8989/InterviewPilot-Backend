from __future__ import annotations

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from app.services.persona_service import load_persona
from app.services.langchain_rag_service import langchain_rag
from app.services.langchain_memory_service import get_memory_context, update_session_memory
from app.services.evaluation_service import langchain_evaluator
from app.core.config import settings


class InterviewState(TypedDict):
    """State for the interview workflow"""
    messages: Annotated[list, add_messages]
    session: Any
    session_id: str
    turn_index: int
    company: str
    target_role: str
    difficulty_current: str
    user_answer: Optional[str]
    previous_question: Optional[str]
    interviewer_type: str
    persona: Any
    evaluation: Optional[Any]
    rag_context: Optional[str]
    memory_summary: str
    memory_context: str
    # Output fields set by nodes
    question_text: Optional[str]
    question_type: Optional[str]
    difficulty_next: Optional[str]
    follow_up_text: Optional[str]


class InterviewAgentGraph:
    """LangGraph-based multi-agent interview system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
            temperature=settings.openai_temperature
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the interview workflow graph"""
        workflow = StateGraph(InterviewState)
        
        # Add nodes
        workflow.add_node("determine_agent", self._determine_agent_type)
        workflow.add_node("load_persona", self._load_persona_node)
        workflow.add_node("get_memory_context", self._get_memory_context_node)
        workflow.add_node("retrieve_rag", self._retrieve_rag_context)
        workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        workflow.add_node("generate_question", self._generate_question_node)
        workflow.add_node("adjust_difficulty", self._adjust_difficulty_node)
        
        # Add edges
        workflow.add_edge(START, "determine_agent")
        workflow.add_edge("determine_agent", "load_persona")
        workflow.add_edge("load_persona", "get_memory_context")
        workflow.add_edge("get_memory_context", "retrieve_rag")
        
        # Conditional edge for evaluation
        workflow.add_conditional_edges(
            "retrieve_rag",
            self._should_evaluate,
            {
                "evaluate": "evaluate_answer",
                "generate": "generate_question"
            }
        )
        
        workflow.add_edge("evaluate_answer", "adjust_difficulty")
        workflow.add_edge("adjust_difficulty", "generate_question")
        workflow.add_edge("generate_question", END)
        
        return workflow.compile()
    
    def _determine_agent_type(self, state: InterviewState) -> InterviewState:
        """Determine which agent type should handle this turn"""
        try:
            print(f"DEBUG: _determine_agent_type called with state keys: {list(state.keys())}")
            turn_index = state.get("turn_index", 0)
            session = state.get("session")
            print(f"DEBUG: session in determine_agent_type: {session}, type: {type(session)}")
            
            if not session:
                interviewer_type = "TECH"  # Default fallback
                print("DEBUG: Using fallback interviewer_type=TECH (session is None)")
            elif not getattr(session, "multi_agent", True):
                interviewer_type = "TECH"
                print("DEBUG: Using fallback interviewer_type=TECH (multi_agent=False)")
            else:
                try:
                    max_turns = int(getattr(session, "max_turns", 10))
                except Exception:
                    max_turns = 10
                
                if turn_index <= 1:
                    interviewer_type = "HR"
                elif turn_index >= max_turns - 2:
                    interviewer_type = "MANAGER"
                else:
                    interviewer_type = "TECH"
                print(f"DEBUG: Determined interviewer_type={interviewer_type} for turn_index={turn_index}")
            
            state["interviewer_type"] = interviewer_type
            return state
        except Exception as e:
            print(f"DEBUG: Error in _determine_agent_type: {e}")
            state["interviewer_type"] = "TECH"
            return state
    
    def _load_persona_node(self, state: InterviewState) -> InterviewState:
        """Load the appropriate persona for the agent type"""
        try:
            print(f"DEBUG: _load_persona_node called with state keys: {list(state.keys())}")
            interviewer_type = state.get("interviewer_type", "TECH")
            company = state.get("company", "")
            print(f"DEBUG: Loading persona for company={company}, interviewer_type={interviewer_type}")
            
            persona = load_persona(company, interviewer_type)
            print(f"DEBUG: Loaded persona: {persona}, type: {type(persona)}")
        except Exception as e:
            print(f"DEBUG: Error loading persona: {e}")
            persona = None  # Fallback if persona loading fails
        
        state["persona"] = persona
        return state
    
    def _get_memory_context_node(self, state: InterviewState) -> InterviewState:
        """Get conversation memory context"""
        session_id = state.get("session_id", "")
        memory_context = get_memory_context(session_id)
        state["memory_context"] = memory_context
        return state
    
    async def _retrieve_rag_context(self, state: InterviewState) -> InterviewState:
        """Retrieve relevant context using LangChain RAG"""
        session = state.get("session")
        user_answer = state.get("user_answer", "")
        
        if not session:
            # Fallback if session is not available
            state["rag_context"] = ""
            state["question_type"] = "conceptual"
            return state
        
        # Determine question type based on persona
        persona = state.get("persona")
        if persona:
            question_type = self._weighted_choice(persona.question_type_weights)
        else:
            question_type = "conceptual"  # Default fallback
        
        try:
            rag = await langchain_rag.get_interview_context(
                company=getattr(session, 'company', ''),
                role=getattr(session, 'target_role', ''),
                difficulty=state.get("difficulty_current", "Medium"),
                question_type=question_type,
                transcript=user_answer or "",
            )
            state["rag_context"] = rag.get("injected_context_text", "")
        except Exception as e:
            print(f"RAG context retrieval failed: {e}")
            state["rag_context"] = ""
        
        state["question_type"] = question_type
        return state
    
    def _should_evaluate(self, state: InterviewState) -> str:
        """Determine if we should evaluate the previous answer"""
        if state.get("user_answer") and state.get("previous_question"):
            return "evaluate"
        return "generate"
    
    def _evaluate_answer_node(self, state: InterviewState) -> InterviewState:
        """Evaluate the user's answer using LangChain evaluation"""
        persona = state.get("persona")
        previous_question = state.get("previous_question")
        user_answer = state.get("user_answer")
        
        # Use LangChain evaluation service
        if persona and previous_question and user_answer:
            evaluation = langchain_evaluator.rule_based_evaluate(
                persona=persona.__dict__,
                question_text=previous_question,
                answer_text=user_answer
            )
        else:
            evaluation = None
        
        state["evaluation"] = evaluation
        return state
    
    def _adjust_difficulty_node(self, state: InterviewState) -> InterviewState:
        """Adjust difficulty based on evaluation"""
        evaluation = state.get("evaluation")
        persona = state.get("persona")
        session = state.get("session")
        
        if evaluation and persona and session:
            try:
                overall = float(evaluation.get("overall", 0.0))
                difficulty_current = getattr(session, 'difficulty_current', 'Medium')
                
                if overall >= getattr(persona, 'difficulty_thresholds', {}).get("good", 7.0):
                    difficulty_next = "Hard" if difficulty_current != "Hard" else "Hard"
                elif overall <= getattr(persona, 'difficulty_thresholds', {}).get("weak", 4.0):
                    difficulty_next = "Easy" if difficulty_current != "Easy" else "Easy"
                else:
                    difficulty_next = "Medium"
            except Exception as e:
                print(f"Difficulty adjustment failed: {e}")
                difficulty_next = getattr(session, 'difficulty_current', 'Medium')
        else:
            difficulty_next = getattr(session, 'difficulty_current', 'Medium') if session else "Medium"
        
        state["difficulty_next"] = difficulty_next
        return state
    
    def _generate_question_node(self, state: InterviewState) -> InterviewState:
        """Generate the next interview question"""
        persona = state.get("persona")
        company = state.get("company", "")
        target_role = state.get("target_role", "")
        difficulty = state.get("difficulty_next", state.get("difficulty_current", "Medium"))
        question_type = state.get("question_type", "conceptual")
        rag_context = state.get("rag_context", "")
        memory_summary = state.get("memory_summary", "")
        interviewer_type = state.get("interviewer_type", "TECH")
        
        if not persona:
            # Fallback if persona is not available
            state["question_text"] = f"Tell me about your experience with {target_role} roles."
            state["question_type"] = question_type
            state["difficulty_next"] = difficulty
            state["follow_up_text"] = None
            return state
        
        try:
            # Create prompt template for question generation
            system_prompt = """You are an AI interview interviewer. Generate one clear, specific interview question.
            
            Keep it concise and focused on the role/company.
            Return ONLY the question text, no formatting or explanations."""
            
            human_prompt = f"""
            You are interviewing for {company}.

            Role: {target_role}
            Interviewer Type: {interviewer_type}
            Difficulty: {difficulty}
            Question Type: {question_type}

            Relevant Knowledge:
            {rag_context}

            Instructions:
            - Ask a non-generic, role-specific question
            - Avoid repetition
            - Make it realistic (like real interview)

            Generate ONLY the question.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            try:
                print(f"DEBUG: About to invoke LLM with system prompt length: {len(system_prompt)}")
                print(f"DEBUG: About to invoke LLM with human prompt length: {len(human_prompt)}")
                print(f"DEBUG: RAG context: {rag_context}")
                response = self.llm.invoke(messages)
                raw_question_text = response.content.strip()
                print(f"DEBUG: LLM response received: '{raw_question_text}', length: {len(raw_question_text)}")
                print(f"DEBUG: Raw LLM response: '{raw_question_text}'")
                print(f"DEBUG: LLM response object: {response}")
                
                # Clean up the response but keep it simple
                question_text = raw_question_text
                
                # Basic cleanup only
                if question_text.startswith("Question:"):
                    question_text = question_text[9:].strip()
                if question_text.startswith("**Question:**"):
                    question_text = question_text[13:].strip()
                
                # Remove markdown formatting
                import re
                question_text = re.sub(r'\*\*(.*?)\*\*', r'\1', question_text)
                question_text = re.sub(r'\*(.*?)\*', r'\1', question_text)
                question_text = question_text.strip()
                
                print(f"DEBUG: Cleaned question: '{question_text}'")
                
                # Ensure we have a question
                if not question_text or len(question_text) < 10:
                    question_text = f"Describe your experience with {target_role} responsibilities and what you enjoy most about it."
                
                # Generate follow-up if needed
                follow_up = None
                evaluation = state.get("evaluation")
                if evaluation and evaluation.get("overall", 0) < 6.0:
                    follow_up = "Follow-up: What assumptions did you make, and how would you validate them with real data?"
                
                state["question_text"] = question_text
                state["question_type"] = question_type
                state["difficulty_next"] = difficulty
                state["follow_up_text"] = follow_up
                print(f"DEBUG: Set state['question_text'] = '{question_text[:50]}...'")
                
            except Exception as e:
                print(f"Question generation failed: {e}")
                import traceback
                traceback.print_exc()
                # Ultimate fallback
                state["question_text"] = f"Describe your experience with {target_role} responsibilities."
                state["question_type"] = question_type
                state["difficulty_next"] = difficulty
                state["follow_up_text"] = None
                print(f"DEBUG: Set fallback state['question_text']")
        
        except Exception as e:
            print(f"DEBUG: Error in _generate_question_node: {e}")
            # Ultimate fallback
            state["question_text"] = f"Describe your experience with {target_role} responsibilities."
            state["question_type"] = question_type
            state["difficulty_next"] = difficulty
            state["follow_up_text"] = None
        
        print(f"DEBUG: _generate_question_node returning state with question_text: '{state.get('question_text', 'NONE')}'")
        return state
    
    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Make a weighted random choice"""
        import random
        items = list(weights.items())
        total = sum(max(0.0, float(w)) for _, w in items) or 1.0
        r = random.random() * total
        upto = 0.0
        for k, w in items:
            upto += max(0.0, float(w))
            if upto >= r:
                return k
        return items[-1][0]
    
    async def run_interview_turn(
        self,
        session,
        turn_index: int,
        user_answer_text: str | None = None,
        previous_question_text: str | None = None,
    ) -> Dict[str, Any]:
        """Run a complete interview turn"""
        
        try:
            session_id = getattr(session, 'id', str(session))
            print(f"DEBUG: Starting LangGraph turn - session_id: {session_id}, turn_index: {turn_index}")
            
            # Update memory with previous exchange if available
            if user_answer_text and previous_question_text:
                # Get interviewer type for this turn
                interviewer_type = self.get_interviewer_type_for_turn(session, turn_index - 1)
                update_session_memory(session_id, previous_question_text, user_answer_text, interviewer_type)
            
            initial_state = InterviewState(
                messages=[],
                session=session,
                session_id=session_id,
                turn_index=turn_index,
                company=getattr(session, 'company', ''),
                target_role=getattr(session, 'target_role', ''),
                difficulty_current=getattr(session, 'difficulty_current', 'Medium'),
                user_answer=user_answer_text,
                previous_question=previous_question_text,
                interviewer_type="",
                persona=None,
                evaluation=None,
                rag_context=None,
                memory_summary=(getattr(session, "memory_summary", "") or "")[:4000],
                memory_context=""
            )
            
            print(f"DEBUG: Initial state created - company: {initial_state['company']}, role: {initial_state['target_role']}")
            
            # Run the workflow
            result = await self.graph.ainvoke(initial_state)
            
            print(f"DEBUG: Full result keys: {list(result.keys())}")
            print(f"DEBUG: result.get('question_text'): {result.get('question_text')}")
            print(f"DEBUG: result.get('question_text') type: {type(result.get('question_text'))}")
            print(f"DEBUG: LangGraph completed successfully - question: {result.get('question_text', 'NONE')}")
            
            # Update memory with new question
            new_question = result.get("question_text", "")
            if new_question:
                interviewer_type = result.get("interviewer_type", "TECH")
                # Note: We'll update memory again when the user answers
                
            return {
                "question_text": result.get("question_text") or f"Tell me about your experience as a {getattr(session, 'target_role', 'candidate')}.",
                "question_type": result.get("question_type") or "conceptual",
                "question_audio_url": None,  # TODO: TTS integration
                "difficulty_next": result.get("difficulty_next") or "Medium",
                "follow_up_text": result.get("follow_up_text"),
                "interviewer_type": result.get("interviewer_type") or "TECH",
                "evaluation": result.get("evaluation"),
                "rag_context": result.get("rag_context"),
                "memory_context": result.get("memory_context")
            }
            
        except Exception as e:
            print(f"DEBUG: LangGraph failed with error: {e}")
            print(f"DEBUG: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            # Re-raise to trigger fallback
            raise e
    
    def get_interviewer_type_for_turn(self, session, turn_index: int) -> str:
        """Helper method to determine interviewer type"""
        if not getattr(session, "multi_agent", True):
            return "TECH"
        try:
            max_turns = int(getattr(session, "max_turns", 10))
        except Exception:
            max_turns = 10

        if turn_index <= 1:
            return "HR"
        elif turn_index >= max_turns - 2:
            return "MANAGER"
        else:
            return "TECH"


# Singleton instance
interview_graph = InterviewAgentGraph()
