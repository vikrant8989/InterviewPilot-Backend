from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import settings


class InterviewMemoryManager:
    """Simplified memory management for interview conversations"""
    
    def __init__(self, memory_type: str = "buffer_window", window_size: int = 10):
        self.memory_type = memory_type
        self.window_size = window_size
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_chat_model,
            temperature=0.3  # Lower temperature for consistent summaries
        )
        
        # Simple message storage
        self.messages: List[BaseMessage] = []
    
    def add_exchange(self, question: str, answer: str, interviewer_type: str = "TECH") -> None:
        """Add a question-answer exchange to memory"""
        # Add question as AI message (interviewer)
        self.messages.append(
            AIMessage(content=f"[{interviewer_type}] {question}")
        )
        
        # Add answer as human message (candidate)
        self.messages.append(
            HumanMessage(content=answer)
        )
        
        # Keep only recent messages
        if len(self.messages) > self.window_size * 2:  # Q&A pairs
            self.messages = self.messages[-(self.window_size * 2):]
    
    def add_system_context(self, context: str) -> None:
        """Add system context to memory"""
        self.messages.append(
            SystemMessage(content=context)
        )
    
    def get_memory_summary(self) -> str:
        """Get a summary of conversation memory"""
        if not self.messages:
            return ""
        
        # Create a formatted summary of recent exchanges
        summary_parts = []
        for i, msg in enumerate(self.messages[-self.window_size:]):
            if isinstance(msg, AIMessage):
                summary_parts.append(f"Interviewer: {msg.content}")
            elif isinstance(msg, HumanMessage):
                summary_parts.append(f"Candidate: {msg.content}")
        
        return "\n".join(summary_parts)
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for LangChain chains"""
        return {"chat_history": self.messages}
    
    def get_recent_messages(self, count: int = 5) -> List[BaseMessage]:
        """Get the most recent messages"""
        return self.messages[-count:] if self.messages else []
    
    def clear_memory(self) -> None:
        """Clear all memory"""
        self.messages = []
    
    def get_context_for_next_question(self) -> str:
        """Get formatted context for generating next question"""
        recent_messages = self.get_recent_messages(6)  # Last 3 exchanges
        context_parts = []
        
        for msg in recent_messages:
            if isinstance(msg, AIMessage):
                context_parts.append(f"Q: {msg.content}")
            elif isinstance(msg, HumanMessage):
                # Truncate long answers
                content = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                context_parts.append(f"A: {content}")
        
        return "\n".join(context_parts)
    
    def extract_key_topics(self) -> List[str]:
        """Extract key topics discussed in the conversation"""
        if not self.messages:
            return []
        
        # Simple keyword extraction from recent messages
        all_text = " ".join([msg.content for msg in self.messages[-10:]])
        
        # Common technical topics to look for
        tech_topics = [
            "algorithm", "data structure", "system design", "api", "database",
            "scalability", "performance", "security", "testing", "deployment",
            "microservices", "architecture", "optimization", "caching",
            "load balancing", "monitoring", "logging", "error handling"
        ]
        
        found_topics = []
        for topic in tech_topics:
            if topic.lower() in all_text.lower():
                found_topics.append(topic)
        
        return found_topics
    
    def get_difficulty_progression(self) -> Dict[str, Any]:
        """Analyze difficulty progression through the conversation"""
        if not self.messages:
            return {"progression": [], "trend": "stable"}
        
        # This is a simplified analysis - in practice, you'd want more sophisticated NLP
        difficulty_indicators = {
            "easy": ["basic", "simple", "what is", "explain", "describe"],
            "medium": ["how would", "approach", "design", "implement"],
            "hard": ["optimize", "scale", "complex", "challenge", "tradeoff"]
        }
        
        progression = []
        for msg in self.messages:
            if isinstance(msg, AIMessage):
                content = msg.content.lower()
                score = 0
                
                for difficulty, indicators in difficulty_indicators.items():
                    for indicator in indicators:
                        if indicator in content:
                            if difficulty == "easy":
                                score -= 1
                            elif difficulty == "hard":
                                score += 1
                            break
                
                if score > 0:
                    progression.append("hard")
                elif score < 0:
                    progression.append("easy")
                else:
                    progression.append("medium")
        
        # Determine trend
        if len(progression) < 2:
            trend = "stable"
        elif progression[-1] == "hard" and progression[0] != "hard":
            trend = "increasing"
        elif progression[-1] == "easy" and progression[0] != "easy":
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "progression": progression,
            "trend": trend,
            "current_difficulty": progression[-1] if progression else "medium"
        }


# Global memory manager instance
memory_manager = InterviewMemoryManager()


def get_session_memory(session_id: str) -> InterviewMemoryManager:
    """Get or create memory for a specific session"""
    # In a production system, you'd store these in a database or cache
    # For now, return the global instance (you could extend this to be session-specific)
    return memory_manager


def update_session_memory(session_id: str, question: str, answer: str, interviewer_type: str = "TECH") -> None:
    """Update memory for a specific session"""
    memory = get_session_memory(session_id)
    memory.add_exchange(question, answer, interviewer_type)


def get_memory_context(session_id: str) -> str:
    """Get memory context for a specific session"""
    memory = get_session_memory(session_id)
    return memory.get_context_for_next_question()


def clear_session_memory(session_id: str) -> None:
    """Clear memory for a specific session"""
    memory = get_session_memory(session_id)
    memory.clear_memory()
