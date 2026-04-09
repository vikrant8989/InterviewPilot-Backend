from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, Any
import json

from app.services.persona_service import DEFAULT_COMPANY_PERSONAS


class LangChainPersonaTemplates:
    """Convert persona definitions to LangChain prompt templates"""
    
    @staticmethod
    def create_interviewer_prompt_template(persona_data: Dict[str, Any], agent_type: str) -> ChatPromptTemplate:
        """Create a LangChain prompt template for a specific persona"""
        
        base_style = persona_data["style"]
        focus_areas = persona_data["focus"]
        difficulty_bias = persona_data["difficulty_bias"]
        follow_up_style = persona_data["follow_up_style"]
        
        # Agent-specific modifications
        if agent_type == "HR":
            agent_modification = """
            HR AGENT FOCUS:
            - Evaluate behavioral competencies using STAR method
            - Focus on communication, leadership, ownership, conflict resolution
            - Ask about past experiences and measurable impact
            - Probe for specific examples and behavioral patterns
            """
            question_focus = "behavioral questions, situational judgment, past experiences"
        elif agent_type == "MANAGER":
            agent_modification = """
            MANAGER AGENT FOCUS:
            - Evaluate scope definition, tradeoffs, prioritization
            - Focus on decision-making process and impact
            - Assess leadership and stakeholder management
            - Probe for strategic thinking and business impact
            """
            question_focus = "system design, tradeoffs, prioritization, leadership scenarios"
        else:  # TECH
            agent_modification = """
            TECH AGENT FOCUS:
            - Evaluate technical depth and problem-solving
            - Focus on algorithms, data structures, system design
            - Assess coding skills and technical decision-making
            - Probe for technical tradeoffs and best practices
            """
            question_focus = "coding problems, technical concepts, system design"
        
        system_prompt = f"""You are an AI interviewer for {persona_data.get('company', 'a tech company')}.

INTERVIEW STYLE: {base_style}

FOCUS AREAS: {', '.join(focus_areas)}

DIFFICULTY BIAS: {difficulty_bias}

FOLLOW-UP STYLE: {follow_up_style}

{agent_modification}

QUESTION TYPES TO FOCUS ON: {question_focus}

GUIDELINES:
1. Ask clear, specific questions tailored to the role and company
2. Adapt difficulty based on candidate responses
3. Use the follow-up style to probe deeper when needed
4. Maintain the company's interview culture and values
5. Ensure questions are practical and relevant to real-world scenarios
6. Consider the candidate's previous responses when formulating new questions"""

        human_prompt = """Based on the following context, generate the next interview question:

COMPANY: {company}
TARGET ROLE: {target_role}
DIFFICULTY LEVEL: {difficulty}
QUESTION TYPE: {question_type}
CANDIDATE'S PREVIOUS ANSWER: {previous_answer}
INTERVIEW CONTEXT: {context}
MEMORY SUMMARY: {memory_summary}

Generate a specific, engaging interview question that follows the guidelines above:"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    @staticmethod
    def create_evaluation_prompt_template(persona_data: Dict[str, Any], agent_type: str) -> ChatPromptTemplate:
        """Create a prompt template for evaluating candidate answers"""
        
        base_style = persona_data["style"]
        focus_areas = persona_data["focus"]
        difficulty_thresholds = persona_data["difficulty_thresholds"]
        
        system_prompt = f"""You are evaluating a candidate's response in a technical interview.

INTERVIEW STYLE: {base_style}
FOCUS AREAS: {', '.join(focus_areas)}
DIFFICULTY THRESHOLDS: Good >= {difficulty_thresholds['good']}, Weak <= {difficulty_thresholds['weak']}

EVALUATION CRITERIA:
1. Technical accuracy and depth
2. Problem-solving approach
3. Communication clarity
4. Consideration of edge cases and tradeoffs
5. Relevance to the question asked

Provide a score from 1-10 for each criterion and an overall score."""

        human_prompt = """Evaluate the following answer:

QUESTION: {question}
CANDIDATE'S ANSWER: {answer}
EXPECTED DIFFICULTY: {difficulty}
FOCUS AREAS: {focus_areas}

Provide your evaluation in JSON format:
{{
    "technical_accuracy": <score 1-10>,
    "problem_solving": <score 1-10>,
    "communication": <score 1-10>,
    "tradeoffs_consideration": <score 1-10>,
    "relevance": <score 1-10>,
    "overall": <average score>,
    "strengths": ["list of strengths"],
    "weaknesses": ["list of areas for improvement"],
    "feedback": "constructive feedback for the candidate"
}}"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
    
    @staticmethod
    def get_all_persona_templates() -> Dict[str, Dict[str, ChatPromptTemplate]]:
        """Get all persona templates organized by company and agent type"""
        templates = {}
        
        for company, persona_data in DEFAULT_COMPANY_PERSONAS.items():
            templates[company] = {}
            for agent_type in ["HR", "TECH", "MANAGER"]:
                templates[company][agent_type] = {
                    "interviewer": LangChainPersonaTemplates.create_interviewer_prompt_template(
                        persona_data, agent_type
                    ),
                    "evaluator": LangChainPersonaTemplates.create_evaluation_prompt_template(
                        persona_data, agent_type
                    )
                }
        
        return templates


# Global instance of all templates
persona_templates = LangChainPersonaTemplates.get_all_persona_templates()
