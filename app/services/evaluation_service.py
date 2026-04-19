from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    rule_score_json: dict
    llm_score_json: dict
    final_score_json: dict
    strengths: list[str]
    weaknesses: list[str]
    improvements: list[str]


def evaluate_answer(*, persona, question_text: str, answer_text: str) -> EvaluationResult:
    """
    Simple synchronous evaluation without any API calls.
    """
    answer_text = answer_text or ""
    
    # Length score
    length_score = min(len(answer_text) / 800.0, 1.0) * 10.0
    
    # Keyword coverage
    keyword_score = 0.0
    matched_keywords = []
    focus_keywords = persona.get("focus", [])
    for kw in focus_keywords:
        if kw.lower() in answer_text.lower():
            keyword_score += 2.0
            matched_keywords.append(kw)
    keyword_score = min(keyword_score, 6.0)
    
    # Structure analysis
    structure_score = 0.0
    structure_indicators = ["approach", "assumption", "tradeoff", "consider"]
    matched_indicators = []
    for indicator in structure_indicators:
        if indicator in answer_text.lower():
            structure_score += 1.0
            matched_indicators.append(indicator)
    structure_score = min(structure_score, 4.0)
    
    overall_unscaled = (length_score * 0.3 + keyword_score * 0.4 + structure_score * 0.3)
    # Scale to max of 10 (current max possible is 6.6: 10*0.3 + 6*0.4 + 4*0.3)
    overall = round(overall_unscaled * (10 / 6.6), 2)
    
    print(f"[Evaluation] Answer length: {len(answer_text)} chars, Length score: {length_score:.2f}")
    print(f"[Evaluation] Focus keywords: {focus_keywords}, Matched: {matched_keywords}, Keyword score: {keyword_score:.2f}")
    print(f"[Evaluation] Structure indicators matched: {matched_indicators}, Structure score: {structure_score:.2f}")
    print(f"[Evaluation] Overall score: {overall}")
    
    # Generate feedback
    strengths = []
    weaknesses = []
    improvements = []
    
    if length_score >= 7:
        strengths.append("Good answer depth")
    else:
        weaknesses.append("Answer lacks depth")
        improvements.append("Add more detail")
    
    if keyword_score > 0:
        strengths.append(f"Good coverage of: {', '.join(matched_keywords)}")
    
    if structure_score >= 2:
        strengths.append("Well-structured")
    else:
        weaknesses.append("Needs better structure")
        improvements.append("Use clear approach and conclusion")
    
    return EvaluationResult(
        rule_score_json={
            "overall": overall,
            "length_score": round(length_score, 2),
            "keyword_score": round(keyword_score, 2),
            "structure_score": round(structure_score, 2),
        },
        llm_score_json={
            "overall": overall,
            "clarity": round(length_score, 2),
            "correctness": round(keyword_score + structure_score, 2),
        },
        final_score_json={
            "overall": overall,
            "length_score": round(length_score, 2),
            "keyword_score": round(keyword_score, 2),
            "structure_score": round(structure_score, 2),
            "clarity": round(length_score, 2),
            "correctness": round(keyword_score + structure_score, 2),
        },
        strengths=strengths or ["Good attempt"],
        weaknesses=weaknesses or ["Can be improved"],
        improvements=improvements or ["Add more detail"],
    )


class LangChainEvaluator:
    """LangChain-compatible evaluator wrapper"""
    
    def rule_based_evaluate(self, *, persona: dict, question_text: str, answer_text: str) -> dict:
        """Evaluate answer using rule-based logic, returns dict for LangGraph compatibility"""
        result = evaluate_answer(
            persona=persona,
            question_text=question_text,
            answer_text=answer_text
        )
        return result.final_score_json


# Singleton instance for LangGraph compatibility
langchain_evaluator = LangChainEvaluator()
