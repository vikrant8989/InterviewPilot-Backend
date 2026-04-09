from dataclasses import dataclass


DEFAULT_COMPANY_PERSONAS: dict[str, dict] = {
    "Google": {
        "style": "analytical, deep reasoning",
        "focus": ["DSA", "System Design"],
        "difficulty_bias": "high",
        "follow_up_style": "deep probing",
        "difficulty_thresholds": {"good": 7.5, "weak": 4.0},
        "question_type_weights": {"coding": 0.5, "system_design": 0.3, "conceptual": 0.2},
    },
    "Amazon": {
        "style": "structured thinking, tradeoffs, ownership mindset",
        "focus": ["System Design", "Behavioral"],
        "difficulty_bias": "medium",
        "follow_up_style": "practical probing",
        "difficulty_thresholds": {"good": 7.0, "weak": 4.0},
        "question_type_weights": {"coding": 0.35, "system_design": 0.35, "conceptual": 0.3},
    },
    "Stripe": {
        "style": "systems thinking, clarity, edge cases",
        "focus": ["System Design", "APIs", "Monitoring"],
        "difficulty_bias": "high",
        "follow_up_style": "edge-case and reliability probing",
        "difficulty_thresholds": {"good": 7.5, "weak": 4.2},
        "question_type_weights": {"coding": 0.4, "system_design": 0.4, "conceptual": 0.2},
    },
    "Startup": {
        "style": "fast iteration, prioritization, pragmatic engineering",
        "focus": ["End-to-end design", "Scope tradeoffs"],
        "difficulty_bias": "medium",
        "follow_up_style": "scope and impact probing",
        "difficulty_thresholds": {"good": 7.0, "weak": 4.0},
        "question_type_weights": {"coding": 0.3, "system_design": 0.5, "conceptual": 0.2},
    },
    "Custom": {
        "style": "neutral and role-focused",
        "focus": ["Core fundamentals"],
        "difficulty_bias": "medium",
        "follow_up_style": "balanced probing",
        "difficulty_thresholds": {"good": 7.0, "weak": 4.0},
        "question_type_weights": {"coding": 0.34, "system_design": 0.33, "conceptual": 0.33},
    },
}


@dataclass(frozen=True)
class Persona:
    agent_type: str
    company: str
    style: str
    focus: list[str]
    difficulty_bias: str
    follow_up_style: str
    difficulty_thresholds: dict
    question_type_weights: dict


def load_persona(company: str, agent_type: str = "TECH") -> Persona:
    company = (company or "").strip() or "Custom"
    raw = DEFAULT_COMPANY_PERSONAS.get(company, DEFAULT_COMPANY_PERSONAS["Custom"])

    agent_type_norm = (agent_type or "").strip().upper() or "TECH"

    # Agent overlays define behavioral vs technical emphasis.
    if agent_type_norm == "HR":
        style = f"{raw['style']}. HR: evaluate behavioral competencies with STAR, deep probing, and measurable impact."
        focus = ["Behavioral", "Communication", "Leadership", "Ownership", "Conflict", "Stakeholder"]
        question_type_weights = {"behavioral": 0.7, "conceptual": 0.3}
        follow_up_style = "STAR deep probing with clarifying behavioral details"
        difficulty_thresholds = raw["difficulty_thresholds"]
    elif agent_type_norm == "MANAGER":
        style = f"{raw['style']}. Hiring Manager: evaluate scope, tradeoffs, prioritization, decision-making, and impact."
        focus = ["System Design", "Tradeoffs", "Prioritization", "Scope", "Impact", "Leadership"]
        question_type_weights = {"system_design": 0.45, "conceptual": 0.35, "coding": 0.2}
        follow_up_style = "impact-and-decision probing with practical tradeoffs"
        difficulty_thresholds = raw["difficulty_thresholds"]
    else:
        style = raw["style"]
        focus = raw["focus"]
        difficulty_thresholds = raw["difficulty_thresholds"]
        question_type_weights = raw["question_type_weights"]
        follow_up_style = raw["follow_up_style"]

    return Persona(
        agent_type=agent_type_norm,
        company=company,
        style=style,
        focus=focus,
        difficulty_bias=raw["difficulty_bias"],
        follow_up_style=follow_up_style,
        difficulty_thresholds=difficulty_thresholds,
        question_type_weights=question_type_weights,
    )

