"""
Semantic skill matching engine.

Maps candidate skills to JD requirements using a taxonomy + fuzzy matching
approach. No external ML models needed — uses carefully crafted skill groups
and string similarity.
"""

from difflib import SequenceMatcher
from config import (
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, MLOPS_PRODUCTION_SKILLS,
    DATA_ENGINEERING_SKILLS, CLOUD_SKILLS,
    NEGATIVE_DOMAIN_SKILLS,
)

# Proficiency level weights
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}

# Skill category weights (how important each category is for this JD)
CATEGORY_WEIGHTS = {
    "must_have": 3.0,      # Embeddings, vector DBs, ranking, Python
    "core_ai_ml": 2.0,     # General ML/DL skills
    "nlp_ir": 2.5,         # NLP & Information Retrieval
    "llm_finetuning": 1.5, # Nice-to-have: LLM fine-tuning
    "mlops_production": 1.5, # Production engineering signals
    "data_engineering": 0.8, # Adjacent and useful
    "cloud": 0.5,          # Minor bonus
    "negative": -2.0,      # Wrong domain
}

# Map category names to their skill sets
SKILL_CATEGORIES = {
    "must_have": MUST_HAVE_SKILLS,
    "core_ai_ml": CORE_AI_ML_SKILLS,
    "nlp_ir": NLP_IR_SKILLS,
    "llm_finetuning": LLM_FINETUNING_SKILLS,
    "mlops_production": MLOPS_PRODUCTION_SKILLS,
    "data_engineering": DATA_ENGINEERING_SKILLS,
    "cloud": CLOUD_SKILLS,
    "negative": NEGATIVE_DOMAIN_SKILLS,
}


def _fuzzy_match(skill_name: str, category_skills: set, threshold: float = 0.75) -> bool:
    """
    Check if a skill name matches any skill in a category using
    exact substring matching. Dropped SequenceMatcher for performance.
    """
    skill_lower = skill_name.lower().strip()
    
    # Exact substring match (fast path)
    for cat_skill in category_skills:
        if cat_skill in skill_lower or skill_lower in cat_skill:
            return True
            
    return False


def _categorize_skill(skill_name: str) -> tuple[str, float]:
    """
    Determine which category a skill belongs to and return
    (category_name, category_weight).
    
    Returns ("uncategorized", 0.0) if no match.
    """
    # Check categories in priority order
    priority_order = [
        "must_have", "nlp_ir", "core_ai_ml", "llm_finetuning",
        "mlops_production", "data_engineering", "cloud", "negative"
    ]
    
    for category in priority_order:
        if _fuzzy_match(skill_name, SKILL_CATEGORIES[category]):
            return category, CATEGORY_WEIGHTS[category]
    
    return "uncategorized", 0.0


def score_skills(candidate: dict) -> dict:
    """
    Score a candidate's skills against the JD requirements.
    
    Returns a dict with:
        - total_score: float (0-1 normalized)
        - category_breakdown: dict of category → count
        - matched_skills: list of (skill_name, category, weight)
        - negative_skills: list of skill names in wrong domain
        - must_have_count: number of must-have skills matched
        - details: human-readable summary
    """
    skills = candidate.get("skills", [])
    
    if not skills:
        return {
            "total_score": 0.0,
            "category_breakdown": {},
            "matched_skills": [],
            "negative_skills": [],
            "must_have_count": 0,
            "core_count": 0,
            "details": "No skills listed",
        }
    
    # Score each skill
    matched_skills = []
    negative_skills = []
    category_counts = {}
    raw_score = 0.0
    
    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "beginner")
        duration_months = skill.get("duration_months", 0)
        endorsements = skill.get("endorsements", 0)
        
        category, cat_weight = _categorize_skill(name)
        
        if category == "uncategorized":
            continue
        
        if category == "negative":
            negative_skills.append(name)
            raw_score += cat_weight  # cat_weight is negative
            continue
        
        # Proficiency weight
        prof_weight = PROFICIENCY_WEIGHTS.get(proficiency, 0.2)
        
        # Duration bonus (longer use = more credible)
        duration_weight = min(duration_months / 36.0, 1.5)  # Cap at 1.5x for 3+ years
        
        # Endorsement bonus (social proof, but capped)
        endorsement_weight = 1.0 + min(endorsements / 50.0, 0.3)  # Up to 1.3x
        
        # Combined skill score
        skill_score = cat_weight * prof_weight * duration_weight * endorsement_weight
        raw_score += skill_score
        
        matched_skills.append((name, category, round(skill_score, 3)))
        category_counts[category] = category_counts.get(category, 0) + 1
    
    # Validate against assessment scores
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    assessment_bonus = 0.0
    for skill_name, score in assessments.items():
        category, _ = _categorize_skill(skill_name)
        if category in ("must_have", "nlp_ir", "core_ai_ml", "llm_finetuning"):
            if score >= 70:
                assessment_bonus += 0.5
            elif score >= 50:
                assessment_bonus += 0.2
            elif score < 30:
                assessment_bonus -= 0.3  # Low score on claimed skill = credibility hit
    
    raw_score += assessment_bonus
    
    # Normalize to 0-1
    # Theoretical max depends on number of skills, but a realistic good score
    # for this JD might be ~15-25 raw points
    max_expected = 20.0
    total_score = max(0.0, raw_score / max_expected)
    
    # Count key categories
    must_have_count = category_counts.get("must_have", 0)
    core_count = (
        category_counts.get("core_ai_ml", 0) +
        category_counts.get("nlp_ir", 0)
    )
    
    # Build details string
    top_matches = sorted(matched_skills, key=lambda x: -x[2])[:5]
    details_parts = []
    if must_have_count > 0:
        details_parts.append(f"{must_have_count} must-have skills")
    if core_count > 0:
        details_parts.append(f"{core_count} core AI/NLP skills")
    if negative_skills:
        details_parts.append(f"{len(negative_skills)} wrong-domain skills")
    if top_matches:
        details_parts.append(
            "top: " + ", ".join(f"{s[0]}({s[1]})" for s in top_matches[:3])
        )
    
    return {
        "total_score": round(total_score, 4),
        "category_breakdown": category_counts,
        "matched_skills": matched_skills,
        "negative_skills": negative_skills,
        "must_have_count": must_have_count,
        "core_count": core_count,
        "details": "; ".join(details_parts) if details_parts else "No relevant skills",
    }
