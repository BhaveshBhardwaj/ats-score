"""
Composite scorer.

Combines career, skills, experience, behavioral, and location scores
into a single composite score with configurable weights.
"""

from config import (
    WEIGHTS,
    IDEAL_EXPERIENCE_MIN, IDEAL_EXPERIENCE_MAX,
    ACCEPTABLE_EXPERIENCE_MIN, ACCEPTABLE_EXPERIENCE_MAX,
)
from pipeline.skills import score_skills
from pipeline.career import score_career
from pipeline.behavioral import score_behavioral, score_location


def _score_experience(candidate: dict) -> dict:
    """
    Score years of experience fit for the JD.
    JD says 5-9 years ideal, but open to strong candidates outside.
    """
    years = candidate.get("profile", {}).get("years_of_experience", 0)
    
    if IDEAL_EXPERIENCE_MIN <= years <= IDEAL_EXPERIENCE_MAX:
        score = 1.0
        details = f"{years} yrs (ideal range)"
    elif ACCEPTABLE_EXPERIENCE_MIN <= years < IDEAL_EXPERIENCE_MIN:
        # Slightly under — linear decay
        score = 0.5 + 0.5 * (years - ACCEPTABLE_EXPERIENCE_MIN) / (IDEAL_EXPERIENCE_MIN - ACCEPTABLE_EXPERIENCE_MIN)
        details = f"{years} yrs (below ideal but acceptable)"
    elif IDEAL_EXPERIENCE_MAX < years <= ACCEPTABLE_EXPERIENCE_MAX:
        # Slightly over — linear decay
        score = 1.0 - 0.4 * (years - IDEAL_EXPERIENCE_MAX) / (ACCEPTABLE_EXPERIENCE_MAX - IDEAL_EXPERIENCE_MAX)
        details = f"{years} yrs (above ideal but acceptable)"
    elif years < ACCEPTABLE_EXPERIENCE_MIN:
        score = 0.2
        details = f"{years} yrs (too junior)"
    else:
        score = 0.3
        details = f"{years} yrs (too senior)"
    
    return {
        "total_score": round(max(0.0, score), 4),
        "years": years,
        "details": details,
    }


def score_candidate(candidate: dict) -> dict:
    """
    Compute the full composite score for a single candidate.
    
    Returns a dict with all dimension scores and the final composite.
    """
    # Score each dimension
    skills_result = score_skills(candidate)
    career_result = score_career(candidate)
    experience_result = _score_experience(candidate)
    behavioral_result = score_behavioral(candidate)
    location_result = score_location(candidate)
    
    # Weighted composite
    composite = (
        WEIGHTS["career_fit"] * career_result["total_score"] +
        WEIGHTS["skills_relevance"] * skills_result["total_score"] +
        WEIGHTS["experience_quality"] * experience_result["total_score"] +
        WEIGHTS["behavioral"] * behavioral_result["total_score"] +
        WEIGHTS["location_fit"] * location_result["total_score"]
    )
    
    # ── Apply multipliers for strong signals ──────────────────────
    
    # Production ML experience is a major differentiator
    if career_result["has_production_ml"]:
        composite *= 1.15
    
    # Must-have skills bonus
    if skills_result["must_have_count"] >= 3:
        composite *= 1.10
    elif skills_result["must_have_count"] >= 1:
        composite *= 1.05
    
    # Consulting-only penalty (from career module, but reinforce here)
    if career_result["is_consulting_only"]:
        composite *= 0.3
    
    # Title relevance boost/penalty
    if career_result["title_relevance"] == "highly_relevant":
        composite *= 1.10
    elif career_result["title_relevance"] == "irrelevant":
        composite *= 0.5
    
    # We removed the cap at 1.0 to allow continuous scoring
    # composite = min(1.0, composite)
    
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "composite_score": round(composite, 4),
        "skills": skills_result,
        "career": career_result,
        "experience": experience_result,
        "behavioral": behavioral_result,
        "location": location_result,
    }
