"""
Stage 1: Fast pre-filter.

Eliminates obviously unfit candidates using cheap boolean/range checks
before the expensive scoring phase. Goal: reduce 100K → ~5K-15K candidates.
"""

from pipeline.honeypot import detect_honeypot
from config import (
    HARD_MIN_EXPERIENCE, HARD_MAX_EXPERIENCE,
    IRRELEVANT_TITLES, NEGATIVE_DOMAIN_SKILLS,
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, MLOPS_PRODUCTION_SKILLS,
    DATA_ENGINEERING_SKILLS, MODERATELY_RELEVANT_TITLES,
    HIGHLY_RELEVANT_TITLES,
)


def _has_any_tech_relevance(candidate: dict) -> bool:
    """
    Check if the candidate has ANY signal of technical/AI relevance.
    This is a generous filter — we only eliminate people with zero tech signal.
    """
    # Check skills
    skills = candidate.get("skills", [])
    skill_names = {s.get("name", "").lower().strip() for s in skills}
    
    all_relevant_skills = (
        MUST_HAVE_SKILLS | CORE_AI_ML_SKILLS | NLP_IR_SKILLS |
        LLM_FINETUNING_SKILLS | MLOPS_PRODUCTION_SKILLS |
        DATA_ENGINEERING_SKILLS
    )
    
    # Check if any skill name partially matches any relevant skill
    for skill_name in skill_names:
        for relevant in all_relevant_skills:
            if relevant in skill_name or skill_name in relevant:
                return True
    
    # Check current title
    current_title = candidate.get("profile", {}).get("current_title", "").lower().strip()
    all_relevant_titles = HIGHLY_RELEVANT_TITLES | MODERATELY_RELEVANT_TITLES
    for title in all_relevant_titles:
        if title in current_title or current_title in title:
            return True
    
    # Check career history titles
    for job in candidate.get("career_history", []):
        job_title = job.get("title", "").lower().strip()
        for title in all_relevant_titles:
            if title in job_title or job_title in title:
                return True
    
    # Check career descriptions for ML/AI keywords
    from config import ML_WORK_KEYWORDS
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        for keyword in ML_WORK_KEYWORDS:
            if keyword in desc:
                return True
    
    # Check education field
    for edu in candidate.get("education", []):
        field = edu.get("field_of_study", "").lower()
        if any(term in field for term in [
            "computer science", "machine learning", "artificial intelligence",
            "data science", "information technology", "software",
            "statistics", "mathematics", "computational",
        ]):
            return True
    
    # Check skill assessment scores (redrob signals)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        for skill_name in assessments:
            skill_lower = skill_name.lower()
            for relevant in all_relevant_skills:
                if relevant in skill_lower or skill_lower in relevant:
                    return True
    
    return False


def _count_negative_domain_skills(candidate: dict) -> int:
    """Count how many explicitly negative-domain skills the candidate has."""
    skills = candidate.get("skills", [])
    count = 0
    for skill in skills:
        skill_name = skill.get("name", "").lower().strip()
        for neg in NEGATIVE_DOMAIN_SKILLS:
            if neg in skill_name or skill_name in neg:
                count += 1
                break
    return count


def prefilter(candidate: dict) -> tuple[bool, str]:
    """
    Fast pre-filter check. Returns (passes, reason_if_filtered).
    
    This is deliberately GENEROUS — we'd rather let some irrelevant
    candidates through than miss a good one. The scorer will handle
    fine-grained differentiation.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    
    # ── Check 1: Experience range ─────────────────────────────────
    years_exp = profile.get("years_of_experience", 0)
    if years_exp < HARD_MIN_EXPERIENCE:
        return False, f"Too little experience: {years_exp} years"
    if years_exp > HARD_MAX_EXPERIENCE:
        return False, f"Too much experience: {years_exp} years"
    
    # ── Check 2: Must have career history ─────────────────────────
    if not career:
        return False, "No career history"
    
    # ── Check 3: Honeypot detection ───────────────────────────────
    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return False, f"Honeypot detected: {honeypot_reason}"
    
    # ── Check 4: Any tech relevance ──────────────────────────────
    # Only filter out candidates with ZERO tech/AI signal
    if not _has_any_tech_relevance(candidate):
        # Double-check: if they have a very irrelevant title AND
        # mostly negative domain skills, filter them out
        current_title = profile.get("current_title", "").lower().strip()
        is_irrelevant_title = any(
            t in current_title or current_title in t
            for t in IRRELEVANT_TITLES
        )
        neg_skills = _count_negative_domain_skills(candidate)
        total_skills = len(candidate.get("skills", []))
        
        if is_irrelevant_title and (total_skills == 0 or neg_skills / max(total_skills, 1) > 0.5):
            return False, f"No tech relevance: title='{profile.get('current_title')}', {neg_skills}/{total_skills} negative skills"
        
        # Even without explicit tech skills, some candidates might be
        # valuable. Only filter if they have a purely irrelevant title
        # and zero tech signals
        if is_irrelevant_title and total_skills > 0:
            # Let them through — the scorer will give them a low score
            pass
        elif not is_irrelevant_title:
            pass
        else:
            return False, f"No tech relevance and no skills: title='{profile.get('current_title')}'"
    
    return True, "Passed pre-filter"
