"""
Career trajectory analysis.

Analyzes career history for fit signals:
- Company type (product vs services/consulting)
- Title relevance and progression
- Role description analysis for actual ML/production work
- Job-hopping detection
- Industry relevance
"""

from config import (
    CONSULTING_SERVICES_COMPANIES,
    HIGHLY_RELEVANT_TITLES, MODERATELY_RELEVANT_TITLES, IRRELEVANT_TITLES,
    TECH_PRODUCT_INDUSTRIES, SERVICES_INDUSTRIES,
    PRODUCTION_KEYWORDS, ML_WORK_KEYWORDS, NON_AI_WORK_KEYWORDS,
)


def _classify_company(company_name: str, industry: str, company_size: str) -> dict:
    """Classify a company as product, services, or unknown."""
    company_lower = company_name.lower().strip()
    industry_lower = industry.lower().strip() if industry else ""
    
    is_consulting = any(
        c in company_lower for c in CONSULTING_SERVICES_COMPANIES
    )
    is_tech_product = any(
        ind in industry_lower for ind in TECH_PRODUCT_INDUSTRIES
    )
    is_services_industry = any(
        ind in industry_lower for ind in SERVICES_INDUSTRIES
    )
    
    if is_consulting:
        company_type = "consulting"
    elif is_tech_product:
        company_type = "product"
    elif is_services_industry:
        company_type = "services"
    else:
        company_type = "other"
    
    # Company size preference for Series A
    # Smaller companies = more relevant startup experience
    size_score = {
        "1-10": 0.9,
        "11-50": 1.0,     # Ideal startup size
        "51-200": 0.95,
        "201-500": 0.85,
        "501-1000": 0.75,
        "1001-5000": 0.65,
        "5001-10000": 0.55,
        "10001+": 0.45,   # Big corp experience is less relevant
    }.get(company_size, 0.5)
    
    return {
        "type": company_type,
        "is_consulting": is_consulting,
        "is_tech_product": is_tech_product,
        "size_score": size_score,
    }


def _title_word_overlap(title_words: set, reference_words: set) -> float:
    """Calculate word overlap ratio between title and reference."""
    if not reference_words:
        return 0.0
    overlap = title_words & reference_words
    return len(overlap) / len(reference_words)


def _classify_title(title: str) -> tuple[str, float]:
    """
    Classify a job title and return (relevance_level, score).
    Uses both substring matching and word-level overlap for flexibility.
    """
    title_lower = title.lower().strip()
    title_words = set(title_lower.split())
    
    # Check highly relevant titles — substring match first
    for t in HIGHLY_RELEVANT_TITLES:
        if t in title_lower or title_lower in t:
            return "highly_relevant", 1.0
    
    # Word-level overlap for highly relevant (catches 'recommendation systems engineer')
    best_overlap = 0.0
    for t in HIGHLY_RELEVANT_TITLES:
        ref_words = set(t.split())
        overlap = _title_word_overlap(title_words, ref_words)
        best_overlap = max(best_overlap, overlap)
    if best_overlap >= 0.6:  # 60%+ word overlap
        return "highly_relevant", 0.9 + 0.1 * best_overlap
    
    # Check moderately relevant titles
    for t in MODERATELY_RELEVANT_TITLES:
        if t in title_lower or title_lower in t:
            return "moderately_relevant", 0.5
    
    # Word-level overlap for moderately relevant
    best_overlap = 0.0
    for t in MODERATELY_RELEVANT_TITLES:
        ref_words = set(t.split())
        overlap = _title_word_overlap(title_words, ref_words)
        best_overlap = max(best_overlap, overlap)
    if best_overlap >= 0.6:
        return "moderately_relevant", 0.4 + 0.1 * best_overlap
    
    # Check irrelevant titles
    for t in IRRELEVANT_TITLES:
        if t in title_lower or title_lower in t:
            return "irrelevant", 0.0
    
    # Word-level check for key AI/ML words in title
    ai_title_words = {
        'ai', 'ml', 'machine', 'learning', 'data', 'nlp',
        'search', 'ranking', 'recommendation', 'retrieval',
        'scientist', 'research', 'deep', 'intelligence',
    }
    if title_words & ai_title_words:
        return "moderately_relevant", 0.5
    
    return "unknown", 0.25


def _analyze_description(description: str) -> dict:
    """
    Analyze a role description for AI/ML and production signals.
    """
    desc_lower = description.lower() if description else ""
    
    ml_keyword_count = sum(1 for kw in ML_WORK_KEYWORDS if kw in desc_lower)
    production_keyword_count = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc_lower)
    non_ai_keyword_count = sum(1 for kw in NON_AI_WORK_KEYWORDS if kw in desc_lower)
    
    # Is this description about actual ML/AI work?
    # Lowered threshold: even 1 ML keyword is a signal
    is_ml_work = ml_keyword_count >= 1
    # Is this description about production/shipping work?
    is_production = production_keyword_count >= 1
    # Is this description about non-AI work?
    is_non_ai = non_ai_keyword_count >= 2
    
    return {
        "ml_keyword_count": ml_keyword_count,
        "production_keyword_count": production_keyword_count,
        "non_ai_keyword_count": non_ai_keyword_count,
        "is_ml_work": is_ml_work,
        "is_production": is_production,
        "is_non_ai": is_non_ai,
    }


def score_career(candidate: dict) -> dict:
    """
    Score a candidate's career trajectory for fit with the JD.
    
    Returns a dict with:
        - total_score: float (0-1 normalized)
        - title_relevance: best title relevance level
        - company_types: list of company types in career
        - has_production_ml: bool
        - is_consulting_only: bool
        - job_hop_score: float (lower = more hopping)
        - details: human-readable summary
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    
    if not career:
        return {
            "total_score": 0.0,
            "title_relevance": "none",
            "company_types": [],
            "has_production_ml": False,
            "is_consulting_only": False,
            "job_hop_score": 0.5,
            "details": "No career history",
        }
    
    # ── Analyze each job ──────────────────────────────────────────
    job_analyses = []
    company_types = []
    title_scores = []
    has_production_ml = False
    all_consulting = True
    ml_roles_count = 0
    production_roles_count = 0
    total_ml_months = 0
    
    for job in career:
        company_info = _classify_company(
            job.get("company", ""),
            job.get("industry", ""),
            job.get("company_size", ""),
        )
        title_level, title_score = _classify_title(job.get("title", ""))
        desc_analysis = _analyze_description(job.get("description", ""))
        
        duration = job.get("duration_months", 0)
        
        if not company_info["is_consulting"]:
            all_consulting = False
        
        if desc_analysis["is_ml_work"]:
            ml_roles_count += 1
            total_ml_months += duration
        
        if desc_analysis["is_production"]:
            production_roles_count += 1
        
        if desc_analysis["is_ml_work"] and desc_analysis["is_production"]:
            has_production_ml = True
        
        company_types.append(company_info["type"])
        title_scores.append(title_score)
        
        job_analyses.append({
            "company": job.get("company", "?"),
            "title": job.get("title", "?"),
            "duration": duration,
            "company_info": company_info,
            "title_level": title_level,
            "title_score": title_score,
            "desc_analysis": desc_analysis,
        })
    
    # ── Score components ──────────────────────────────────────────
    
    # 1. Title relevance (best title matters most)
    best_title_score = max(title_scores) if title_scores else 0.0
    current_title_level, current_title_score = _classify_title(
        profile.get("current_title", "")
    )
    # Current title matters more
    title_component = 0.6 * current_title_score + 0.4 * best_title_score
    
    # 2. Company type score
    has_product_company = "product" in company_types
    product_ratio = company_types.count("product") / max(len(company_types), 1)
    consulting_ratio = company_types.count("consulting") / max(len(company_types), 1)
    
    company_component = 0.3  # Baseline
    if has_product_company:
        company_component += 0.4 * product_ratio
    if all_consulting:
        company_component = 0.05  # JD explicitly says this is a disqualifier
    else:
        company_component -= 0.2 * consulting_ratio
    company_component = max(0.0, min(1.0, company_component))
    
    # Company size bonus (prefer startup/mid-size experience)
    avg_size_score = sum(
        ja["company_info"]["size_score"] for ja in job_analyses
    ) / max(len(job_analyses), 1)
    company_component = company_component * 0.7 + avg_size_score * 0.3
    
    # 3. ML/AI depth in career
    ml_depth_component = 0.0
    if ml_roles_count > 0:
        ml_depth_component = min(1.0, ml_roles_count / 3.0) * 0.5
        ml_depth_component += min(1.0, total_ml_months / 48.0) * 0.5
    
    # 4. Production experience
    production_component = 0.0
    if has_production_ml:
        production_component = 1.0
    elif production_roles_count > 0:
        production_component = 0.5
    elif ml_roles_count > 0:
        production_component = 0.3
    
    # 5. Job stability (job-hopping detection)
    # JD explicitly says they don't want title-chasers switching every 1.5 years
    if len(career) > 1:
        avg_tenure_months = sum(j.get("duration_months", 0) for j in career) / len(career)
        if avg_tenure_months >= 36:  # 3+ years average = stable
            job_hop_score = 1.0
        elif avg_tenure_months >= 24:  # 2+ years = fine
            job_hop_score = 0.8
        elif avg_tenure_months >= 18:  # 1.5 years = concerning
            job_hop_score = 0.5
        elif avg_tenure_months >= 12:  # 1 year average = red flag
            job_hop_score = 0.3
        else:
            job_hop_score = 0.1
    else:
        job_hop_score = 0.7  # Single job — neutral
    
    # 6. Non-AI work penalty
    non_ai_roles = sum(
        1 for ja in job_analyses if ja["desc_analysis"]["is_non_ai"]
    )
    non_ai_ratio = non_ai_roles / max(len(job_analyses), 1)
    non_ai_penalty = max(0.0, non_ai_ratio - 0.3)  # Penalty kicks in above 30%
    
    # ── Composite career score ────────────────────────────────────
    raw_score = (
        0.30 * title_component +
        0.20 * company_component +
        0.25 * ml_depth_component +
        0.15 * production_component +
        0.10 * job_hop_score
    )
    raw_score -= non_ai_penalty * 0.3
    total_score = max(0.0, raw_score)
    
    # Hard disqualifier: entire career at consulting companies
    if all_consulting and len(career) >= 2:
        total_score *= 0.1  # Massive penalty per JD
    
    # ── Build details ─────────────────────────────────────────────
    details_parts = []
    details_parts.append(f"current: {profile.get('current_title', '?')}")
    if has_production_ml:
        details_parts.append("has production ML")
    if all_consulting:
        details_parts.append("ALL consulting (disqualifier)")
    if ml_roles_count > 0:
        details_parts.append(f"{ml_roles_count} ML roles ({total_ml_months}mo)")
    if non_ai_roles > 0:
        details_parts.append(f"{non_ai_roles} non-AI roles")
    
    best_title_level = "highly_relevant"
    if current_title_score >= 0.8:
        best_title_level = "highly_relevant"
    elif current_title_score >= 0.4:
        best_title_level = "moderately_relevant"
    else:
        best_title_level = "irrelevant"
    
    return {
        "total_score": round(total_score, 4),
        "title_relevance": best_title_level,
        "company_types": company_types,
        "has_production_ml": has_production_ml,
        "is_consulting_only": all_consulting and len(career) >= 2,
        "job_hop_score": round(job_hop_score, 2),
        "details": "; ".join(details_parts),
    }
