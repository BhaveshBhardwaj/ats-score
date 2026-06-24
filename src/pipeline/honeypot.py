"""
Honeypot detection module.

Identifies candidates with subtly impossible profiles:
- Duration vs timeline mismatches
- Impossible experience claims
- Skill inflation (expert with 0 duration)
- Career timeline impossibilities
"""

from datetime import datetime, date


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Check if a candidate is a honeypot with an impossible profile.
    
    Returns:
        (is_honeypot: bool, reason: str)
    """
    flags = []
    score = 0  # Accumulate suspicion points
    
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    
    # ── Check 1: Skill inflation ──────────────────────────────────
    # "Expert" proficiency with very low or zero duration
    expert_zero_duration = 0
    expert_low_duration = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        if prof == "expert" and dur == 0:
            expert_zero_duration += 1
        elif prof == "expert" and dur < 3:
            expert_low_duration += 1
        elif prof == "advanced" and dur == 0:
            expert_zero_duration += 1
    
    if expert_zero_duration >= 3:
        flags.append(f"{expert_zero_duration} expert/advanced skills with 0 months duration")
        score += expert_zero_duration * 2
    elif expert_zero_duration >= 1:
        score += expert_zero_duration
    
    if expert_low_duration >= 5:
        flags.append(f"{expert_low_duration} expert skills with <3 months duration")
        score += expert_low_duration
    
    # ── Check 2: Career timeline impossibilities ──────────────────
    years_exp = profile.get("years_of_experience", 0)
    
    # Check if total career months drastically exceed or fall short of years_of_experience
    total_career_months = sum(
        job.get("duration_months", 0) for job in career
    )
    expected_months = years_exp * 12
    
    if total_career_months > 0 and expected_months > 0:
        ratio = total_career_months / expected_months
        if ratio > 2.5:  # Career months are 2.5x+ the stated experience
            flags.append(f"Career months ({total_career_months}) >> stated experience ({years_exp} yrs)")
            score += 3
        elif ratio < 0.3 and years_exp > 3:  # Very few career months for stated experience
            flags.append(f"Career months ({total_career_months}) << stated experience ({years_exp} yrs)")
            score += 2
    
    # ── Check 3: Duration vs date range mismatch ──────────────────
    for job in career:
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        claimed_months = job.get("duration_months", 0)
        
        if start_str and end_str:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                
                if actual_months > 0 and claimed_months > 0:
                    diff = abs(actual_months - claimed_months)
                    if diff > 24:  # More than 2 years off
                        flags.append(
                            f"Job at {job.get('company', '?')}: claimed {claimed_months}mo "
                            f"but dates show ~{actual_months}mo"
                        )
                        score += 3
                    elif diff > 12:  # More than 1 year off
                        score += 1
                
                # End before start
                if end < start:
                    flags.append(f"Job at {job.get('company', '?')}: end date before start date")
                    score += 4
                    
            except (ValueError, TypeError):
                pass
        
        # Current job but has end_date, or not current but no end_date
        is_current = job.get("is_current", False)
        if is_current and end_str is not None:
            # Minor inconsistency, but could be data format
            score += 0.5
    
    # ── Check 4: Impossible years of experience ──────────────────
    # If they have 8 years experience but graduated 3 years ago
    if education:
        latest_grad = max(
            (edu.get("end_year", 0) for edu in education),
            default=0
        )
        if latest_grad > 0:
            years_since_grad = 2026 - latest_grad
            if years_exp > years_since_grad + 3:  # Allow some overlap
                flags.append(
                    f"Claims {years_exp} yrs exp but graduated in {latest_grad} "
                    f"({years_since_grad} yrs ago)"
                )
                score += 2
    
    # ── Check 5: Too many expert skills for the experience ────────
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 10 and years_exp < 5:
        flags.append(f"{expert_count} expert skills with only {years_exp} yrs experience")
        score += 3
    elif expert_count >= 8 and years_exp < 3:
        flags.append(f"{expert_count} expert skills with only {years_exp} yrs experience")
        score += 3
    
    # ── Check 6: Suspiciously perfect engagement metrics ──────────
    signals = candidate.get("redrob_signals", {})
    perfect_count = 0
    if signals.get("recruiter_response_rate", 0) >= 0.99:
        perfect_count += 1
    if signals.get("interview_completion_rate", 0) >= 0.99:
        perfect_count += 1
    if signals.get("offer_acceptance_rate", 0) >= 0.99:
        perfect_count += 1
    if signals.get("profile_completeness_score", 0) >= 99.5:
        perfect_count += 1
    
    if perfect_count >= 3:
        score += 1  # Mildly suspicious but not conclusive
    
    # ── Check 7: Title-description mismatch honeypots ─────────────
    # Titles that don't match their descriptions at all
    mismatch_count = 0
    for job in career:
        title_lower = job.get("title", "").lower()
        desc_lower = job.get("description", "").lower()
        
        # Marketing Manager doing mechanical engineering
        if "marketing" in title_lower and ("solidworks" in desc_lower or "cad" in desc_lower or "ansys" in desc_lower):
            mismatch_count += 1
        # HR Manager doing accounting
        if "hr" in title_lower and ("accounting" in desc_lower or "tax filings" in desc_lower or "financial reporting" in desc_lower):
            mismatch_count += 1
        # Accountant doing customer support
        if "accountant" in title_lower and ("support agent" in desc_lower or "tier-1" in desc_lower):
            mismatch_count += 1
        # Operations Manager doing content writing
        if "operations" in title_lower and ("seo strategy" in desc_lower or "editorial calendar" in desc_lower):
            mismatch_count += 1
        # Customer Support doing business analysis
        if "customer support" in title_lower and ("consulting firm" in desc_lower or "business diagnostics" in desc_lower):
            mismatch_count += 1
    
    if mismatch_count >= 2:
        flags.append(f"{mismatch_count} jobs with title-description mismatch")
        score += 2
    
    # ── Check 8: Compound impossibility detection ───────────────
    # Multiple individually-plausible signals that compound
    compound_score = 0
    
    # Skill diversity vs career focus mismatch
    # If someone has skills in 5+ unrelated domains, suspicious
    skill_domains = set()
    for skill in skills:
        name = skill.get("name", "").lower()
        if any(kw in name for kw in ["python", "java", "sql", "javascript"]):
            skill_domains.add("programming")
        elif any(kw in name for kw in ["machine learning", "deep learning", "neural", "ai"]):
            skill_domains.add("ml")
        elif any(kw in name for kw in ["marketing", "seo", "brand"]):
            skill_domains.add("marketing")
        elif any(kw in name for kw in ["accounting", "tax", "audit", "finance"]):
            skill_domains.add("accounting")
        elif any(kw in name for kw in ["design", "photoshop", "figma", "sketch"]):
            skill_domains.add("design")
        elif any(kw in name for kw in ["hr", "recruitment", "payroll"]):
            skill_domains.add("hr")
        elif any(kw in name for kw in ["sales", "crm", "lead"]):
            skill_domains.add("sales")
    
    non_tech_domains = skill_domains - {"programming", "ml"}
    if len(non_tech_domains) >= 3 and "ml" in skill_domains:
        compound_score += 2
        flags.append(f"Skills span {len(skill_domains)} unrelated domains")
    
    # All skills have identical endorsement counts (manufactured)
    endorsement_counts = [s.get("endorsements", 0) for s in skills if s.get("endorsements", 0) > 0]
    if len(endorsement_counts) >= 5 and len(set(endorsement_counts)) == 1:
        compound_score += 1
        flags.append("All skills have identical endorsement counts")
    
    # All skills have identical duration (manufactured)
    duration_counts = [s.get("duration_months", 0) for s in skills if s.get("duration_months", 0) > 0]
    if len(duration_counts) >= 5 and len(set(duration_counts)) == 1:
        compound_score += 1
        flags.append("All skills have identical durations")
    
    score += compound_score
    
    # ── Final decision ────────────────────────────────────────────
    is_honeypot = score >= 5
    reason = "; ".join(flags) if flags else "No honeypot signals"
    
    return is_honeypot, reason
