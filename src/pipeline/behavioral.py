"""
Behavioral signal scoring.

Scores candidates on availability, responsiveness, reliability,
and platform engagement using Redrob behavioral signals.
"""

from datetime import datetime, date
from config import (
    RESPONSE_RATE_GOOD, RESPONSE_RATE_BAD,
    RESPONSE_TIME_GOOD_HOURS, RESPONSE_TIME_BAD_HOURS,
    NOTICE_PERIOD_IDEAL, NOTICE_PERIOD_OK, NOTICE_PERIOD_BAD,
    INACTIVE_DAYS_THRESHOLD, INACTIVE_DAYS_WARNING,
    SALARY_IDEAL_MIN, SALARY_IDEAL_MAX, SALARY_HARD_MAX,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS, PREFERRED_COUNTRY,
)

# Reference date for recency calculations
REFERENCE_DATE = date(2026, 6, 1)


def _days_since(date_str: str) -> int:
    """Calculate days between a date string and reference date."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except (ValueError, TypeError):
        return 999  # Treat missing dates as very old


def _score_availability(signals: dict) -> float:
    """Score availability signals (0-1)."""
    score = 0.0
    
    # Open to work flag (strong positive signal)
    if signals.get("open_to_work_flag", False):
        score += 0.3
    
    # Recency of activity
    last_active = signals.get("last_active_date", "")
    days_inactive = _days_since(last_active)
    
    if days_inactive <= 7:
        score += 0.3  # Very recently active
    elif days_inactive <= 30:
        score += 0.25
    elif days_inactive <= 60:
        score += 0.15
    elif days_inactive <= INACTIVE_DAYS_WARNING:
        score += 0.05
    elif days_inactive <= INACTIVE_DAYS_THRESHOLD:
        score += 0.0  # Neutral
    else:
        score -= 0.15  # Inactive for 6+ months — major concern
    
    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= RESPONSE_RATE_GOOD:
        score += 0.25
    elif response_rate >= 0.3:
        score += 0.15
    elif response_rate >= RESPONSE_RATE_BAD:
        score += 0.05
    else:
        score -= 0.1  # Very low response rate
    
    # Response time
    response_time = signals.get("avg_response_time_hours", 999)
    if response_time <= RESPONSE_TIME_GOOD_HOURS:
        score += 0.15
    elif response_time <= 48:
        score += 0.10
    elif response_time <= RESPONSE_TIME_BAD_HOURS:
        score += 0.0
    else:
        score -= 0.05
    
    return max(0.0, score)


def _score_reliability(signals: dict) -> float:
    """Score reliability signals (0-1)."""
    score = 0.0
    
    # Interview completion rate
    completion = signals.get("interview_completion_rate", 0)
    if completion >= 0.9:
        score += 0.35
    elif completion >= 0.7:
        score += 0.25
    elif completion >= 0.5:
        score += 0.15
    else:
        score += 0.05
    
    # Offer acceptance rate
    acceptance = signals.get("offer_acceptance_rate", -1)
    if acceptance >= 0:  # Has offer history
        if acceptance >= 0.7:
            score += 0.25
        elif acceptance >= 0.4:
            score += 0.15
        else:
            score += 0.05
    else:
        score += 0.1  # No offer history — neutral
    
    # Notice period
    notice = signals.get("notice_period_days", 90)
    if notice <= NOTICE_PERIOD_IDEAL:
        score += 0.25  # Sub-30 is what the JD says they'd love
    elif notice <= NOTICE_PERIOD_OK:
        score += 0.15  # Up to 60 is fine
    elif notice <= NOTICE_PERIOD_BAD:
        score += 0.05  # 60-90 is ok but bar gets higher
    else:
        score -= 0.05  # 90+ is a negative per the JD
    
    # Verification signals
    verified_count = 0
    if signals.get("verified_email", False):
        verified_count += 1
    if signals.get("verified_phone", False):
        verified_count += 1
    if signals.get("linkedin_connected", False):
        verified_count += 1
    score += verified_count * 0.05  # Small bonus per verification
    
    return max(0.0, score)


def _score_market_validation(signals: dict) -> float:
    """Score market validation signals (0-1)."""
    score = 0.0
    
    # Saved by recruiters (social proof)
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 10:
        score += 0.3
    elif saved >= 5:
        score += 0.2
    elif saved >= 1:
        score += 0.1
    
    # Profile views (visibility)
    views = signals.get("profile_views_received_30d", 0)
    if views >= 20:
        score += 0.25
    elif views >= 10:
        score += 0.15
    elif views >= 3:
        score += 0.1
    
    # Search appearances
    appearances = signals.get("search_appearance_30d", 0)
    if appearances >= 100:
        score += 0.2
    elif appearances >= 50:
        score += 0.15
    elif appearances >= 10:
        score += 0.1
    
    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0)
    if completeness >= 90:
        score += 0.15
    elif completeness >= 70:
        score += 0.1
    elif completeness >= 50:
        score += 0.05
    
    # Connection count (mild network signal)
    connections = signals.get("connection_count", 0)
    if connections >= 500:
        score += 0.1
    elif connections >= 200:
        score += 0.05
    
    return max(0.0, score)


def _score_technical_signals(signals: dict) -> float:
    """Score technical activity signals (0-1)."""
    score = 0.0
    
    # GitHub activity (very important for an AI engineer role)
    github = signals.get("github_activity_score", -1)
    if github >= 0:  # Has GitHub linked
        if github >= 70:
            score += 0.6
        elif github >= 50:
            score += 0.45
        elif github >= 30:
            score += 0.3
        elif github >= 10:
            score += 0.15
        else:
            score += 0.05
    else:
        score += 0.0  # No GitHub — not penalized but no bonus
    
    # Skill assessments completed
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_score = sum(assessments.values()) / len(assessments)
        if avg_score >= 70:
            score += 0.3
        elif avg_score >= 50:
            score += 0.2
        elif avg_score >= 30:
            score += 0.1
        
        # Bonus for number of assessments completed
        score += min(len(assessments) * 0.05, 0.1)
    
    return max(0.0, score)


def score_behavioral(candidate: dict) -> dict:
    """
    Score a candidate's behavioral signals.
    
    Returns a dict with:
        - total_score: float (0-1 normalized)
        - availability: float (0-1)
        - reliability: float (0-1)
        - market_validation: float (0-1)
        - technical_signals: float (0-1)
        - details: human-readable summary
    """
    signals = candidate.get("redrob_signals", {})
    
    if not signals:
        return {
            "total_score": 0.3,  # Neutral baseline
            "availability": 0.3,
            "reliability": 0.3,
            "market_validation": 0.0,
            "technical_signals": 0.0,
            "details": "No behavioral signals",
        }
    
    availability = _score_availability(signals)
    reliability = _score_reliability(signals)
    market = _score_market_validation(signals)
    technical = _score_technical_signals(signals)
    
    # Weighted combination
    total = (
        0.35 * availability +
        0.25 * reliability +
        0.15 * market +
        0.25 * technical
    )
    
    # Build details
    details_parts = []
    
    # Activity status
    last_active = signals.get("last_active_date", "")
    days_inactive = _days_since(last_active)
    if days_inactive <= 30:
        details_parts.append("recently active")
    elif days_inactive > INACTIVE_DAYS_THRESHOLD:
        details_parts.append(f"inactive {days_inactive}d")
    
    if signals.get("open_to_work_flag", False):
        details_parts.append("open to work")
    
    response_rate = signals.get("recruiter_response_rate", 0)
    details_parts.append(f"response rate {response_rate:.0%}")
    
    notice = signals.get("notice_period_days", 0)
    details_parts.append(f"notice {notice}d")
    
    github = signals.get("github_activity_score", -1)
    if github >= 0:
        details_parts.append(f"GitHub {github:.0f}")
    
    return {
        "total_score": round(total, 4),
        "availability": round(availability, 4),
        "reliability": round(reliability, 4),
        "market_validation": round(market, 4),
        "technical_signals": round(technical, 4),
        "details": "; ".join(details_parts),
    }


def score_location(candidate: dict) -> dict:
    """
    Score location fit for the JD.
    
    JD: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR welcome.
    India required. Willing to relocate is a plus.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    location = profile.get("location", "").lower().strip()
    country = profile.get("country", "").lower().strip()
    willing_to_relocate = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "")
    
    score = 0.0
    details_parts = []

    in_preferred_country = (country == PREFERRED_COUNTRY or "india" in country or "india" in location)

    if in_preferred_country:
        score += 0.4
        is_preferred = any(loc in location for loc in PREFERRED_LOCATIONS)
        is_acceptable = any(loc in location for loc in ACCEPTABLE_LOCATIONS)
        
        if is_preferred:
            score += 0.4
            details_parts.append(f"preferred location ({location})")
        elif is_acceptable:
            score += 0.25
            details_parts.append(f"acceptable location ({location})")
        else:
            score += 0.1
            details_parts.append(f"India, other city ({location})")
            if willing_to_relocate:
                score += 0.15
                details_parts.append("willing to relocate")
    else:
        score += 0.05  # Non-India
        details_parts.append(f"outside India ({country})")
        if willing_to_relocate:
            score += 0.15
            details_parts.append("willing to relocate")
    
    # Work mode preference (JD says hybrid, flexible cadence)
    if work_mode in ("hybrid", "flexible"):
        score += 0.1
    elif work_mode == "onsite":
        score += 0.05  # Fine but not bonus
    elif work_mode == "remote":
        score += 0.0  # JD is hybrid-preferred
    
    # Salary fit
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)
    
    if sal_max > 0:
        if SALARY_IDEAL_MIN <= sal_min and sal_max <= SALARY_IDEAL_MAX:
            score += 0.05  # Perfect fit
        elif sal_max > SALARY_HARD_MAX:
            score -= 0.05  # May be too expensive for Series A
            details_parts.append(f"high salary ({sal_min}-{sal_max} LPA)")
    
    total = max(0.0, score)
    
    return {
        "total_score": round(total, 4),
        "details": "; ".join(details_parts) if details_parts else "No location info",
    }
