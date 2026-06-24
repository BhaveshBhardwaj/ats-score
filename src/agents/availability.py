"""
Agent 5: The Availability Analyst

Bayesian estimation of actual hireability. Combines all behavioral
signals into a posterior probability that the candidate is actually
available and hireable right now.

Key capabilities:
- Bayesian availability posterior P(hireable | all signals)
- Response probability modeling
- Salary negotiation risk assessment
- Notice period cost estimation
- Market demand inference from saved/viewed counts
"""

import math
from datetime import datetime, date
from agents import (
    BaseAgent, Evidence, Verdict,
    EvidenceType, EvidencePolarity,
)
from config import (
    RESPONSE_RATE_GOOD, RESPONSE_RATE_BAD,
    RESPONSE_TIME_GOOD_HOURS, RESPONSE_TIME_BAD_HOURS,
    NOTICE_PERIOD_IDEAL, NOTICE_PERIOD_OK, NOTICE_PERIOD_BAD,
    INACTIVE_DAYS_THRESHOLD,
    SALARY_IDEAL_MIN, SALARY_IDEAL_MAX, SALARY_HARD_MAX,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS, PREFERRED_COUNTRY,
)


REFERENCE_DATE = date(2026, 6, 1)


def _days_since(date_str: str) -> int:
    """Calculate days between a date string and reference date."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except (ValueError, TypeError):
        return 999


class AvailabilityAgent(BaseAgent):
    """
    The Availability Analyst — Bayesian hireability estimation.
    
    Computes P(candidate is hireable right now) using Bayesian
    updating with behavioral signals as likelihood functions.
    """
    
    AGENT_ID = "availability"
    
    def evaluate(self, candidate: dict) -> Verdict:
        evidence = []
        signals = candidate.get("redrob_signals", {})
        profile = candidate.get("profile", {})
        
        # 1. Bayesian availability estimation
        p_available, avail_evidence = self._estimate_availability(signals)
        evidence.extend(avail_evidence)
        
        # 2. Response probability
        p_response, resp_evidence = self._estimate_response_probability(signals)
        evidence.extend(resp_evidence)
        
        # 3. Salary fit probability
        p_salary, salary_evidence = self._estimate_salary_fit(signals)
        evidence.extend(salary_evidence)
        
        # 4. Location fit
        p_location, loc_evidence = self._estimate_location_fit(profile, signals)
        evidence.extend(loc_evidence)
        
        # 5. Market demand signal
        evidence.extend(self._assess_market_demand(signals))
        
        # 6. Notice period cost
        evidence.extend(self._assess_notice_period(signals))
        
        # Compute composite hire probability
        # P(hire) = P(available) × P(responds) × P(salary_fits) × P(location_ok)
        p_hire = p_available * p_response * p_salary * p_location
        
        # Adjust by behavioral evidence weight
        behavioral_boost = sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.POSITIVE
        ) * 0.1
        behavioral_penalty = abs(sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.NEGATIVE
        )) * 0.1
        
        raw_score = min(1.0, max(0.0, p_hire + behavioral_boost - behavioral_penalty))
        
        confidence = min(1.0, len(evidence) / 6.0)
        
        top = sorted(evidence, key=lambda e: abs(e.effective_weight), reverse=True)[:4]
        reasoning_parts = [e.details for e in top if e.details]
        
        return Verdict(
            agent_id=self.AGENT_ID,
            candidate_id=candidate.get("candidate_id", ""),
            score=raw_score,
            confidence=confidence,
            evidence=evidence,
            reasoning="; ".join(reasoning_parts),
        )
    
    def _estimate_availability(self, signals: dict) -> tuple:
        """
        Bayesian estimation of P(candidate is available for a new role).
        
        Prior: P(available) = 0.3 (base rate — most people aren't actively looking)
        Likelihood updates from signals.
        """
        evidence = []
        
        # Prior
        p_available = 0.3
        
        # Update 1: Open to work flag (strong signal)
        if signals.get("open_to_work_flag", False):
            # P(open_to_work | available) ≈ 0.8, P(open_to_work | not available) ≈ 0.05
            p_available = self._bayesian_update(p_available, 0.8, 0.05)
            evidence.append(self._make_evidence(
                claim="Open to work flag set",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.6,
                evidence_type=EvidenceType.CORROBORATED,
                details="open to work: strong availability signal",
            ))
        else:
            # P(!open_to_work | available) ≈ 0.2
            p_available = self._bayesian_update(p_available, 0.2, 0.95)
        
        # Update 2: Recent activity
        last_active = signals.get("last_active_date", "")
        days_inactive = _days_since(last_active)
        
        if days_inactive <= 7:
            p_available = self._bayesian_update(p_available, 0.7, 0.3)
            evidence.append(self._make_evidence(
                claim="Active within last week",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"active {days_inactive}d ago: high availability",
            ))
        elif days_inactive <= 30:
            p_available = self._bayesian_update(p_available, 0.5, 0.35)
            evidence.append(self._make_evidence(
                claim="Active within last month",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.3,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"active {days_inactive}d ago: moderate availability",
            ))
        elif days_inactive > INACTIVE_DAYS_THRESHOLD:
            p_available = self._bayesian_update(p_available, 0.1, 0.7)
            evidence.append(self._make_evidence(
                claim=f"Inactive for {days_inactive} days",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.6,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"inactive {days_inactive}d: very low availability",
            ))
        
        # Update 3: Applications submitted recently
        apps = signals.get("applications_submitted_30d", 0)
        if apps >= 5:
            p_available = self._bayesian_update(p_available, 0.8, 0.1)
            evidence.append(self._make_evidence(
                claim=f"{apps} applications in last 30 days",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"{apps} recent applications: actively job-seeking",
            ))
        elif apps >= 1:
            p_available = self._bayesian_update(p_available, 0.6, 0.2)
        
        return p_available, evidence
    
    def _estimate_response_probability(self, signals: dict) -> tuple:
        """
        Estimate P(candidate will respond to our outreach).
        Based on historical response rate and response time.
        """
        evidence = []
        
        response_rate = signals.get("recruiter_response_rate", 0)
        response_time = signals.get("avg_response_time_hours", 999)
        
        # Response rate is the strongest signal
        p_response = max(0.05, min(0.95, response_rate))
        
        # Adjust by response time
        if response_time <= RESPONSE_TIME_GOOD_HOURS:
            p_response = min(0.95, p_response * 1.2)
        elif response_time >= RESPONSE_TIME_BAD_HOURS:
            p_response *= 0.7
        
        # Generate evidence
        if response_rate >= RESPONSE_RATE_GOOD:
            evidence.append(self._make_evidence(
                claim=f"High response rate: {response_rate:.0%}",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.6,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"response rate {response_rate:.0%}: will likely respond",
            ))
        elif response_rate <= RESPONSE_RATE_BAD:
            evidence.append(self._make_evidence(
                claim=f"Very low response rate: {response_rate:.0%}",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.7,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"response rate {response_rate:.0%}: unlikely to respond",
            ))
        
        return p_response, evidence
    
    def _estimate_salary_fit(self, signals: dict) -> tuple:
        """
        Estimate P(salary negotiation succeeds).
        """
        evidence = []
        
        salary_range = signals.get("expected_salary_range_inr_lpa", {})
        sal_min = salary_range.get("min", 0)
        sal_max = salary_range.get("max", 0)
        
        if sal_max <= 0:
            return 0.6, evidence  # No salary info = neutral
        
        # Perfect fit
        if SALARY_IDEAL_MIN <= sal_min and sal_max <= SALARY_IDEAL_MAX:
            p_salary = 0.85
            evidence.append(self._make_evidence(
                claim=f"Salary expectation fits: {sal_min}-{sal_max} LPA",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.4,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"salary {sal_min}-{sal_max} LPA: fits budget",
            ))
        elif sal_max > SALARY_HARD_MAX:
            p_salary = 0.2
            evidence.append(self._make_evidence(
                claim=f"Salary too high: {sal_min}-{sal_max} LPA",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.5,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"salary {sal_min}-{sal_max} LPA: likely too expensive",
            ))
        elif sal_min > SALARY_IDEAL_MAX:
            p_salary = 0.4
            evidence.append(self._make_evidence(
                claim=f"Salary above ideal: {sal_min}-{sal_max} LPA",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.3,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"salary {sal_min}-{sal_max} LPA: above ideal range",
            ))
        else:
            p_salary = 0.7
        
        return p_salary, evidence
    
    def _estimate_location_fit(self, profile: dict, signals: dict) -> tuple:
        """Estimate P(location works for this role)."""
        evidence = []
        
        location = profile.get("location", "").lower().strip()
        country = profile.get("country", "").lower().strip()
        willing_to_relocate = signals.get("willing_to_relocate", False)
        
        if country == PREFERRED_COUNTRY or "india" in country:
            is_preferred = any(loc in location for loc in PREFERRED_LOCATIONS)
            is_acceptable = any(loc in location for loc in ACCEPTABLE_LOCATIONS)
            
            if is_preferred:
                p_location = 0.95
                evidence.append(self._make_evidence(
                    claim=f"Preferred location: {location}",
                    source="profile",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.CORROBORATED,
                    details=f"preferred location ({location})",
                ))
            elif is_acceptable:
                p_location = 0.8
            elif willing_to_relocate:
                p_location = 0.7
                evidence.append(self._make_evidence(
                    claim="Willing to relocate within India",
                    source="redrob_signals",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.3,
                    evidence_type=EvidenceType.CORROBORATED,
                    details="willing to relocate to preferred location",
                ))
            else:
                p_location = 0.5
        else:
            if willing_to_relocate:
                p_location = 0.3
                evidence.append(self._make_evidence(
                    claim=f"Outside India ({country}) but willing to relocate",
                    source="profile + redrob_signals",
                    polarity=EvidencePolarity.NEUTRAL,
                    strength=0.3,
                    evidence_type=EvidenceType.CORROBORATED,
                    details=f"outside India ({country}), willing to relocate",
                ))
            else:
                p_location = 0.15
                evidence.append(self._make_evidence(
                    claim=f"Outside India ({country}), not willing to relocate",
                    source="profile + redrob_signals",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.CORROBORATED,
                    details=f"outside India ({country}), won't relocate",
                ))
        
        return p_location, evidence
    
    def _assess_market_demand(self, signals: dict) -> list:
        """Infer market demand for this candidate from social signals."""
        evidence = []
        
        saved = signals.get("saved_by_recruiters_30d", 0)
        views = signals.get("profile_views_received_30d", 0)
        search_appearances = signals.get("search_appearance_30d", 0)
        
        # High demand = other recruiters want them too
        demand_score = (
            min(1.0, saved / 15.0) * 0.4 +
            min(1.0, views / 25.0) * 0.3 +
            min(1.0, search_appearances / 100.0) * 0.3
        )
        
        if demand_score >= 0.5:
            evidence.append(self._make_evidence(
                claim=f"High market demand ({saved} saves, {views} views)",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.4,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"high demand: {saved} saves, {views} views, {search_appearances} appearances",
            ))
        
        return evidence
    
    def _assess_notice_period(self, signals: dict) -> list:
        """Assess notice period as a hiring cost factor."""
        evidence = []
        
        notice = signals.get("notice_period_days", 90)
        
        if notice <= NOTICE_PERIOD_IDEAL:
            evidence.append(self._make_evidence(
                claim=f"Short notice period: {notice} days",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"notice period {notice}d: ideal (<{NOTICE_PERIOD_IDEAL}d)",
            ))
        elif notice > NOTICE_PERIOD_BAD:
            evidence.append(self._make_evidence(
                claim=f"Long notice period: {notice} days",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.4,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"notice period {notice}d: long (>{NOTICE_PERIOD_BAD}d)",
            ))
        
        # Interview completion and offer acceptance as reliability signals
        completion = signals.get("interview_completion_rate", 0)
        if completion >= 0.9:
            evidence.append(self._make_evidence(
                claim=f"High interview completion: {completion:.0%}",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.3,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"interview completion {completion:.0%}: reliable",
            ))
        elif completion < 0.5 and completion > 0:
            evidence.append(self._make_evidence(
                claim=f"Low interview completion: {completion:.0%}",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.3,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"interview completion {completion:.0%}: unreliable",
            ))
        
        return evidence
    
    @staticmethod
    def _bayesian_update(prior: float, likelihood_if_true: float, likelihood_if_false: float) -> float:
        """
        Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        where P(E) = P(E|H)*P(H) + P(E|!H)*P(!H)
        """
        p_evidence = likelihood_if_true * prior + likelihood_if_false * (1 - prior)
        if p_evidence <= 0:
            return prior
        posterior = (likelihood_if_true * prior) / p_evidence
        return max(0.01, min(0.99, posterior))
