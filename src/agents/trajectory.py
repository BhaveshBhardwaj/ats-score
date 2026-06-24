"""
Agent 4: The Trajectory Analyst

Uses Monte Carlo-inspired simulation of career paths to assess not
just where a candidate IS, but where their trajectory is GOING.

Key capabilities:
- Career velocity scoring (rate of progression)
- Company quality trajectory (improving or declining)
- Role depth analysis (duration × description richness × company type)
- Counterfactual career modeling
"""

import math
from agents import (
    BaseAgent, Evidence, Verdict,
    EvidenceType, EvidencePolarity,
)
from config import (
    TECH_PRODUCT_INDUSTRIES, SERVICES_INDUSTRIES,
    CONSULTING_SERVICES_COMPANIES,
    ML_WORK_KEYWORDS, PRODUCTION_KEYWORDS,
    HIGHLY_RELEVANT_TITLES, MODERATELY_RELEVANT_TITLES,
)


# Title level mapping for trajectory analysis
TITLE_LEVELS = {
    "intern": 0, "trainee": 0, "fresher": 0,
    "junior": 1, "associate": 1, "analyst": 1.5,
    "engineer": 2, "developer": 2, "programmer": 2,
    "senior": 3, "specialist": 3,
    "lead": 3.5, "team lead": 3.5,
    "staff": 4, "architect": 4,
    "principal": 4.5, "manager": 3.5,
    "director": 5, "head": 5,
    "vp": 5.5, "chief": 6,
}

# Ideal career trajectory for this JD
IDEAL_TRAJECTORY = {
    "title_level": 3.5,          # Senior/Lead level
    "years_ml": 4.5,             # 4-5 years in ML roles
    "product_company_ratio": 0.7, # 70%+ at product companies
    "production_signals": 3,      # Strong production evidence
    "ml_depth": 0.7,             # High ML depth ratio
    "company_quality_trend": 0.6, # Trending toward better companies
}


class TrajectoryAgent(BaseAgent):
    """
    The Trajectory Analyst — Monte Carlo career simulation.
    
    Instead of asking "is this career good?", asks
    "given this trajectory, what's the probability this person
    can do the job?"
    """
    
    AGENT_ID = "trajectory"
    
    def evaluate(self, candidate: dict) -> Verdict:
        evidence = []
        
        # 1. Career velocity analysis
        evidence.extend(self._analyze_velocity(candidate))
        
        # 2. Company quality trajectory
        evidence.extend(self._analyze_company_trajectory(candidate))
        
        # 3. Role depth analysis
        evidence.extend(self._analyze_role_depth(candidate))
        
        # 4. Counterfactual career similarity to ideal
        counterfactual_score = self._compute_counterfactual(candidate)
        evidence.append(self._make_evidence(
            claim=f"Career trajectory similarity to ideal: {counterfactual_score:.2f}",
            source="trajectory_simulation",
            polarity=EvidencePolarity.POSITIVE if counterfactual_score > 0.4 else EvidencePolarity.NEGATIVE,
            strength=counterfactual_score,
            evidence_type=EvidenceType.INFERRED,
            details=f"trajectory similarity: {counterfactual_score:.0%} match to ideal",
        ))
        
        # 5. ML career depth over time
        evidence.extend(self._analyze_ml_depth(candidate))
        
        # Compute trajectory score
        positive = sum(e.effective_weight for e in evidence if e.polarity == EvidencePolarity.POSITIVE)
        negative = abs(sum(e.effective_weight for e in evidence if e.polarity == EvidencePolarity.NEGATIVE))
        
        total = positive + negative
        raw_score = (positive / total) if total > 0 else 0.5
        
        # Blend with counterfactual
        raw_score = 0.6 * raw_score + 0.4 * counterfactual_score
        
        confidence = min(1.0, len(evidence) / 5.0)
        
        top = sorted(evidence, key=lambda e: abs(e.effective_weight), reverse=True)[:4]
        reasoning_parts = [e.details for e in top if e.details]
        
        return Verdict(
            agent_id=self.AGENT_ID,
            candidate_id=candidate.get("candidate_id", ""),
            score=min(1.0, max(0.0, raw_score)),
            confidence=confidence,
            evidence=evidence,
            reasoning="; ".join(reasoning_parts),
        )
    
    def _get_title_level(self, title: str) -> float:
        """Extract numeric level from a title string."""
        title_lower = title.lower().strip()
        best_level = 2.0  # Default mid-level
        
        for keyword, level in TITLE_LEVELS.items():
            if keyword in title_lower:
                best_level = max(best_level, level)
        
        return best_level
    
    def _is_product_company(self, job: dict) -> bool:
        """Check if a job was at a product company."""
        industry = job.get("industry", "").lower()
        company = job.get("company", "").lower()
        
        is_consulting = any(c in company for c in CONSULTING_SERVICES_COMPANIES)
        is_product = any(ind in industry for ind in TECH_PRODUCT_INDUSTRIES)
        
        return is_product and not is_consulting
    
    def _analyze_velocity(self, candidate: dict) -> list:
        """
        Analyze career velocity — how fast the candidate progresses.
        Good velocity: meaningful title progression over time.
        Bad velocity: stagnation or regression.
        """
        evidence = []
        career = candidate.get("career_history", [])
        
        if len(career) < 2:
            return evidence
        
        # Sort chronologically
        sorted_career = sorted(
            career,
            key=lambda j: j.get("start_date", "1900-01-01"),
        )
        
        levels = [self._get_title_level(j.get("title", "")) for j in sorted_career]
        
        # Career duration in years
        total_months = sum(j.get("duration_months", 0) for j in career)
        total_years = max(total_months / 12.0, 1.0)
        
        # Velocity = level change per year
        if len(levels) >= 2:
            level_change = levels[-1] - levels[0]
            velocity = level_change / total_years
            
            if velocity >= 0.3:
                evidence.append(self._make_evidence(
                    claim=f"High career velocity: {velocity:.2f} levels/year",
                    source="career_history",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=min(0.7, 0.3 + velocity * 0.5),
                    evidence_type=EvidenceType.INFERRED,
                    details=f"high velocity: {levels[0]:.0f}→{levels[-1]:.0f} in {total_years:.1f}yr",
                ))
            elif velocity < -0.1:
                evidence.append(self._make_evidence(
                    claim=f"Negative career velocity: {velocity:.2f}",
                    source="career_history",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.4,
                    evidence_type=EvidenceType.INFERRED,
                    details=f"declining: {levels[0]:.0f}→{levels[-1]:.0f} in {total_years:.1f}yr",
                ))
        
        return evidence
    
    def _analyze_company_trajectory(self, candidate: dict) -> list:
        """
        Analyze whether the candidate is moving toward better companies.
        Product company ratio should be increasing over career.
        """
        evidence = []
        career = candidate.get("career_history", [])
        
        if len(career) < 2:
            return evidence
        
        sorted_career = sorted(
            career,
            key=lambda j: j.get("start_date", "1900-01-01"),
        )
        
        mid = len(sorted_career) // 2
        first_half = sorted_career[:max(mid, 1)]
        second_half = sorted_career[max(mid, 1):]
        
        if not second_half:
            return evidence
        
        first_product = sum(1 for j in first_half if self._is_product_company(j)) / len(first_half)
        second_product = sum(1 for j in second_half if self._is_product_company(j)) / len(second_half)
        
        if second_product > first_product + 0.1:
            evidence.append(self._make_evidence(
                claim="Improving company quality trajectory",
                source="career_history",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.INFERRED,
                details=f"company quality improving: {first_product:.0%}→{second_product:.0%} product",
            ))
        elif first_product > second_product + 0.3:
            evidence.append(self._make_evidence(
                claim="Declining company quality",
                source="career_history",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.4,
                evidence_type=EvidenceType.INFERRED,
                details=f"company quality declining: {first_product:.0%}→{second_product:.0%} product",
            ))
        
        return evidence
    
    def _analyze_role_depth(self, candidate: dict) -> list:
        """
        Analyze the depth of each role: duration × description richness × company type.
        Long tenure at a product company with rich ML descriptions = deep experience.
        """
        evidence = []
        career = candidate.get("career_history", [])
        
        deep_roles = 0
        shallow_roles = 0
        
        for job in career:
            duration = job.get("duration_months", 0)
            desc = job.get("description", "")
            
            # Description richness: how detailed is the description?
            desc_length = len(desc)
            ml_hits = sum(1 for kw in ML_WORK_KEYWORDS if kw in desc.lower())
            production_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc.lower())
            
            is_product = self._is_product_company(job)
            
            # Depth score for this role
            depth = 0
            depth += min(1.0, duration / 24.0) * 0.3  # Duration (cap at 2 yrs)
            depth += min(1.0, desc_length / 500.0) * 0.2  # Description richness
            depth += min(1.0, ml_hits / 5.0) * 0.3  # ML evidence
            depth += min(1.0, production_hits / 3.0) * 0.1  # Production evidence
            depth += 0.1 if is_product else 0.0  # Product company bonus
            
            if depth >= 0.5:
                deep_roles += 1
            elif depth < 0.2:
                shallow_roles += 1
        
        if deep_roles >= 2:
            evidence.append(self._make_evidence(
                claim=f"{deep_roles} deep ML roles",
                source="career_history",
                polarity=EvidencePolarity.POSITIVE,
                strength=min(0.8, 0.3 + deep_roles * 0.15),
                evidence_type=EvidenceType.INFERRED,
                details=f"{deep_roles} deep roles (long tenure + rich ML description)",
            ))
        
        if shallow_roles > deep_roles and len(career) >= 3:
            evidence.append(self._make_evidence(
                claim=f"Mostly shallow roles ({shallow_roles}/{len(career)})",
                source="career_history",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.4,
                evidence_type=EvidenceType.INFERRED,
                details=f"shallow roles: {shallow_roles}/{len(career)} roles lack depth",
            ))
        
        return evidence
    
    def _compute_counterfactual(self, candidate: dict) -> float:
        """
        Monte Carlo-inspired counterfactual career modeling.
        
        Given this candidate's trajectory, how closely does it match
        the ideal trajectory for this role? Uses multi-dimensional
        distance in career-feature space.
        """
        career = candidate.get("career_history", [])
        profile = candidate.get("profile", {})
        
        if not career:
            return 0.0
        
        # Compute candidate's trajectory features
        sorted_career = sorted(
            career,
            key=lambda j: j.get("start_date", "1900-01-01"),
        )
        
        # 1. Title level
        current_level = self._get_title_level(
            profile.get("current_title", "")
        )
        title_sim = 1.0 - abs(current_level - IDEAL_TRAJECTORY["title_level"]) / 6.0
        
        # 2. Years in ML roles
        all_desc = " ".join(j.get("description", "").lower() for j in career)
        ml_months = sum(
            j.get("duration_months", 0) for j in career
            if sum(1 for kw in ML_WORK_KEYWORDS if kw in j.get("description", "").lower()) >= 2
        )
        ml_years = ml_months / 12.0
        ml_sim = 1.0 - min(1.0, abs(ml_years - IDEAL_TRAJECTORY["years_ml"]) / 5.0)
        
        # 3. Product company ratio
        product_count = sum(1 for j in career if self._is_product_company(j))
        product_ratio = product_count / max(len(career), 1)
        product_sim = 1.0 - abs(product_ratio - IDEAL_TRAJECTORY["product_company_ratio"])
        
        # 4. Production signals
        production_count = sum(1 for kw in PRODUCTION_KEYWORDS if kw in all_desc)
        production_sim = min(1.0, production_count / max(IDEAL_TRAJECTORY["production_signals"], 1))
        
        # 5. ML depth ratio
        ml_roles = sum(
            1 for j in career
            if sum(1 for kw in ML_WORK_KEYWORDS if kw in j.get("description", "").lower()) >= 1
        )
        ml_depth = ml_roles / max(len(career), 1)
        ml_depth_sim = 1.0 - abs(ml_depth - IDEAL_TRAJECTORY["ml_depth"])
        
        # Weighted combination (simulating probability over trajectories)
        counterfactual = (
            0.20 * max(0, title_sim) +
            0.25 * max(0, ml_sim) +
            0.20 * max(0, product_sim) +
            0.20 * max(0, production_sim) +
            0.15 * max(0, ml_depth_sim)
        )
        
        return min(1.0, max(0.0, counterfactual))
    
    def _analyze_ml_depth(self, candidate: dict) -> list:
        """Analyze the depth and progression of ML experience."""
        evidence = []
        career = candidate.get("career_history", [])
        
        sorted_career = sorted(
            career,
            key=lambda j: j.get("start_date", "1900-01-01"),
        )
        
        # Track ML sophistication over time
        ml_sophistication = []
        for job in sorted_career:
            desc = job.get("description", "").lower()
            
            # Basic ML keywords
            basic = sum(1 for kw in ["machine learning", "model", "training", "prediction"] if kw in desc)
            # Advanced ML keywords
            advanced = sum(1 for kw in [
                "embedding", "transformer", "ranking", "retrieval",
                "fine-tuning", "distributed", "real-time inference",
                "a/b test", "ndcg", "evaluation framework",
            ] if kw in desc)
            
            if basic + advanced > 0:
                sophistication = (basic * 1.0 + advanced * 2.0) / (basic + advanced)
                ml_sophistication.append(sophistication)
        
        if len(ml_sophistication) >= 2:
            # Check if ML sophistication is increasing
            first_half_avg = sum(ml_sophistication[:len(ml_sophistication)//2]) / max(len(ml_sophistication)//2, 1)
            second_half_avg = sum(ml_sophistication[len(ml_sophistication)//2:]) / max(len(ml_sophistication) - len(ml_sophistication)//2, 1)
            
            if second_half_avg > first_half_avg:
                evidence.append(self._make_evidence(
                    claim="Increasing ML sophistication over career",
                    source="career_history",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.INFERRED,
                    details=f"ML sophistication increasing: {first_half_avg:.1f}→{second_half_avg:.1f}",
                ))
        
        return evidence
