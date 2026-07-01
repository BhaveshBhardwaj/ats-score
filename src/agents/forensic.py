"""
Agent 3: The Forensic Auditor

Cross-validates ALL claims across data sources using the
Cross-Evidence Corroboration Matrix (CECM). Detects inconsistencies
that single-dimension scorers miss.

Key capabilities:
- Build corroboration matrix: skill claims vs evidence sources
- Timeline integrity analysis (enhanced honeypot detection)
- Assessment vs claim validation
- Endorsement anomaly detection
"""

from agents import (
    BaseAgent, Evidence, Verdict,
    EvidenceType, EvidencePolarity,
)
from config import (
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, MLOPS_PRODUCTION_SKILLS,
    ML_WORK_KEYWORDS,
)


class ForensicAgent(BaseAgent):
    """
    The Forensic Auditor — cross-validates all claims.
    
    This agent doesn't have a bias toward positive or negative.
    It's purely analytical: does the evidence support the claims?
    The CECM (Cross-Evidence Corroboration Matrix) is its core tool.
    """
    
    AGENT_ID = "forensic"
    
    def evaluate(self, candidate: dict) -> Verdict:
        evidence = []
        
        # 1. Build and analyze corroboration matrix
        cecm_evidence, cecm_score = self._build_corroboration_matrix(candidate)
        evidence.extend(cecm_evidence)
        
        # 2. Timeline integrity
        evidence.extend(self._check_timeline_integrity(candidate))
        
        # 3. Assessment validation
        evidence.extend(self._validate_assessments(candidate))
        
        # 4. Endorsement anomalies
        evidence.extend(self._check_endorsement_anomalies(candidate))
        
        # 5. Profile consistency
        evidence.extend(self._check_profile_consistency(candidate))
        
        # 6. Assessment-verified must-have skills (strongest objective signal)
        evidence.extend(self._check_assessment_verified_musthaves(candidate))
        
        # Forensic score is based on integrity — how trustworthy is this profile?
        positive = sum(e.effective_weight for e in evidence if e.polarity == EvidencePolarity.POSITIVE)
        negative = abs(sum(e.effective_weight for e in evidence if e.polarity == EvidencePolarity.NEGATIVE))
        
        total_weight = positive + negative
        if total_weight > 0:
            # Integrity score: high = trustworthy profile, low = suspicious
            raw_score = (positive + cecm_score) / (total_weight + 1.0)
        else:
            raw_score = 0.5
        
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
    
    def _build_corroboration_matrix(self, candidate: dict) -> tuple:
        """
        Build the Cross-Evidence Corroboration Matrix (CECM).
        
        For each claimed skill, check how many independent evidence
        sources corroborate it:
        1. Skills list (self-reported)
        2. Career descriptions (behavioral evidence)
        3. Assessment scores (tested evidence)
        4. Endorsements (social evidence)
        
        A skill corroborated by ≥2 sources is credible.
        A skill with only self-reported evidence is discounted.
        """
        evidence = []
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})
        
        all_desc = " ".join(j.get("description", "").lower() for j in career)
        summary = candidate.get("profile", {}).get("summary", "").lower()
        headline = candidate.get("profile", {}).get("headline", "").lower()
        all_text = all_desc + " " + summary + " " + headline
        
        all_relevant = MUST_HAVE_SKILLS | CORE_AI_ML_SKILLS | NLP_IR_SKILLS | LLM_FINETUNING_SKILLS | MLOPS_PRODUCTION_SKILLS
        
        corroborated_count = 0
        uncorroborated_count = 0
        contradicted_count = 0
        total_relevant = 0
        
        for skill in skills:
            name = skill.get("name", "").lower().strip()
            proficiency = skill.get("proficiency", "beginner")
            duration = skill.get("duration_months", 0)
            endorsements = skill.get("endorsements", 0)
            
            # Check if this is a relevant skill
            is_relevant = any(
                rel in name or name in rel
                for rel in all_relevant
            )
            if not is_relevant:
                continue
            
            total_relevant += 1
            sources = 1  # Self-reported is always 1
            
            # Check career description corroboration
            # Require ≥5 chars to avoid false positives (e.g. "deep" matching
            # unrelated contexts when skill is "deep learning")
            career_match = False
            name_parts = name.split()
            for part in name_parts:
                if len(part) >= 5 and part in all_text:
                    career_match = True
                    break
            # Also check the full skill name for shorter multi-word skills
            if not career_match and len(name) >= 5 and name in all_text:
                career_match = True
            if career_match:
                sources += 1
            
            # Check assessment corroboration
            assessment_match = False
            assessment_score = None
            for assessed_skill, score in assessments.items():
                if name in assessed_skill.lower() or assessed_skill.lower() in name:
                    assessment_match = True
                    assessment_score = score
                    sources += 1
                    break
            
            # Check endorsement corroboration
            if endorsements >= 5:
                sources += 1
            
            # Determine corroboration level
            if sources >= 3:
                corroborated_count += 1
                if proficiency in ("expert", "advanced"):
                    evidence.append(self._make_evidence(
                        claim=f"Strongly corroborated: {name} ({proficiency})",
                        source="CECM",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=0.6,
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"CECM: '{name}' corroborated by {sources} sources",
                    ))
            elif sources == 2:
                corroborated_count += 1
            elif sources == 1:
                uncorroborated_count += 1
                if proficiency == "expert":
                    evidence.append(self._make_evidence(
                        claim=f"Uncorroborated expert claim: {name}",
                        source="CECM",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.4,
                        evidence_type=EvidenceType.SELF_REPORTED,
                        details=f"CECM: expert '{name}' only self-reported, no corroboration",
                    ))
            
            # Check for contradiction
            if assessment_match and assessment_score is not None:
                if proficiency == "expert" and assessment_score < 30:
                    contradicted_count += 1
                    evidence.append(self._make_evidence(
                        claim=f"Assessment contradicts claim: {name}",
                        source="CECM",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.7,
                        evidence_type=EvidenceType.CONTRADICTED,
                        details=f"CECM: '{name}' expert claim but {assessment_score}/100 assessment",
                    ))
                elif proficiency in ("expert", "advanced") and assessment_score >= 70:
                    evidence.append(self._make_evidence(
                        claim=f"Assessment confirms claim: {name}",
                        source="CECM",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=0.5,
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"CECM: '{name}' confirmed by {assessment_score}/100 assessment",
                    ))
        
        # Overall corroboration score
        if total_relevant > 0:
            cecm_score = corroborated_count / total_relevant
            
            if corroborated_count >= 3:
                evidence.append(self._make_evidence(
                    claim=f"High profile integrity: {corroborated_count}/{total_relevant} skills corroborated",
                    source="CECM",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.FORENSIC,
                    details=f"profile integrity: {corroborated_count}/{total_relevant} corroborated",
                ))
            
            if uncorroborated_count > corroborated_count and total_relevant >= 4:
                evidence.append(self._make_evidence(
                    claim=f"Low profile integrity: {uncorroborated_count}/{total_relevant} skills uncorroborated",
                    source="CECM",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.FORENSIC,
                    details=f"low integrity: {uncorroborated_count}/{total_relevant} uncorroborated",
                ))
        else:
            cecm_score = 0.0
        
        return evidence, cecm_score
    
    def _check_timeline_integrity(self, candidate: dict) -> list:
        """Enhanced timeline integrity checks."""
        evidence = []
        career = candidate.get("career_history", [])
        profile = candidate.get("profile", {})
        education = candidate.get("education", [])
        years_exp = profile.get("years_of_experience", 0)
        
        # Career months vs stated experience
        total_months = sum(j.get("duration_months", 0) for j in career)
        expected_months = years_exp * 12
        
        if total_months > 0 and expected_months > 0:
            ratio = total_months / expected_months
            if ratio > 2.0:
                evidence.append(self._make_evidence(
                    claim=f"Career months ({total_months}) >> stated experience ({years_exp} yrs)",
                    source="timeline",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.6,
                    evidence_type=EvidenceType.CONTRADICTED,
                    details=f"timeline mismatch: {total_months}mo career vs {years_exp}yr stated",
                ))
            elif ratio < 0.3 and years_exp > 3:
                evidence.append(self._make_evidence(
                    claim=f"Career months ({total_months}) << stated experience ({years_exp} yrs)",
                    source="timeline",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.CONTRADICTED,
                    details=f"timeline gap: only {total_months}mo career for {years_exp}yr stated",
                ))
            elif 0.7 <= ratio <= 1.3:
                evidence.append(self._make_evidence(
                    claim="Career timeline consistent with stated experience",
                    source="timeline",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.3,
                    evidence_type=EvidenceType.CORROBORATED,
                    details="timeline consistent",
                ))
        
        # Education vs career timeline
        if education:
            latest_grad = max(
                (edu.get("end_year", 0) for edu in education), default=0
            )
            if latest_grad > 0:
                years_since_grad = 2026 - latest_grad
                if years_exp > years_since_grad + 3:
                    evidence.append(self._make_evidence(
                        claim=f"Experience ({years_exp}yr) exceeds time since graduation ({years_since_grad}yr)",
                        source="education + profile",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.7,
                        evidence_type=EvidenceType.CONTRADICTED,
                        details=f"impossible: {years_exp}yr exp but graduated {years_since_grad}yr ago",
                    ))
        
        return evidence
    
    def _validate_assessments(self, candidate: dict) -> list:
        """Validate assessment scores against skill claims."""
        evidence = []
        skills = candidate.get("skills", [])
        assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
        
        if not assessments:
            return evidence
        
        # High assessment scores are a strong positive
        high_scores = {k: v for k, v in assessments.items() if v >= 70}
        low_scores = {k: v for k, v in assessments.items() if v < 30}
        
        if len(high_scores) >= 2:
            evidence.append(self._make_evidence(
                claim=f"{len(high_scores)} high assessment scores",
                source="assessments",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"strong assessments: {len(high_scores)} scores ≥70",
            ))
        
        if len(low_scores) >= 2:
            evidence.append(self._make_evidence(
                claim=f"{len(low_scores)} low assessment scores",
                source="assessments",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.4,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"weak assessments: {len(low_scores)} scores <30",
            ))
        
        return evidence
    
    def _check_endorsement_anomalies(self, candidate: dict) -> list:
        """Detect endorsement patterns that suggest manipulation."""
        evidence = []
        skills = candidate.get("skills", [])
        
        # Check for high endorsements on zero-duration skills
        for s in skills:
            endorsements = s.get("endorsements", 0)
            duration = s.get("duration_months", 0)
            name = s.get("name", "")
            
            if endorsements >= 20 and duration == 0:
                evidence.append(self._make_evidence(
                    claim=f"High endorsements ({endorsements}) on zero-duration skill: {name}",
                    source="skills_list",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.5,
                    evidence_type=EvidenceType.FORENSIC,
                    details=f"anomaly: {endorsements} endorsements on '{name}' with 0mo",
                ))
        
        return evidence
    
    def _check_profile_consistency(self, candidate: dict) -> list:
        """Check for overall profile consistency signals."""
        evidence = []
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        
        # Profile completeness as trust signal
        completeness = signals.get("profile_completeness_score", 0)
        if completeness >= 80:
            evidence.append(self._make_evidence(
                claim="High profile completeness",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.2,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"profile {completeness:.0f}% complete",
            ))
        elif completeness < 40:
            evidence.append(self._make_evidence(
                claim="Low profile completeness",
                source="redrob_signals",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.2,
                evidence_type=EvidenceType.FORENSIC,
                details=f"profile only {completeness:.0f}% complete",
            ))
        
        # Verified credentials
        verified_count = sum([
            signals.get("verified_email", False),
            signals.get("verified_phone", False),
            signals.get("linkedin_connected", False),
        ])
        if verified_count >= 3:
            evidence.append(self._make_evidence(
                claim="Fully verified profile",
                source="redrob_signals",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.2,
                evidence_type=EvidenceType.CORROBORATED,
                details="all 3 verifications (email, phone, LinkedIn)",
            ))
        
        return evidence
    
    def _check_assessment_verified_musthaves(self, candidate: dict) -> list:
        """
        Check if must-have skills have been verified by Redrob assessments.
        A candidate who scored 85/100 on Python assessment is objectively
        more credible than one who just lists "Python Expert".
        This is the most trustworthy signal in the entire dataset.
        """
        evidence = []
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})
        
        if not assessments:
            return evidence
        
        must_have_keywords = {
            "python", "embedding", "embeddings", "vector", "search",
            "retrieval", "ranking", "nlp", "machine learning",
            "deep learning", "pytorch", "tensorflow",
        }
        
        verified_musthave_count = 0
        
        for assessed_skill, score in assessments.items():
            skill_lower = assessed_skill.lower()
            is_musthave = any(kw in skill_lower for kw in must_have_keywords)
            
            if is_musthave and score >= 70:
                verified_musthave_count += 1
                if score >= 85:
                    evidence.append(self._make_evidence(
                        claim=f"Assessment-verified must-have: {assessed_skill}",
                        source="assessment",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=0.8,
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"assessment-verified '{assessed_skill}' scored {score}/100",
                    ))
                else:
                    evidence.append(self._make_evidence(
                        claim=f"Assessment-verified must-have: {assessed_skill}",
                        source="assessment",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=0.5,
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"assessment-verified '{assessed_skill}' scored {score}/100",
                    ))
        
        # Bonus for having multiple must-haves verified
        if verified_musthave_count >= 3:
            evidence.append(self._make_evidence(
                claim="Multiple must-have skills assessment-verified",
                source="assessment",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.7,
                evidence_type=EvidenceType.CORROBORATED,
                details=f"{verified_musthave_count} must-have skills verified by assessment",
            ))
        
        return evidence
