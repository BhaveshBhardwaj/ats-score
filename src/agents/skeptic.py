"""
Agent 2: The Skeptic

Adversarial counterpart to the Advocate. Actively searches for
RED FLAGS and reasons NOT to hire. This agent ensures that keyword
stuffers, credential inflators, and superficially impressive but
actually weak candidates are caught.

Key capabilities:
- Credential inflation detection (expert claims vs evidence)
- Keyword stuffing detection (many buzzwords, no substance)
- Career pattern red flags (hopping, decline, stagnation)
- Consulting masquerade detection
- Skill-career mismatch detection
"""

from flashtext import KeywordProcessor

from agents import (
    BaseAgent, Evidence, Verdict,
    EvidenceType, EvidencePolarity,
)
from config import (
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, NEGATIVE_DOMAIN_SKILLS,
    CONSULTING_SERVICES_COMPANIES,
    ML_WORK_KEYWORDS, NON_AI_WORK_KEYWORDS,
    IRRELEVANT_TITLES,
)

# ── Initialize FlashText Processors ──────────────────────────────
_all_relevant_processor = KeywordProcessor(case_sensitive=False)
for kw in (MUST_HAVE_SKILLS | CORE_AI_ML_SKILLS | NLP_IR_SKILLS | LLM_FINETUNING_SKILLS):
    _all_relevant_processor.add_keyword(kw, kw)

_ml_processor = KeywordProcessor(case_sensitive=False)
for kw in ML_WORK_KEYWORDS:
    _ml_processor.add_keyword(kw, "ML")


class SkepticAgent(BaseAgent):
    """
    The Skeptic agent — finds reasons NOT to hire.
    
    Biased toward finding negative signals. Cross-validates claims
    and flags inconsistencies. This agent is the adversary that
    forces the system to justify positive assessments.
    """
    
    AGENT_ID = "skeptic"
    
    def evaluate(self, candidate: dict) -> Verdict:
        evidence = []
        
        # 1. Credential inflation detection
        evidence.extend(self._detect_credential_inflation(candidate))
        
        # 2. Keyword stuffing detection
        evidence.extend(self._detect_keyword_stuffing(candidate))
        
        # 3. Career red flags
        evidence.extend(self._detect_career_red_flags(candidate))
        
        # 4. Skill-career mismatch
        evidence.extend(self._detect_skill_career_mismatch(candidate))
        
        # 5. Recency and relevance decay
        evidence.extend(self._assess_recency(candidate))
        
        # Compute skeptic score (inverted — high negative weight = low score)
        negative_weight = abs(sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.NEGATIVE
        ))
        positive_weight = sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.POSITIVE
        )
        
        # Skeptic inverts: many red flags = low score
        if negative_weight + positive_weight > 0:
            raw_score = positive_weight / (positive_weight + negative_weight * 1.5)
        else:
            raw_score = 0.5  # No evidence = neutral
        
        confidence = min(1.0, len(evidence) / 6.0)
        
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
    
    def challenge(self, evidence: Evidence, candidate: dict):
        """
        The Skeptic challenges positive evidence by looking for
        contradicting information in the candidate's profile.
        """
        if evidence.polarity != EvidencePolarity.POSITIVE:
            return None  # Only challenge positive evidence
        
        # Challenge skill claims
        if "skill" in evidence.claim.lower() or "corroborated" in evidence.claim.lower():
            return self._challenge_skill_claim(evidence, candidate)
        
        # Challenge career claims
        if "career" in evidence.claim.lower() or "trajectory" in evidence.claim.lower():
            return self._challenge_career_claim(evidence, candidate)
        
        return None
    
    def _detect_credential_inflation(self, candidate: dict) -> list:
        """
        Detect candidates who inflate their credentials.
        Expert claims should be backed by duration, endorsements,
        and career evidence.
        """
        evidence = []
        skills = candidate.get("skills", [])
        profile = candidate.get("profile", {})
        years_exp = profile.get("years_of_experience", 0)
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})
        
        expert_count = 0
        advanced_count = 0
        inflation_signals = 0
        
        for skill in skills:
            proficiency = skill.get("proficiency", "beginner")
            duration = skill.get("duration_months", 0)
            endorsements = skill.get("endorsements", 0)
            name = skill.get("name", "")
            
            if proficiency == "expert":
                expert_count += 1
                
                # Expert with 0 duration
                if duration == 0:
                    inflation_signals += 1
                    evidence.append(self._make_evidence(
                        claim=f"Expert claim with 0 months: {name}",
                        source="skills_list",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.4,
                        evidence_type=EvidenceType.SELF_REPORTED,
                        details=f"expert '{name}' but 0mo duration",
                    ))
                
                # Expert with very low duration
                elif duration < 6:
                    inflation_signals += 1
                
                # Expert but low assessment score
                name_lower = name.lower()
                for assessed_skill, score in assessments.items():
                    if name_lower in assessed_skill.lower() or assessed_skill.lower() in name_lower:
                        if score < 40:
                            evidence.append(self._make_evidence(
                                claim=f"Expert claim contradicted by assessment: {name}",
                                source="assessment_scores",
                                polarity=EvidencePolarity.NEGATIVE,
                                strength=0.7,
                                evidence_type=EvidenceType.CONTRADICTED,
                                details=f"claims expert '{name}' but scored {score}/100 on assessment",
                            ))
            
            elif proficiency == "advanced":
                advanced_count += 1
                if duration == 0:
                    inflation_signals += 1
        
        # Too many expert skills for experience level
        if expert_count >= 8 and years_exp < 5:
            evidence.append(self._make_evidence(
                claim=f"{expert_count} expert skills with only {years_exp} yrs experience",
                source="skills_list + profile",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.8,
                evidence_type=EvidenceType.CONTRADICTED,
                details=f"credential inflation: {expert_count} expert skills in {years_exp} yrs",
            ))
        
        # General inflation score
        if inflation_signals >= 3:
            evidence.append(self._make_evidence(
                claim=f"Systematic credential inflation ({inflation_signals} signals)",
                source="skills_list",
                polarity=EvidencePolarity.NEGATIVE,
                strength=min(0.8, 0.3 + inflation_signals * 0.1),
                evidence_type=EvidenceType.FORENSIC,
                details=f"systematic inflation: {inflation_signals} unsupported claims",
            ))
        
        return evidence
    
    def _detect_keyword_stuffing(self, candidate: dict) -> list:
        """
        Detect profiles that list many relevant keywords but show no
        evidence of actually using them. This is the JD's primary trap.
        """
        evidence = []
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        
        # Count relevant skills listed via FlashText
        skill_names = " ".join([s.get("name", "") for s in skills])
        relevant_skills = set(_all_relevant_processor.extract_keywords(skill_names))
        
        # Count ML evidence in career descriptions via FlashText
        all_desc = " ".join(j.get("description", "") for j in career)
        ml_evidence_count = len(_ml_processor.extract_keywords(all_desc))
        
        # Many skills listed but zero career evidence
        if len(relevant_skills) >= 5 and ml_evidence_count <= 1:
            evidence.append(self._make_evidence(
                claim="Keyword stuffing: many skills listed, minimal career evidence",
                source="skills_list vs career_description",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.85,
                evidence_type=EvidenceType.FORENSIC,
                details=f"keyword stuffing: {len(relevant_skills)} AI skills listed but only {ml_evidence_count} career refs",
            ))
        
        # Check title vs skills mismatch
        current_title = candidate.get("profile", {}).get("current_title", "").lower()
        is_irrelevant_title = any(
            t in current_title or current_title in t
            for t in IRRELEVANT_TITLES
        )
        
        if is_irrelevant_title and len(relevant_skills) >= 5:
            evidence.append(self._make_evidence(
                claim=f"Title '{current_title}' contradicts {len(relevant_skills)} AI skills",
                source="profile + skills_list",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.9,
                evidence_type=EvidenceType.CONTRADICTED,
                details=f"title-skill mismatch: '{current_title}' lists {len(relevant_skills)} AI skills",
            ))
        
        return evidence
    
    def _detect_career_red_flags(self, candidate: dict) -> list:
        """Detect career-level red flags."""
        evidence = []
        career = candidate.get("career_history", [])
        
        if not career:
            evidence.append(self._make_evidence(
                claim="No career history",
                source="career_history",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.8,
                evidence_type=EvidenceType.FORENSIC,
                details="no career history provided",
            ))
            return evidence
        
        # Job hopping detection
        if len(career) >= 3:
            tenures = [j.get("duration_months", 0) for j in career]
            short_stints = sum(1 for t in tenures if t < 12)
            if short_stints >= len(career) * 0.6:
                evidence.append(self._make_evidence(
                    claim="Job hopper: majority of roles < 12 months",
                    source="career_history",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.6,
                    evidence_type=EvidenceType.FORENSIC,
                    details=f"job hopper: {short_stints}/{len(career)} roles < 12 months",
                ))
        
        # Consulting-only career
        consulting_count = 0
        for job in career:
            company = job.get("company", "").lower()
            if any(c in company for c in CONSULTING_SERVICES_COMPANIES):
                consulting_count += 1
        
        if consulting_count == len(career) and len(career) >= 2:
            evidence.append(self._make_evidence(
                claim="Entire career at consulting/services companies",
                source="career_history",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.9,
                evidence_type=EvidenceType.FORENSIC,
                details="all-consulting career (JD explicit disqualifier)",
            ))
        
        # Non-AI career
        non_ai_count = sum(
            1 for j in career
            if sum(1 for kw in NON_AI_WORK_KEYWORDS if kw in j.get("description", "").lower()) >= 2
        )
        if non_ai_count >= len(career) * 0.7 and len(career) >= 2:
            evidence.append(self._make_evidence(
                claim=f"Predominantly non-AI career ({non_ai_count}/{len(career)} roles)",
                source="career_description",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.7,
                evidence_type=EvidenceType.FORENSIC,
                details=f"non-AI career: {non_ai_count}/{len(career)} roles",
            ))
        
        return evidence
    
    def _detect_skill_career_mismatch(self, candidate: dict) -> list:
        """
        Detect when listed skills don't match career descriptions.
        A candidate listing "PyTorch expert" whose career is all about
        accounting is suspicious.
        """
        evidence = []
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        
        # Count negative domain skills
        neg_count = 0
        for s in skills:
            name = s.get("name", "").lower()
            if any(ns in name or name in ns for ns in NEGATIVE_DOMAIN_SKILLS):
                neg_count += 1
        
        total_skills = len(skills)
        if total_skills > 0 and neg_count / total_skills > 0.4:
            evidence.append(self._make_evidence(
                claim=f"Wrong-domain skills dominate profile ({neg_count}/{total_skills})",
                source="skills_list",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.75,
                evidence_type=EvidenceType.FORENSIC,
                details=f"wrong-domain: {neg_count}/{total_skills} skills are non-tech",
            ))
        
        return evidence
    
    def _assess_recency(self, candidate: dict) -> list:
        """Assess whether the candidate's relevant experience is recent."""
        evidence = []
        career = candidate.get("career_history", [])
        
        # Check if current role is AI-related
        if career:
            current_jobs = [j for j in career if j.get("is_current", False)]
            if current_jobs:
                current = current_jobs[0]
                desc = current.get("description", "").lower()
                ml_hits = sum(1 for kw in ML_WORK_KEYWORDS if kw in desc)
                
                if ml_hits == 0:
                    evidence.append(self._make_evidence(
                        claim="Current role shows no ML/AI work",
                        source="career_description",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.5,
                        evidence_type=EvidenceType.FORENSIC,
                        details="current role has no ML signals",
                    ))
                else:
                    evidence.append(self._make_evidence(
                        claim="Current role involves ML/AI work",
                        source="career_description",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=0.5,
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"currently doing ML work ({ml_hits} signals)",
                    ))
        
        return evidence
    
    def _challenge_skill_claim(self, evidence: Evidence, candidate: dict):
        """Challenge a positive skill-related claim."""
        career = candidate.get("career_history", [])
        all_desc = " ".join(j.get("description", "").lower() for j in career)
        
        # If the claim is about a specific skill, check career for evidence
        claim_lower = evidence.claim.lower()
        
        # Extract skill name from claim
        for skill_cat in ["embeddings", "ranking", "recommendation", "nlp", "vector"]:
            if skill_cat in claim_lower:
                if skill_cat not in all_desc:
                    return self._make_evidence(
                        claim=f"No career evidence for claimed {skill_cat}",
                        source="career_description",
                        polarity=EvidencePolarity.NEGATIVE,
                        strength=0.5,
                        evidence_type=EvidenceType.CONTRADICTED,
                        details=f"challenged: no career evidence for {skill_cat}",
                    )
        
        return None
    
    def _challenge_career_claim(self, evidence: Evidence, candidate: dict):
        """Challenge a positive career-related claim."""
        # Challenge upward trajectory claims for consulting-only careers
        career = candidate.get("career_history", [])
        consulting_count = sum(
            1 for j in career
            if any(c in j.get("company", "").lower() for c in CONSULTING_SERVICES_COMPANIES)
        )
        
        if consulting_count == len(career) and "trajectory" in evidence.claim.lower():
            return self._make_evidence(
                claim="Upward trajectory at consulting firms is less meaningful",
                source="career_history",
                polarity=EvidencePolarity.NEGATIVE,
                strength=0.4,
                evidence_type=EvidenceType.FORENSIC,
                details="challenged: trajectory at consulting firms",
            )
        
        return None
