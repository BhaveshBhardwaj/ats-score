"""
Agent 1: The Advocate

Actively searches for POSITIVE signals with a generous lens.
This agent's job is to find reasons TO hire a candidate, including
hidden gems that other scoring systems would miss.

Key capabilities:
- Latent skill detection from career descriptions
- Career upward trajectory analysis
- Adjacent domain transfer recognition
- Hidden gem identification (modest profiles, strong evidence)
"""

import re
from collections import Counter
from typing import Optional
from flashtext import KeywordProcessor

from agents import (
    BaseAgent, Evidence, Verdict,
    EvidenceType, EvidencePolarity,
)
from config import (
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, MLOPS_PRODUCTION_SKILLS,
    HIGHLY_RELEVANT_TITLES, MODERATELY_RELEVANT_TITLES,
    TECH_PRODUCT_INDUSTRIES, PRODUCTION_KEYWORDS, ML_WORK_KEYWORDS,
)


# Skills that can be inferred from career descriptions
# even if not explicitly listed in the skills section
LATENT_SKILL_PATTERNS = {
    "embeddings": [
        "embedding", "vector representation", "dense retrieval",
        "semantic search", "similarity search", "sentence-transformer",
        "text embedding", "sentence embedding",
    ],
    "ranking_systems": [
        "ranking system", "search ranking", "relevance ranking",
        "learning to rank", "re-ranking", "reranking",
        "candidate ranking", "result ranking",
    ],
    "recommendation_systems": [
        "recommendation system", "recommendation engine",
        "collaborative filtering", "content-based filtering",
        "personalization", "recommended", "suggestions engine",
    ],
    "vector_databases": [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss",
        "vector database", "vector store", "vector index",
        "approximate nearest neighbor", "ann search",
    ],
    "production_ml": [
        "deployed model", "model serving", "ml pipeline",
        "production model", "model in production", "real-time inference",
        "batch inference", "model monitoring", "a/b test",
        "shipped", "scaled to", "million users", "real users",
    ],
    "nlp_expertise": [
        "natural language", "text classification", "named entity",
        "sentiment analysis", "question answering", "text mining",
        "language model", "transformer", "bert", "attention mechanism",
    ],
    "evaluation_frameworks": [
        "ndcg", "mrr", "map", "precision@", "recall@",
        "offline evaluation", "online evaluation", "a/b testing",
        "evaluation framework", "benchmark", "metric",
    ],
}

# ── Initialize FlashText Processor ──────────────────────────────
_latent_processor = KeywordProcessor(case_sensitive=False)
for category, patterns in LATENT_SKILL_PATTERNS.items():
    for p in patterns:
        _latent_processor.add_keyword(p, category)

# Title progressions that indicate upward trajectory
TITLE_PROGRESSION_LEVELS = {
    "intern": 0, "trainee": 0,
    "junior": 1, "associate": 1,
    "mid": 2, "": 2,  # No prefix = mid-level
    "senior": 3, "lead": 3.5,
    "staff": 4, "principal": 4.5,
    "director": 5, "vp": 5.5,
    "head": 5, "chief": 6,
}


class AdvocateAgent(BaseAgent):
    """
    The Advocate agent — finds reasons TO hire a candidate.
    
    Biased toward finding positive signals. Uses generous matching
    and infers skills from career context. This agent ensures that
    hidden gems aren't missed by strict keyword matching.
    """
    
    AGENT_ID = "advocate"
    
    def evaluate(self, candidate: dict) -> Verdict:
        evidence = []
        
        # 1. Latent skill detection
        evidence.extend(self._detect_latent_skills(candidate))
        
        # 2. Career trajectory analysis
        evidence.extend(self._analyze_trajectory(candidate))
        
        # 3. Hidden gem detection
        evidence.extend(self._detect_hidden_gems(candidate))
        
        # 4. Adjacent domain transfer
        evidence.extend(self._assess_domain_transfer(candidate))
        
        # 5. Education quality signals
        evidence.extend(self._assess_education(candidate))
        
        # Compute advocate score from evidence
        positive_weight = sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.POSITIVE
        )
        negative_weight = abs(sum(
            e.effective_weight for e in evidence
            if e.polarity == EvidencePolarity.NEGATIVE
        ))
        
        # Advocate is biased toward positive — weights positive 1.5x
        raw_score = (positive_weight * 1.5) / max(positive_weight * 1.5 + negative_weight, 0.01)
        
        # Confidence based on amount of evidence found
        confidence = min(1.0, len(evidence) / 8.0)
        
        # Build reasoning from top evidence
        top = sorted(evidence, key=lambda e: abs(e.effective_weight), reverse=True)[:4]
        reasoning_parts = [e.details for e in top if e.details]
        
        return Verdict(
            agent_id=self.AGENT_ID,
            candidate_id=candidate.get("candidate_id", ""),
            score=min(1.0, raw_score),
            confidence=confidence,
            evidence=evidence,
            reasoning="; ".join(reasoning_parts),
        )
    
    def _detect_latent_skills(self, candidate: dict) -> list:
        """
        Detect skills from career descriptions even if not listed in skills.
        A candidate who "built a recommendation system" has recommendation
        skills even if they didn't list them.
        """
        evidence = []
        career = candidate.get("career_history", [])
        listed_skills = {
            s.get("name", "").lower().strip()
            for s in candidate.get("skills", [])
        }
        
        # Collect all career text
        all_desc = " ".join(
            job.get("description", "").lower() for job in career
        )
        profile_summary = candidate.get("profile", {}).get("summary", "").lower()
        all_text = all_desc + " " + profile_summary
        
        # FlashText O(N) extraction
        matches_list = _latent_processor.extract_keywords(all_text)
        match_counts = Counter(matches_list)
        
        for skill_category, matches in match_counts.items():
            if matches > 0:
                patterns = LATENT_SKILL_PATTERNS.get(skill_category, [])
                # Check if this skill is already explicitly listed
                already_listed = any(
                    skill_category.replace("_", " ") in s or
                    any(p in s for p in patterns[:3])
                    for s in listed_skills
                )
                
                if already_listed:
                    # Skill is both listed AND evidenced in career — corroborated
                    evidence.append(self._make_evidence(
                        claim=f"Skill '{skill_category}' corroborated by career",
                        source="career_description + skills_list",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=min(0.9, 0.4 + matches * 0.15),
                        evidence_type=EvidenceType.CORROBORATED,
                        details=f"{skill_category} confirmed by {matches} career references",
                    ))
                else:
                    # Latent skill — not listed but evidenced
                    evidence.append(self._make_evidence(
                        claim=f"Latent skill '{skill_category}' inferred from career",
                        source="career_description",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=min(0.7, 0.2 + matches * 0.15),
                        evidence_type=EvidenceType.INFERRED,
                        details=f"latent {skill_category} ({matches} career refs, not listed)",
                    ))
        
        return evidence
    
    def _analyze_trajectory(self, candidate: dict) -> list:
        """
        Analyze career trajectory for upward movement.
        A candidate going Junior → Mid → Senior at good companies
        is more promising than one going Senior → Senior → Senior.
        """
        evidence = []
        career = candidate.get("career_history", [])
        
        if len(career) < 2:
            return evidence
        
        # Sort by start_date to get chronological order
        sorted_career = sorted(
            career,
            key=lambda j: j.get("start_date", "1900-01-01"),
        )
        
        # Extract title levels
        title_levels = []
        for job in sorted_career:
            title = job.get("title", "").lower()
            level = 2.0  # Default mid-level
            for prefix, lvl in TITLE_PROGRESSION_LEVELS.items():
                if prefix and prefix in title:
                    level = max(level, lvl)
            title_levels.append(level)
        
        # Check for upward trajectory
        if len(title_levels) >= 2:
            trajectory = title_levels[-1] - title_levels[0]
            if trajectory > 0:
                evidence.append(self._make_evidence(
                    claim="Upward career trajectory",
                    source="career_history",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=min(0.8, 0.3 + trajectory * 0.2),
                    evidence_type=EvidenceType.INFERRED,
                    details=f"career trajectory: level {title_levels[0]:.0f}→{title_levels[-1]:.0f}",
                ))
            elif trajectory < -1:
                evidence.append(self._make_evidence(
                    claim="Downward career trajectory",
                    source="career_history",
                    polarity=EvidencePolarity.NEGATIVE,
                    strength=0.4,
                    evidence_type=EvidenceType.INFERRED,
                    details="downward career trajectory detected",
                ))
        
        # Check for increasing company quality using ratios to handle odd-length career histories
        n_first = len(sorted_career) // 2
        n_second = len(sorted_career) - n_first
        
        product_count_first_half = sum(
            1 for j in sorted_career[:n_first]
            if any(ind in j.get("industry", "").lower() for ind in TECH_PRODUCT_INDUSTRIES)
        )
        product_count_second_half = sum(
            1 for j in sorted_career[n_first:]
            if any(ind in j.get("industry", "").lower() for ind in TECH_PRODUCT_INDUSTRIES)
        )
        
        ratio_first = product_count_first_half / n_first if n_first > 0 else 0.0
        ratio_second = product_count_second_half / n_second if n_second > 0 else 0.0
        
        if ratio_second > ratio_first and ratio_second >= 0.5:
            evidence.append(self._make_evidence(
                claim="Moving toward product companies",
                source="career_history",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.INFERRED,
                details=f"trend toward product companies: {ratio_first:.0%} → {ratio_second:.0%}",
            ))
        
        return evidence
    
    def _detect_hidden_gems(self, candidate: dict) -> list:
        """
        Identify candidates whose profiles are modest but whose
        career evidence is strong. These are the people that keyword
        matching misses entirely.
        """
        evidence = []
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])
        skills = candidate.get("skills", [])
        
        # Count ML keywords in career descriptions
        all_desc = " ".join(j.get("description", "").lower() for j in career)
        ml_keyword_hits = sum(1 for kw in ML_WORK_KEYWORDS if kw in all_desc)
        production_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in all_desc)
        
        # Count explicitly listed relevant skills
        relevant_skill_count = 0
        for s in skills:
            name = s.get("name", "").lower()
            if any(ms in name or name in ms for ms in MUST_HAVE_SKILLS | CORE_AI_ML_SKILLS | NLP_IR_SKILLS):
                relevant_skill_count += 1
        
        # Hidden gem: strong career evidence but few listed skills (and actual description content)
        if ml_keyword_hits >= 5 and relevant_skill_count <= 3 and len(all_desc) > 200:
            evidence.append(self._make_evidence(
                claim="Hidden gem: strong career evidence despite few listed skills",
                source="career_description",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.7,
                evidence_type=EvidenceType.INFERRED,
                details=f"hidden gem: {ml_keyword_hits} ML career refs, only {relevant_skill_count} listed skills",
            ))
        
        # Production builder: evidence of shipping to real users
        if production_hits >= 3:
            evidence.append(self._make_evidence(
                claim="Evidence of production ML deployment",
                source="career_description",
                polarity=EvidencePolarity.POSITIVE,
                strength=min(0.9, 0.4 + production_hits * 0.1),
                evidence_type=EvidenceType.CORROBORATED,
                details=f"production signals: {production_hits} deployment references",
            ))
        
        return evidence
    
    def _assess_domain_transfer(self, candidate: dict) -> list:
        """
        Recognize transferable domain experience.
        A Search Engineer at Google → AI Engineer at startup is great.
        An ML Engineer at an e-commerce company has ranking/recommendation experience.
        """
        evidence = []
        career = candidate.get("career_history", [])
        
        transfer_domains = {
            "search": ["search", "information retrieval", "ranking", "relevance"],
            "recommendation": ["recommendation", "personalization", "discovery", "content feed"],
            "marketplace": ["marketplace", "matching", "two-sided", "supply demand"],
        }
        
        for job in career:
            desc = job.get("description", "").lower()
            title = job.get("title", "").lower()
            company = job.get("company", "")
            
            # Skip jobs with suspiciously short descriptions (prevents keyword stuffing)
            if len(desc) < 100:
                continue
            
            for domain, keywords in transfer_domains.items():
                hits = sum(1 for kw in keywords if kw in desc or kw in title)
                if hits >= 2:
                    evidence.append(self._make_evidence(
                        claim=f"Transferable {domain} domain experience at {company}",
                        source="career_description",
                        polarity=EvidencePolarity.POSITIVE,
                        strength=min(0.7, 0.3 + hits * 0.1),
                        evidence_type=EvidenceType.INFERRED,
                        details=f"transferable {domain} experience at {company}",
                    ))
                    break  # One domain match per job
        
        return evidence
    
    def _assess_education(self, candidate: dict) -> list:
        """Assess education quality as positive signal."""
        evidence = []
        education = candidate.get("education", [])
        
        for edu in education:
            tier = edu.get("tier", "unknown")
            field = edu.get("field_of_study", "").lower()
            degree = edu.get("degree", "").lower()
            
            # Tier-1 institution bonus
            if tier == "tier_1":
                evidence.append(self._make_evidence(
                    claim=f"Tier-1 institution: {edu.get('institution', '?')}",
                    source="education",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=0.4,
                    evidence_type=EvidenceType.CORROBORATED,
                    details=f"tier-1 education ({edu.get('institution', '?')})",
                ))
            
            # Relevant field bonus
            relevant_fields = [
                "computer science", "machine learning", "artificial intelligence",
                "data science", "statistics", "mathematics", "computational",
                "information technology", "software engineering",
            ]
            if any(rf in field for rf in relevant_fields):
                strength = 0.35 if "master" in degree or "phd" in degree or "m.tech" in degree else 0.2
                evidence.append(self._make_evidence(
                    claim=f"Relevant education: {degree} in {field}",
                    source="education",
                    polarity=EvidencePolarity.POSITIVE,
                    strength=strength,
                    evidence_type=EvidenceType.CORROBORATED,
                    details=f"relevant education: {degree} in {field}",
                ))
        
        return evidence

    def challenge(self, evidence: Evidence, candidate: dict) -> Optional[Evidence]:
        """
        The Advocate challenges negative evidence (Skeptic red flags) 
        if the candidate possesses strong compensating positive factors.
        """
        if evidence.polarity != EvidencePolarity.NEGATIVE:
            return None  # Only challenge negative evidence
            
        claim_lower = evidence.claim.lower()
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})
        education = candidate.get("education", [])
        
        # 1. Counter "consulting career" or "job hopper" flags if they have outstanding academic credentials
        has_tier1 = any(edu.get("tier") == "tier_1" for edu in education)
        if ("consulting" in claim_lower or "job hopper" in claim_lower) and has_tier1:
            return self._make_evidence(
                claim="Compensating academic pedigree: Tier-1 education",
                source="education",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.5,
                evidence_type=EvidenceType.FORENSIC,
                details="challenged: Tier-1 pedigree offsets consulting/hopper risk",
            )
            
        # 2. Counter "unsupported expert claim" or "keyword stuffing" flags if they have high assessment scores
        has_high_assessment = any(score >= 80 for score in assessments.values())
        if ("unsupported" in claim_lower or "stuffing" in claim_lower or "mismatch" in claim_lower) and has_high_assessment:
            top_skill = max(assessments, key=assessments.get)
            top_score = assessments[top_skill]
            return self._make_evidence(
                claim=f"Compensating tested competency: {top_skill} ({top_score}/100)",
                source="assessment_scores",
                polarity=EvidencePolarity.POSITIVE,
                strength=0.6,
                evidence_type=EvidenceType.FORENSIC,
                details=f"challenged: high tested skills ({top_skill}: {top_score}/100) corroborate capability",
            )
            
        return None
