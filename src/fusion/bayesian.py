"""
Bayesian Belief Fusion

Converts multi-agent evidence into a posterior probability of
candidate fitness using Bayesian belief propagation.

Produces:
- Posterior P(candidate is good fit | all evidence)
- Uncertainty quantification (entropy)
- Credibility-weighted evidence summary
"""

import math
from typing import List, Dict, Tuple
from agents import Evidence, Verdict, EvidencePolarity, EvidenceType


# Prior beliefs about candidate quality
PRIOR_GOOD_FIT = 0.05  # Base rate: ~5% of candidates are likely good fits

# Evidence type credibility priors
TYPE_CREDIBILITY = {
    EvidenceType.CORROBORATED: 1.0,    # Multi-source confirmation
    EvidenceType.FORENSIC: 0.9,         # Analytical finding
    EvidenceType.INFERRED: 0.6,         # Derived from indirect signals
    EvidenceType.SELF_REPORTED: 0.4,    # Only claimed by candidate
    EvidenceType.CONTRADICTED: 0.2,     # Contradicted by other evidence
}

# Agent expertise weights (how much to trust each agent's domain)
AGENT_WEIGHTS = {
    "advocate": 0.20,     # Finds positives — important but biased
    "skeptic": 0.25,      # Finds negatives — critical for avoiding bad hires
    "forensic": 0.25,     # Cross-validates — most objective
    "trajectory": 0.15,   # Career analysis — valuable but speculative
    "availability": 0.15, # Hireability — important but separate from fit
}


def bayesian_fusion(
    verdicts: List[Verdict],
    all_evidence: List[Evidence],
) -> Dict:
    """
    Fuse multi-agent verdicts into a single posterior score with
    uncertainty quantification.
    
    Returns:
        Dict with:
        - posterior: P(good fit | evidence)
        - uncertainty: Shannon entropy of the belief
        - confidence_interval: (low, high) 90% confidence interval
        - evidence_summary: top contributing evidence
        - agent_agreement: how much agents agree
    """
    # ── Method 1: Weighted agent score fusion ──────────────────────
    weighted_score = 0.0
    total_weight = 0.0
    
    for verdict in verdicts:
        agent_weight = AGENT_WEIGHTS.get(verdict.agent_id, 0.15)
        # Weight by both agent importance and verdict confidence
        effective_weight = agent_weight * verdict.confidence
        weighted_score += verdict.score * effective_weight
        total_weight += effective_weight
    
    if total_weight > 0:
        agent_fused_score = weighted_score / total_weight
    else:
        agent_fused_score = 0.5
    
    # ── Method 2: Evidence-based Bayesian update ──────────────────
    posterior = PRIOR_GOOD_FIT
    
    for evidence in all_evidence:
        # Skip neutral evidence
        if evidence.polarity == EvidencePolarity.NEUTRAL:
            continue
        
        # Compute likelihood ratios based on evidence
        type_cred = TYPE_CREDIBILITY.get(evidence.evidence_type, 0.5)
        eff_strength = evidence.strength * evidence.credibility * type_cred
        
        if evidence.polarity == EvidencePolarity.POSITIVE:
            # P(evidence | good_fit) is high, P(evidence | bad_fit) is low
            lr_if_true = 0.5 + 0.5 * eff_strength   # 0.5 to 1.0
            lr_if_false = 0.5 - 0.3 * eff_strength   # 0.2 to 0.5
        else:
            # P(evidence | good_fit) is low, P(evidence | bad_fit) is high
            lr_if_true = 0.5 - 0.4 * eff_strength    # 0.1 to 0.5
            lr_if_false = 0.5 + 0.4 * eff_strength   # 0.5 to 0.9
        
        # Bayesian update
        p_evidence = lr_if_true * posterior + lr_if_false * (1 - posterior)
        if p_evidence > 0:
            posterior = (lr_if_true * posterior) / p_evidence
        
        # Clip to avoid extreme values
        posterior = max(0.001, min(0.999, posterior))
    
    # ── Combine both methods ──────────────────────────────────────
    # Agent fusion is more stable, Bayesian is more principled
    # Blend them 60/40
    combined_score = 0.6 * agent_fused_score + 0.4 * posterior
    
    # ── Uncertainty quantification ────────────────────────────────
    # Shannon entropy of the belief
    p = combined_score
    if 0 < p < 1:
        entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    else:
        entropy = 0.0
    
    # Agent disagreement contributes to uncertainty
    agent_scores = [v.score for v in verdicts]
    if len(agent_scores) >= 2:
        score_variance = sum(
            (s - agent_fused_score) ** 2 for s in agent_scores
        ) / len(agent_scores)
        score_std = math.sqrt(score_variance)
    else:
        score_std = 0.2  # Default uncertainty
    
    # Combined uncertainty
    uncertainty = min(1.0, entropy * 0.5 + score_std * 0.5)
    
    # ── Confidence interval ───────────────────────────────────────
    # 90% confidence interval using uncertainty
    margin = 1.645 * score_std  # 90% z-score
    ci_low = max(0.0, combined_score - margin)
    ci_high = min(1.0, combined_score + margin)
    
    # ── Evidence summary ──────────────────────────────────────────
    # Top 5 most influential pieces of evidence
    sorted_evidence = sorted(
        all_evidence,
        key=lambda e: abs(e.effective_weight) * TYPE_CREDIBILITY.get(e.evidence_type, 0.5),
        reverse=True,
    )
    
    top_positive = [
        e for e in sorted_evidence
        if e.polarity == EvidencePolarity.POSITIVE
    ][:3]
    top_negative = [
        e for e in sorted_evidence
        if e.polarity == EvidencePolarity.NEGATIVE
    ][:2]
    
    # ── Agent agreement ───────────────────────────────────────────
    if len(agent_scores) >= 2:
        spread = max(agent_scores) - min(agent_scores)
        agreement = max(0.0, 1.0 - spread)
    else:
        agreement = 0.5
    
    return {
        "posterior": round(combined_score, 6),
        "uncertainty": round(uncertainty, 4),
        "confidence_interval": (round(ci_low, 4), round(ci_high, 4)),
        "entropy": round(entropy, 4),
        "agent_fused_score": round(agent_fused_score, 4),
        "bayesian_posterior": round(posterior, 4),
        "agent_agreement": round(agreement, 4),
        "top_positive_evidence": top_positive,
        "top_negative_evidence": top_negative,
        "evidence_count": len(all_evidence),
    }


def compute_lcb_score(fusion_result: Dict, k: float = 0.8) -> float:
    """
    Lower Confidence Bound scoring for robust ranking.
    
    LCB = score - k × uncertainty
    
    This ensures that candidates with high uncertainty are ranked
    conservatively, while candidates with confident high scores
    rank at the top.
    
    Inspired by Thompson Sampling / UCB from multi-armed bandit
    theory, but inverted (we want to avoid false positives).
    
    Args:
        fusion_result: Output of bayesian_fusion()
        k: Pessimism parameter (higher = more conservative)
    
    Returns:
        LCB score (can be negative for very uncertain candidates)
    """
    score = fusion_result["posterior"]
    uncertainty = fusion_result["uncertainty"]
    
    lcb = score - k * uncertainty
    
    # Boost for high agent agreement
    agreement_bonus = fusion_result["agent_agreement"] * 0.1
    lcb += agreement_bonus
    
    return max(0.0, lcb)
