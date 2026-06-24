"""
Adversarial Evidence Challenge Protocol (AECP)

When agents disagree about a candidate, this protocol runs
cross-examination: each agent can challenge another's evidence.
Evidence that survives challenge gains credibility.
Evidence that is contradicted loses credibility.

This is the core innovation of NEXUS — disagreement between
agents is a SIGNAL, not noise to be averaged away.
"""

from typing import List, Dict, Tuple
from agents import Evidence, Verdict, EvidencePolarity, BaseAgent


# Debate configuration
DISAGREEMENT_THRESHOLD = 0.25  # Score gap that triggers full debate
CHALLENGE_DISCOUNT = 0.5       # Credibility multiplier when challenged
SURVIVAL_BOOST = 1.15          # Credibility multiplier when challenge survives
MAX_CHALLENGES_PER_CANDIDATE = 10  # Cap for performance


def run_debate(
    verdicts: List[Verdict],
    agents: List[BaseAgent],
    candidate: dict,
) -> List[Verdict]:
    """
    Run the Adversarial Evidence Challenge Protocol.
    
    Steps:
    1. Identify disagreements between agents
    2. For each disagreement, run cross-examination
    3. Update evidence credibility based on challenges
    4. Return updated verdicts
    
    Args:
        verdicts: List of Verdict objects from each agent
        agents: List of agent instances (for challenge methods)
        candidate: The candidate being evaluated
    
    Returns:
        Updated list of Verdicts with modified evidence credibility
    """
    # Find max and min scores to identify disagreement
    if len(verdicts) < 2:
        return verdicts
    
    scores = [v.score for v in verdicts]
    max_score = max(scores)
    min_score = min(scores)
    
    # Check if debate is needed
    if max_score - min_score < DISAGREEMENT_THRESHOLD:
        # Agents mostly agree — mark evidence as unchallenged
        for verdict in verdicts:
            for evidence in verdict.evidence:
                if evidence.polarity == EvidencePolarity.POSITIVE:
                    evidence.credibility *= 1.05  # Slight boost for consensus
        return verdicts
    
    # Build agent lookup
    agent_map = {agent.AGENT_ID: agent for agent in agents}
    
    # Run cross-examination
    challenge_count = 0
    
    # High-scoring agents defend against low-scoring agents
    high_verdicts = [v for v in verdicts if v.score >= max_score - 0.1]
    low_verdicts = [v for v in verdicts if v.score <= min_score + 0.1]
    
    for high_v in high_verdicts:
        for low_v in low_verdicts:
            low_agent = agent_map.get(low_v.agent_id)
            if not low_agent:
                continue
            
            # Low scorer challenges high scorer's positive evidence
            for evidence in high_v.evidence:
                if challenge_count >= MAX_CHALLENGES_PER_CANDIDATE:
                    break
                
                if evidence.polarity != EvidencePolarity.POSITIVE:
                    continue
                if evidence.strength < 0.3:
                    continue  # Not worth challenging weak evidence
                
                counter = low_agent.challenge(evidence, candidate)
                challenge_count += 1
                
                if counter is not None:
                    # Challenge succeeded — discount the original evidence
                    evidence.credibility *= CHALLENGE_DISCOUNT
                    evidence.details += " [challenged]"
                    
                    # Add counter-evidence to the low scorer's verdict
                    counter.challenge_survived = True
                    low_v.evidence.append(counter)
                else:
                    # Challenge failed — evidence is more credible
                    evidence.credibility *= SURVIVAL_BOOST
                    evidence.challenge_survived = True
    
    # Also let high scorers challenge low scorers' negative evidence
    for low_v in low_verdicts:
        for high_v in high_verdicts:
            high_agent = agent_map.get(high_v.agent_id)
            if not high_agent:
                continue
            
            for evidence in low_v.evidence:
                if challenge_count >= MAX_CHALLENGES_PER_CANDIDATE:
                    break
                
                if evidence.polarity != EvidencePolarity.NEGATIVE:
                    continue
                if evidence.strength < 0.3:
                    continue
                
                counter = high_agent.challenge(evidence, candidate)
                challenge_count += 1
                
                if counter is not None:
                    evidence.credibility *= CHALLENGE_DISCOUNT
                    evidence.details += " [challenged]"
                    high_v.evidence.append(counter)
                else:
                    evidence.credibility *= SURVIVAL_BOOST
                    evidence.challenge_survived = True
    
    return verdicts


def compute_disagreement_profile(verdicts: List[Verdict]) -> Dict:
    """
    Compute a disagreement profile for logging and analysis.
    
    Returns:
        Dict with disagreement metrics
    """
    scores = [v.score for v in verdicts]
    
    if not scores:
        return {"spread": 0, "max": 0, "min": 0, "agents": {}}
    
    return {
        "spread": max(scores) - min(scores),
        "max": max(scores),
        "min": min(scores),
        "mean": sum(scores) / len(scores),
        "agents": {v.agent_id: v.score for v in verdicts},
        "needs_debate": (max(scores) - min(scores)) >= DISAGREEMENT_THRESHOLD,
    }


def aggregate_evidence(verdicts: List[Verdict]) -> List[Evidence]:
    """
    Collect all evidence from all agents into a single list,
    deduplicated by claim similarity.
    """
    all_evidence = []
    seen_claims = set()
    
    for verdict in verdicts:
        for evidence in verdict.evidence:
            # Simple dedup by claim text
            claim_key = evidence.claim.lower()
            if claim_key not in seen_claims:
                all_evidence.append(evidence)
                seen_claims.add(claim_key)
    
    return all_evidence
