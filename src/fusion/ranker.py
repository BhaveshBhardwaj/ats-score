"""
NEXUS Final Ranker

Information-theoretic ranking with debate-derived reasoning.
Produces the final top-100 CSV submission.
"""

import csv
from pathlib import Path
from typing import List, Dict

from agents import EvidencePolarity
from fusion.bayesian import compute_lcb_score


def generate_nexus_reasoning(fusion_result: Dict, verdicts: list) -> str:
    """
    Generate reasoning from the multi-agent debate transcript.
    
    Unlike template-based reasoning, this produces specific,
    evidence-backed explanations that reference actual profile details.
    """
    parts = []
    
    # Lead with the strongest positive evidence
    top_positive = fusion_result.get("top_positive_evidence", [])
    top_negative = fusion_result.get("top_negative_evidence", [])
    
    for ev in top_positive[:2]:
        if ev.details:
            detail = ev.details
            if ev.challenge_survived:
                detail += " (debate-validated)"
            parts.append(detail)
    
    # Add key concerns
    for ev in top_negative[:1]:
        if ev.details:
            parts.append(ev.details)
    
    # Add confidence signal
    ci_low, ci_high = fusion_result.get("confidence_interval", (0, 1))
    agreement = fusion_result.get("agent_agreement", 0)
    
    if agreement >= 0.8:
        parts.append("high agent consensus")
    elif agreement < 0.4:
        parts.append("agents disagree significantly")
    
    # Compose
    reasoning = "; ".join(parts)
    
    # Trim for CSV cleanliness
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."
    
    return reasoning if reasoning else "Insufficient evidence for detailed reasoning"


def rank_and_output_nexus(
    scored_candidates: List[Dict],
    output_path: str,
    top_n: int = 100,
) -> list:
    """
    Rank candidates using LCB scores and output CSV.
    
    Args:
        scored_candidates: List of dicts from the NEXUS pipeline
            Each dict has:
            - candidate_id
            - fusion_result (from bayesian_fusion)
            - verdicts (list of agent Verdicts)
            - lcb_score (from compute_lcb_score)
        output_path: Path to write CSV
        top_n: Number of candidates to include
    
    Returns:
        List of output row dicts
    """
    # Sort by LCB score descending, ties broken by candidate_id
    scored_candidates.sort(
        key=lambda x: (-x["lcb_score"], x["candidate_id"])
    )
    
    # Take top N
    top_candidates = scored_candidates[:top_n]
    
    if not top_candidates:
        print("[WARN] No candidates to rank!")
        return []
    
    # Normalize against the FULL evaluated pool, not just top N.
    # The top 100 out of 15,000 are the elite — their scores should
    # cluster at the high end of the 0-1 range.
    all_lcb = [c["lcb_score"] for c in scored_candidates]
    pool_max = max(all_lcb)
    pool_min = min(all_lcb)
    pool_range = max(pool_max - pool_min, 0.001)
    
    # Generate output rows
    rows = []
    for rank, candidate in enumerate(top_candidates, start=1):
        fusion = candidate["fusion_result"]
        verdicts = candidate.get("verdicts", [])
        
        reasoning = generate_nexus_reasoning(fusion, verdicts)
        
        # Normalize against full pool — top 100 will naturally land in the
        # upper portion of the 0.0-1.0 range since they're the best of 15K
        raw = candidate["lcb_score"]
        ats_score = max(0.0, min(1.0, (raw - pool_min) / pool_range))
        
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank": rank,
            "score": f"{ats_score:.4f}",
            "reasoning": reasoning,
        })
    
    # Write CSV
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n[OK] NEXUS wrote {len(rows)} ranked candidates to {output_path}")
    
    # Print top 10 summary
    print(f"\n{'='*80}")
    print(f"  NEXUS TOP 10 CANDIDATES")
    print(f"{'='*80}")
    for row in rows[:10]:
        print(f"  Rank {row['rank']:3d} | {row['candidate_id']} | Score: {row['score']} | {row['reasoning'][:70]}")
    print(f"{'='*80}\n")
    
    return rows
