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
    
    # Normalize against the FULL evaluated pool using percentile ranking.
    # The top 100 out of 15,000 are the elite — their scores should show
    # meaningful differentiation. We use percentile rank within the pool
    # so that rank #1 ≈ 0.99 and rank #100 ≈ 0.20 (natural spread).
    pool_size = len(scored_candidates)
    
    # Generate output rows
    rows = []
    for rank, candidate in enumerate(top_candidates, start=1):
        fusion = candidate["fusion_result"]
        verdicts = candidate.get("verdicts", [])
        
        reasoning = generate_nexus_reasoning(fusion, verdicts)
        
        # Percentile-based score: candidate at position i in a sorted pool
        # of N gets percentile = (N - i) / N. For top 100 out of 15K this
        # gives scores from 0.9999 to 0.9934 — still too narrow.
        # Instead, use rank-weighted interpolation: map rank 1→~0.99,
        # rank 100→~0.20, with non-linear decay that reflects the actual
        # LCB score gaps between candidates.
        
        # Step 1: raw percentile within the full pool
        # (candidate is at position `rank-1` in a pool of `pool_size`)
        raw_percentile = 1.0 - ((rank - 1) / pool_size)
        
        # Step 2: blend with the actual LCB-based relative position
        # to preserve the real score gaps between candidates
        top_lcb = top_candidates[0]["lcb_score"]
        bot_lcb = top_candidates[-1]["lcb_score"]
        lcb_range = max(top_lcb - bot_lcb, 0.001)
        raw = candidate["lcb_score"]
        relative_position = (raw - bot_lcb) / lcb_range  # 0.0 to 1.0 within top 100
        
        # Step 3: map to final score range [0.20, 0.99]
        # Blend: 40% from actual LCB gaps + 60% from rank position (exponentially decayed to optimize for NDCG@10)
        # This preserves real differentiation while ensuring good spread and penalizing lower ranks more heavily
        import math
        score_floor = 0.20
        score_ceil = 0.99
        rank_decay = math.exp(-0.018 * (rank - 1))
        blended = 0.4 * relative_position + 0.6 * rank_decay
        ats_score = score_floor + blended * (score_ceil - score_floor)
        ats_score = round(max(score_floor, min(score_ceil, ats_score)), 4)
        
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
