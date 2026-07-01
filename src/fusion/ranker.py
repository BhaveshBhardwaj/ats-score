"""
NEXUS Final Ranker

Information-theoretic ranking with debate-derived reasoning.
Produces the final top-100 CSV + XLSX submission.
"""

import csv
import math
from pathlib import Path
from typing import List, Dict

from agents import EvidencePolarity
from fusion.bayesian import compute_lcb_score


def generate_nexus_reasoning(fusion_result: Dict, verdicts: list, candidate: dict = None) -> str:
    """
    Generate reasoning from the multi-agent debate transcript.
    
    Unlike template-based reasoning, this produces specific,
    evidence-backed explanations that reference actual profile details.
    """
    parts = []
    
    # Lead with candidate-specific context when available
    if candidate:
        profile = candidate.get("profile", {})
        title = profile.get("current_title", "")
        company = profile.get("current_company", "")
        years = profile.get("years_of_experience", 0)
        if title and company:
            parts.append(f"{title} at {company} ({years}yr)")
        elif title:
            parts.append(f"{title} ({years}yr)")
    
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
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."
    
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
    
    # Generate output rows
    rows = []
    for rank, candidate_data in enumerate(top_candidates, start=1):
        fusion = candidate_data["fusion_result"]
        verdicts = candidate_data.get("verdicts", [])
        raw_candidate = candidate_data.get("_candidate")  # may be present from app.py
        
        reasoning = generate_nexus_reasoning(fusion, verdicts, raw_candidate)
        
        # Score normalization optimized for NDCG@10 (50% of hackathon score).
        # Use rank-weighted interpolation: map rank 1→~0.99, rank 100→~0.20.
        # Steeper exponential decay in top-10 creates maximum differentiation
        # where NDCG rewards it most.
        
        # Step 1: LCB-based relative position within top 100
        top_lcb = top_candidates[0]["lcb_score"]
        bot_lcb = top_candidates[-1]["lcb_score"]
        lcb_range = max(top_lcb - bot_lcb, 0.001)
        raw = candidate_data["lcb_score"]
        relative_position = (raw - bot_lcb) / lcb_range  # 0.0 to 1.0 within top 100
        
        # Step 2: map to final score range [0.20, 0.99]
        # Blend: 40% from actual LCB gaps + 60% from rank position
        # Steeper decay (0.035) maximizes NDCG@10 differentiation
        score_floor = 0.20
        score_ceil = 0.99
        rank_decay = math.exp(-0.035 * (rank - 1))
        blended = 0.4 * relative_position + 0.6 * rank_decay
        ats_score = score_floor + blended * (score_ceil - score_floor)
        ats_score = round(max(score_floor, min(score_ceil, ats_score)), 4)
        
        rows.append({
            "candidate_id": candidate_data["candidate_id"],
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
    
    # Write XLSX for portal submission
    xlsx_path = path.with_suffix(".xlsx")
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Submission"
        ws.append(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            ws.append([
                row["candidate_id"],
                int(row["rank"]),
                float(row["score"]),
                row["reasoning"],
            ])
        wb.save(xlsx_path)
        print(f"[OK] NEXUS wrote XLSX submission to {xlsx_path}")
    except ImportError:
        print("[WARN] openpyxl not installed — XLSX not generated. Run: pip install openpyxl")
    
    # Print top 10 summary
    print(f"\n{'='*80}")
    print(f"  NEXUS TOP 10 CANDIDATES")
    print(f"{'='*80}")
    for row in rows[:10]:
        print(f"  Rank {row['rank']:3d} | {row['candidate_id']} | Score: {row['score']} | {row['reasoning'][:70]}")
    print(f"{'='*80}\n")
    
    return rows
