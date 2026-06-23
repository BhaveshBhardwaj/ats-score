"""
Final ranking and reasoning generation.

Takes scored candidates, sorts them, selects top 100,
and generates per-candidate natural language reasoning.
"""

import csv
from pathlib import Path


def generate_reasoning(scored: dict) -> str:
    """
    Generate a specific, honest, 1-2 sentence reasoning for a candidate.
    References actual profile details — no templates or hallucinations.
    """
    parts = []
    
    career = scored["career"]
    skills = scored["skills"]
    experience = scored["experience"]
    behavioral = scored["behavioral"]
    location = scored["location"]
    
    # ── Lead with the strongest signal ────────────────────────────
    
    # Title and experience
    career_details = career.get("details", "")
    if "current:" in career_details:
        current_title = career_details.split("current:")[1].split(";")[0].strip()
    else:
        current_title = "Professional"
    
    years = experience.get("years", 0)
    parts.append(f"{current_title} with {years} yrs exp")
    
    # Key career signals
    if career["has_production_ml"]:
        parts.append("production ML experience")
    
    if career["is_consulting_only"]:
        parts.append("consulting-only career (concern)")
    
    # Skills summary
    must_have = skills.get("must_have_count", 0)
    core = skills.get("core_count", 0)
    if must_have > 0 or core > 0:
        skill_parts = []
        if must_have > 0:
            skill_parts.append(f"{must_have} must-have")
        if core > 0:
            skill_parts.append(f"{core} core AI/NLP")
        parts.append(f"skills: {', '.join(skill_parts)}")
    
    negative_skills = skills.get("negative_skills", [])
    if len(negative_skills) >= 3:
        parts.append(f"{len(negative_skills)} wrong-domain skills")
    
    # Behavioral highlights
    behavioral_details = behavioral.get("details", "")
    # Pick the most notable behavioral signals
    if "inactive" in behavioral_details:
        inactive_part = [p for p in behavioral_details.split(";") if "inactive" in p]
        if inactive_part:
            parts.append(inactive_part[0].strip())
    elif "recently active" in behavioral_details:
        parts.append("recently active")
    
    # Response rate
    if "response rate" in behavioral_details:
        rate_part = [p for p in behavioral_details.split(";") if "response rate" in p]
        if rate_part:
            parts.append(rate_part[0].strip())
    
    # Notice period concern
    if "notice" in behavioral_details:
        notice_part = [p for p in behavioral_details.split(";") if "notice" in p]
        if notice_part:
            notice_str = notice_part[0].strip()
            # Only mention if it's a concern
            try:
                days = int(notice_str.replace("notice", "").replace("d", "").strip())
                if days > 60:
                    parts.append(f"notice period {days}d (concern)")
            except ValueError:
                pass
    
    # GitHub signal for engineers
    if "GitHub" in behavioral_details:
        github_part = [p for p in behavioral_details.split(";") if "GitHub" in p]
        if github_part:
            parts.append(github_part[0].strip())
    
    # Location
    location_details = location.get("details", "")
    if "preferred location" in location_details:
        parts.append("preferred location")
    elif "outside India" in location_details:
        parts.append("outside India")
    
    # Combine into reasoning
    reasoning = "; ".join(parts)
    
    # Ensure it's not too long (keep under 200 chars for CSV cleanliness)
    if len(reasoning) > 200:
        reasoning = reasoning[:197] + "..."
    
    return reasoning


def rank_and_output(scored_candidates: list, output_path: str, top_n: int = 100):
    """
    Sort scored candidates, take top N, generate reasoning, and write CSV.
    
    Args:
        scored_candidates: list of dicts from score_candidate()
        output_path: path to write the CSV file
        top_n: number of candidates to include (default 100)
    """
    # Find max score to normalize out of 100
    max_score = max(c["composite_score"] for c in scored_candidates) if scored_candidates else 1.0
    
    # Sort by composite score descending
    # Ties broken by candidate_id ascending (per submission spec)
    scored_candidates.sort(
        key=lambda x: (-x["composite_score"], x["candidate_id"])
    )
    
    # Take top N
    top_candidates = scored_candidates[:top_n]
    
    # Generate output rows
    rows = []
    for rank, scored in enumerate(top_candidates, start=1):
        reasoning = generate_reasoning(scored)
        
        # Calculate normalized ATS score (out of 100)
        raw_score = scored["composite_score"]
        ats_score = min(100.0, max(0.0, (raw_score / max_score) * 100.0))
        
        rows.append({
            "candidate_id": scored["candidate_id"],
            "rank": rank,
            "score": f"{ats_score:.2f}",
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
    
    print(f"\n[OK] Wrote {len(rows)} ranked candidates to {output_path}")
    
    # Print top 10 summary
    print(f"\n{'='*80}")
    print(f"  TOP 10 CANDIDATES")
    print(f"{'='*80}")
    for row in rows[:10]:
        print(f"  Rank {row['rank']:3d} | {row['candidate_id']} | Score: {row['score']} | {row['reasoning'][:80]}")
    print(f"{'='*80}\n")
    
    return rows
