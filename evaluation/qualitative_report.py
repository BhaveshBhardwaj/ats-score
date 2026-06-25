import json
import csv
from pathlib import Path

def generate_report(submission_csv, candidates_json, output_md):
    # 1. Load candidates into a dictionary
    with open(candidates_json, "r", encoding="utf-8") as f:
        candidates_list = json.load(f)
    
    candidates = {c["candidate_id"]: c for c in candidates_list}
    
    # 2. Load submission CSV
    scored_candidates = []
    with open(submission_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored_candidates.append(row)
            
    if not scored_candidates:
        print("No candidates found in submission CSV.")
        return

    # 3. Pick Top 5, Middle 5, Bottom 5
    top_5 = scored_candidates[:5]
    
    mid_index = len(scored_candidates) // 2
    middle_5 = scored_candidates[max(0, mid_index-2):min(len(scored_candidates), mid_index+3)]
    
    bottom_5 = scored_candidates[-5:]

    def format_candidate(c_row):
        cid = c_row["candidate_id"]
        c_data = candidates.get(cid, {})
        rank = c_row["rank"]
        score = c_row["score"]
        justification = c_row.get("reasoning", "No justification provided.")
        
        profile = c_data.get("profile", {})
        title = profile.get("current_title", "N/A")
        yoe = profile.get("years_of_experience", 0)
        
        skills = ", ".join([s.get("name", "") for s in c_data.get("skills", [])][:8])
        
        career = c_data.get("career_history", [])
        recent_jobs = []
        for job in career[:2]:
            j_title = job.get("title", "N/A")
            j_company = job.get("company", "N/A")
            recent_jobs.append(f"**{j_title}** at {j_company}")
            
        jobs_str = "<br>".join(recent_jobs) if recent_jobs else "N/A"
        
        return f"""
#### Rank {rank}: {title} ({yoe} YOE) - Score: {score}
**Candidate ID:** `{cid}`

**Skills:** {skills}
**Recent Jobs:**
{jobs_str}

**NEXUS Justification:**
> {justification[:500]}...
"""

    # 4. Generate Markdown
    md_content = f"""# NEXUS Qualitative Profiling Report

This report compares the top-ranked candidates against borderline and rejected candidates to visually demonstrate the ranking quality.

## Top 5 Candidates (The Best Fits)
These candidates should clearly exhibit production ML/AI experience, strong semantic matching, and high availability.
"""
    for c in top_5:
        md_content += format_candidate(c)

    md_content += """
---
## Borderline Candidates (The Middle Pack)
These candidates might have some relevant skills but lack production experience, or have irrelevant backgrounds with keyword overlap.
"""
    for c in middle_5:
        md_content += format_candidate(c)

    md_content += """
---
## Rejected Candidates (The Bottom Pack)
These candidates are the lowest scoring of the batch, likely due to irrelevant experience, keyword stuffing, or lack of availability.
"""
    for c in bottom_5:
        md_content += format_candidate(c)

    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Qualitative report generated at: {output_md}")

if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    sub_csv = base_path / "evaluation" / "eval_submission.csv"
    cand_json = base_path / "data" / "eval_candidates.json"
    out_md = base_path / "evaluation" / "eval_report.md"
    
    # Fallback to sample data if eval data doesn't exist
    if not sub_csv.exists() or not cand_json.exists():
        sub_csv = base_path / "outputs" / "submission.csv"
        cand_json = base_path / "data" / "sample_candidates.json"
        
    generate_report(sub_csv, cand_json, out_md)
