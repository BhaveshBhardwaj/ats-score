import csv
import math
from pathlib import Path

def generate_comparison_report(nexus_csv, llm_csv, out_md):
    nexus_data = {}
    llm_data = {}
    
    # 1. Load NEXUS Rankings
    try:
        with open(nexus_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["candidate_id"]
                nexus_data[cid] = {
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "reasoning": row.get("reasoning", "")
                }
    except Exception as e:
        print(f"Error reading {nexus_csv}: {e}")
        return

    # 2. Load LLM Rankings
    try:
        with open(llm_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["candidate_id"]
                llm_data[cid] = {
                    "rank": int(row["rank"]),
                    "score": int(row["llm_score"])
                }
    except Exception as e:
        print(f"Error reading {llm_csv}: {e}")
        return

    # 3. Find intersection of candidates
    common_cids = set(nexus_data.keys()).intersection(set(llm_data.keys()))
    if not common_cids:
        print("No intersecting candidates found between the two CSVs.")
        return

    # 4. Compute Spearman's Rank Correlation
    n = len(common_cids)
    sum_d_squared = 0
    for cid in common_cids:
        d = nexus_data[cid]["rank"] - llm_data[cid]["rank"]
        sum_d_squared += d**2
        
    if n > 1:
        spearman_rho = 1 - ((6 * sum_d_squared) / (n * (n**2 - 1)))
    else:
        spearman_rho = 0.0

    # 5. Top 10 Overlap
    nexus_top_10 = set([cid for cid in common_cids if nexus_data[cid]["rank"] <= 10])
    llm_top_10 = set([cid for cid in common_cids if llm_data[cid]["rank"] <= 10])
    top_10_overlap = nexus_top_10.intersection(llm_top_10)

    # 6. Golden Candidate Performance
    golden_cids = [cid for cid in common_cids if cid.startswith("GOLDEN")]
    
    md = f"""# NEXUS vs LLM Comparison Report

This report compares the highly-optimized multi-agent NEXUS pipeline against a massive zero-shot baseline (`openai/gpt-oss-120b`).

### Statistical Overview
* **Total Candidates Evaluated:** {n}
* **Spearman's Rank Correlation ($\rho$):** `{spearman_rho:.3f}` 
  *(1.0 is perfect agreement, 0 is random, -1.0 is total disagreement)*
* **Top 10 Overlap:** {len(top_10_overlap)}/10 candidates ({len(top_10_overlap)*10}%)

### Synthetic "Golden" Candidate Performance
The true test of an ATS pipeline is whether it correctly identifies perfect resumes synthetically injected into the pool.

| Candidate ID | NEXUS Rank | LLM Rank |
|---|---|---|
"""
    for cid in sorted(golden_cids):
        n_rank = nexus_data[cid]["rank"]
        l_rank = llm_data[cid]["rank"]
        md += f"| `{cid}` | **#{n_rank}** | **#{l_rank}** |\n"
        
    md += """
### Biggest Disagreements
Where did the two models disagree the most? (Highest absolute rank difference)

| Candidate ID | NEXUS Rank | LLM Rank | Difference |
|---|---|---|---|
"""
    disagreements = []
    for cid in common_cids:
        diff = abs(nexus_data[cid]["rank"] - llm_data[cid]["rank"])
        disagreements.append((cid, diff))
        
    disagreements.sort(key=lambda x: -x[1])
    for cid, diff in disagreements[:5]:
        n_rank = nexus_data[cid]["rank"]
        l_rank = llm_data[cid]["rank"]
        md += f"| `{cid}` | #{n_rank} | #{l_rank} | $\\Delta$ {diff} |\n"

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Comparison report generated at {out_md}")

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    generate_comparison_report(
        nexus_csv=base / "evaluation" / "eval_submission.csv",
        llm_csv=base / "evaluation" / "llm_submission.csv",
        out_md=base / "evaluation" / "comparison_report.md"
    )
