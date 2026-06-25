# NEXUS vs LLM Comparison Report

This report compares the highly-optimized multi-agent NEXUS pipeline against a massive zero-shot baseline (`openai/gpt-oss-120b`).

### Statistical Overview
* **Total Candidates Evaluated:** 60
* **Spearman's Rank Correlation ($ho$):** `0.264` 
  *(1.0 is perfect agreement, 0 is random, -1.0 is total disagreement)*
* **Top 10 Overlap:** 4/10 candidates (40%)

### Synthetic "Golden" Candidate Performance
The true test of an ATS pipeline is whether it correctly identifies perfect resumes synthetically injected into the pool.

| Candidate ID | NEXUS Rank | LLM Rank |
|---|---|---|
| `GOLDEN_0001` | **#10** | **#3** |
| `GOLDEN_0002` | **#11** | **#4** |
| `GOLDEN_0003` | **#12** | **#5** |
| `GOLDEN_0004` | **#13** | **#6** |
| `GOLDEN_0005` | **#14** | **#1** |

### Biggest Disagreements
Where did the two models disagree the most? (Highest absolute rank difference)

| Candidate ID | NEXUS Rank | LLM Rank | Difference |
|---|---|---|---|
| `CAND_0000045` | #2 | #50 | $\Delta$ 48 |
| `CAND_0000003` | #59 | #13 | $\Delta$ 46 |
| `CAND_0000043` | #56 | #11 | $\Delta$ 45 |
| `CAND_0000006` | #60 | #16 | $\Delta$ 44 |
| `CAND_0000012` | #55 | #22 | $\Delta$ 33 |
