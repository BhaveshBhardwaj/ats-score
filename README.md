# Redrob AI Candidate Ranker

**Intelligent Candidate Discovery & Ranking System** for the Redrob Hackathon.

Ranks 100K candidates for a Senior AI Engineer position by *actually understanding* who fits the role — not keyword matching.

## Quick Start

```bash
# Install dependencies (only Python 3.9+ standard library needed for ranking)
pip install -r requirements.txt

# Run the ranker
python src/rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
```

**That's it.** One command, produces `submission.csv` in the `outputs/` folder with the top 100 ranked candidates.

## Architecture

This system uses a **multi-stage pipeline** that processes 100K candidates in under 5 minutes on CPU:

```
100K candidates
    |
    v
[Stage 1: Fast Pre-Filter]     -- Experience range, honeypot detection, tech relevance
    |                              Eliminates ~70-80% of obviously unfit candidates
    v
[Stage 2: Multi-Dimensional Scoring]
    |   - Career Fit (35%)      -- Title relevance, company types, production ML experience
    |   - Skills Match (25%)    -- Semantic skill taxonomy matching, not keyword counting
    |   - Experience (15%)      -- Years + trajectory quality
    |   - Behavioral (15%)      -- Availability, responsiveness, GitHub, assessments
    |   - Location (10%)        -- India location preference, relocation willingness
    v
[Stage 3: Rank + Reason]       -- Composite scoring, tie-breaking, reasoning generation
    |
    v
Top 100 candidates with scores and reasoning (submission.csv)
```

## Why This Approach Works

### The JD is a trap detector, not a checklist

The job description for "Senior AI Engineer — Founding Team" at Redrob AI explicitly warns against keyword matching. It says:

> *"The right answer is not 'find candidates whose skills section contains the most AI keywords.' That's a trap we've explicitly built into the dataset."*

Our system handles this by:

1. **Career trajectory analysis over keyword counting**: A candidate titled "Marketing Manager" with 9 AI skills listed is scored low. A "Search Engineer" at a product company with 3 relevant skills is scored high.

2. **Company type classification**: The JD explicitly disqualifies candidates whose entire career is at consulting/services companies (TCS, Infosys, Wipro, etc.). Our system detects this.

3. **Production ML signals**: We analyze career descriptions for evidence of actual production deployment — "shipped", "deployed", "scaled", "A/B testing" — not just "worked on ML projects".

4. **Honeypot detection**: The dataset contains ~80 candidates with subtly impossible profiles (e.g., 8 years at a company founded 3 years ago, "expert" in 10 skills with 0 duration). Our system catches these through cross-validation of dates, durations, and proficiency claims.

5. **Behavioral signal integration**: A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate gets down-weighted, per the JD's explicit guidance.

### Semantic skill matching without ML models

Instead of expensive embedding models, we use a **carefully crafted skill taxonomy** with fuzzy matching:

- **Must-have skills** (3x weight): embeddings, vector DBs, FAISS, Pinecone, ranking, Python
- **NLP/IR skills** (2.5x weight): NLP, transformers, BERT, information retrieval
- **Core AI/ML** (2x weight): PyTorch, scikit-learn, deep learning
- **Nice-to-have** (1.5x weight): LoRA, fine-tuning, LLM experience
- **Wrong-domain skills** (-2x penalty): Photoshop, accounting, CAD, mechanical design

Skills are weighted by proficiency level, duration of use, and endorsement count — not just presence/absence.

## Project Structure

```
.
├── src/                        # Source code
│   ├── rank.py                 # CLI entry point
│   ├── config.py               # JD-encoded configuration, skill taxonomy, weights
│   ├── pipeline/               # Core pipeline modules
│   │   ├── loader.py           # Streaming JSONL/JSON data loader
│   │   ├── prefilter.py        # Stage 1: fast boolean/range elimination
│   │   ├── skills.py           # Semantic skill matching engine
│   │   ├── career.py           # Career trajectory analysis
│   │   ├── behavioral.py       # Behavioral signal scoring
│   │   ├── scorer.py           # Composite multi-dimensional scorer
│   │   └── ranker.py           # Final ranking + reasoning generation
│   └── app.py                  # Streamlit demo app (sandbox)
├── data/                       # Datasets
│   ├── candidates.jsonl        # (100K file)
│   └── sample_candidates.json  
├── outputs/                    # Output directory
│   └── submission.csv          # Final ranked output
├── docs/                       # Challenge documentation & references
│   ├── job_description.txt
│   ├── submission_spec.txt
│   └── ...
├── requirements.txt
└── README.md
```

## Compute Constraints

- **Runtime**: < 5 minutes on CPU (typically ~2-3 min)
- **Memory**: < 16GB RAM (streaming loader, no full dataset in memory)
- **GPU**: Not required, not used
- **Network**: Not required during ranking
- **Dependencies**: Python 3.9+ standard library only (no ML models needed)

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Rule-based over ML models | 5-min CPU constraint; skill taxonomy is more interpretable than embeddings for this task |
| Streaming loader | 487MB JSONL file; can't load all 100K candidates into memory at once on constrained systems |
| Multi-dimensional composite score | Mirrors how a great recruiter thinks: career fit first, then skills, then availability |
| Multiplicative disqualifiers | Consulting-only career or honeypot detection = near-zero score, regardless of skills |
| Word-level title matching | "Recommendation Systems Engineer" should match "recommendation engineer" — substring fails here |

## Scoring Weights

These weights reflect the JD's priorities:

| Dimension | Weight | Why |
|-----------|--------|-----|
| Career Fit | 35% | The JD cares most about career trajectory at product companies |
| Skills Relevance | 25% | Must-have skills are important but not the primary signal |
| Experience Quality | 15% | 5-9 years ideal, flexible outside |
| Behavioral Signals | 15% | Availability and responsiveness determine if you can actually hire them |
| Location Fit | 10% | India preferred, Pune/Noida ideal |

## License

MIT
