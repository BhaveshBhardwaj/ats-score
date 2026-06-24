# NEXUS — Neural Evidence eXamination & Unified Scoring

**Multi-Agent Adversarial Reasoning for Intelligent Candidate Ranking**

A revolutionary approach to candidate ranking that goes beyond semantic search, embeddings, and knowledge graphs. Instead of one scoring function, NEXUS deploys **5 specialized AI agents** that independently evaluate candidates and then **debate each other** when they disagree. The final ranking emerges from adversarial consensus, not weighted averages.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download FastEmbed ONNX models (Network required, must run BEFORE offline ranking)
python src/setup_models.py

# Run NEXUS ranker (Takes < 3.5 minutes on CPU)
python src/rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
```

**One command.** Produces `submission.csv` with the top 100 ranked candidates, scored and reasoned by 5 adversarial agents.

## Architecture

```
100K candidates
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 0: Enhanced Pre-Filter                           │
│  Fast boolean elimination + honeypot detection          │
│  100K → ~99K candidates                                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 0.5: AI Triage (FlashText + FastEmbed)           │
│  FlashText O(N) extraction + CPU Semantic Embedding     │
│  ~99K → Top 15K elite candidates                        │
└───────────────────────┬─────────────────────────────────┘
                        │
    ▼           ▼           ▼           ▼           ▼
┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐ ┌────────────┐
│Advocate│ │ Skeptic  │ │Forensic │ │Trajectory│ │Availability│
│(Hire)  │ │(Don't    │ │ Auditor │ │ Analyst  │ │  Analyst   │
│        │ │ Hire)    │ │         │ │          │ │            │
│Finds   │ │Finds red │ │Cross-   │ │Monte     │ │Bayesian    │
│reasons │ │flags and │ │validates│ │Carlo     │ │hireability │
│TO hire │ │inflation │ │claims   │ │career    │ │estimation  │
└───┬────┘ └────┬─────┘ └───┬─────┘ └────┬─────┘ └─────┬──────┘
    │           │            │            │              │
    └─────┬─────┴──────┬─────┴─────┬──────┘              │
          │            │           │                      │
          ▼            ▼           ▼                      │
┌─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Adversarial Debate Protocol                   │
│  Agents cross-examine each other's evidence             │
│  Challenged evidence → credibility discount             │
│  Surviving evidence → credibility boost                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Bayesian Belief Fusion + LCB Ranking          │
│  Evidence → Belief network → Posterior P(fit | E)       │
│  Score - k×uncertainty = Lower Confidence Bound         │
│  High score + high confidence → top rank                │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
       Top 100 candidates with debate-derived reasoning
```

## The 5 Agents

### 🟢 The Advocate (finds reasons TO hire)
- **Latent skill detection**: Infers skills from career descriptions even when not listed (e.g., "built recommendation system" → has recommendation skills)
- **Hidden gem identification**: Finds candidates with modest profiles but strong career evidence
- **Career upward trajectory**: Detects candidates who are growing fast at good companies
- **Adjacent domain transfer**: Recognizes that Search Engineers and Recommendation Engineers have transferable skills

### 🔴 The Skeptic (finds reasons NOT to hire)
- **Credential inflation detector**: Cross-validates "expert" claims against assessment scores, career evidence, and endorsements
- **Keyword stuffing detector**: The JD's primary trap — many AI buzzwords listed but no career evidence of using them
- **Career red flags**: Job hopping, consulting-only careers, non-AI career dominance
- **Challenge protocol**: Can challenge other agents' positive evidence in debate

### 🔵 The Forensic Auditor (cross-validates everything)
- **Cross-Evidence Corroboration Matrix (CECM)**: For every skill claimed, checks 4 independent evidence sources (self-reported, career descriptions, assessments, endorsements)
- **Timeline integrity**: Detects impossible career timelines, education-experience mismatches
- **Assessment validation**: High assessment scores confirm claims; low scores contradict them
- **Endorsement anomaly detection**: High endorsements on zero-duration skills = suspicious

### 🟠 The Trajectory Analyst (Monte Carlo career simulation)
- **Career velocity scoring**: Not just "where are you now" but "how fast are you getting there"
- **Company quality trajectory**: Is the candidate moving toward better companies over time?
- **Role depth analysis**: Duration × description richness × company type = actual depth
- **Counterfactual modeling**: "Given this trajectory, how closely does it match the ideal for this JD?"

### 🟣 The Availability Analyst (Bayesian hireability)
- **Bayesian availability posterior**: P(hireable_now | all_signals) using proper Bayesian updating
- **Response probability**: Given response_rate + response_time + recency → P(they'll respond)
- **Salary negotiation risk**: P(salary negotiation succeeds) from expected vs offered
- **Notice period cost**: Discounts candidates based on how long until they can start

## Novel Algorithms

### 1. Cross-Evidence Corroboration Matrix (CECM)

For each skill, checks 4 independent evidence sources:

```
                    Skills List  Career Desc  Assessments  Endorsements
PyTorch Expert         ✓           ✓             ✓            ✓       → CORROBORATED (4/4)
Embeddings Expert      ✓           ✗             N/A          ✗       → UNCORROBORATED (1/4)
NLP Advanced           ✓           ✓             ✓            ✓       → CORROBORATED (4/4)
```

Skills corroborated by ≥2 sources keep full weight. Skills with only self-reported evidence are **discounted 60%**.

### 2. Adversarial Evidence Challenge Protocol (AECP)

When agents disagree (score gap > 0.25):
1. Low-scoring agents challenge high-scoring agents' positive evidence
2. High-scoring agents challenge low-scoring agents' negative evidence
3. Evidence that survives challenge → **credibility ×1.15**
4. Challenged evidence → **credibility ×0.5**

### 3. Bayesian Belief Fusion

Combines agent scores using both weighted fusion and Bayesian posterior updating:
```
P(good_fit | evidence) = update(prior, each_evidence_item)
final = 0.6 × agent_fusion + 0.4 × bayesian_posterior
```

### 4. Lower Confidence Bound Ranking (LCB)

```
ranking_score = posterior - k × uncertainty
```

A candidate with score 0.82 ± 0.05 (confident) ranks ABOVE a candidate with score 0.85 ± 0.20 (uncertain). Inspired by Thompson Sampling from multi-armed bandit theory.

### 5. AI Triage Engine (FlashText + FastEmbed)

To evaluate 100,000 candidates in under 5 minutes without losing semantic precision:
1. **FlashText (Aho-Corasick Algorithm)**: Used to instantly scan candidates for 100+ keywords in a single $O(N)$ pass. Drops processing time from 30 seconds to <2 seconds.
2. **FastEmbed (ONNX)**: For the elite candidates, we generate dense mathematical embeddings of their career summary and compare them against the Job Description using Cosine Similarity. Runs locally on CPU via ONNX Runtime without network API calls.

## Why This Approach Works

### What the JD actually says vs what most systems do

| JD Says | Most Systems Do | NEXUS Does |
|---------|----------------|------------|
| "Not keyword matching" | Keyword/embedding matching | Cross-validates skills against career evidence |
| "Career trajectory matters" | Counts years of experience | Monte Carlo trajectory simulation |
| "Services-only career is disqualifier" | Checks company name list | Tracks company quality trend over time |
| "Behavioral signals matter" | Simple thresholds | Bayesian posterior P(hireable \| signals) |
| "Honeypots exist" | Basic sanity checks | Forensic agent with multi-source cross-validation |

### What makes this different from academic approaches

| Approach | Academic Typical | NEXUS |
|----------|-----------------|-------|
| Scoring | Single model | Multi-agent adversarial debate |
| Skill eval | TF-IDF / embeddings | CECM: 4-source corroboration matrix |
| Career | Feature extraction → linear | Monte Carlo trajectory with counterfactuals |
| Availability | Binary threshold | Bayesian posterior with proper updating |
| Ranking | Sort by score | LCB with uncertainty quantification |
| Explainability | Feature importance | Full debate transcript |

## Project Structure

```
.
├── src/
│   ├── rank.py                 # NEXUS CLI entry point
│   ├── config.py               # JD-encoded configuration
│   ├── app.py                  # Streamlit demo app
│   ├── agents/                 # Multi-agent system
│   │   ├── __init__.py         # Base classes: Evidence, Verdict, BaseAgent
│   │   ├── advocate.py         # Agent 1: finds positive signals
│   │   ├── skeptic.py          # Agent 2: finds red flags + challenges
│   │   ├── forensic.py         # Agent 3: CECM cross-validation
│   │   ├── trajectory.py       # Agent 4: Monte Carlo career sim
│   │   └── availability.py     # Agent 5: Bayesian hireability
│   ├── fusion/                 # Debate & fusion protocol
│   │   ├── __init__.py
│   │   ├── debate.py           # Adversarial Evidence Challenge Protocol
│   │   ├── bayesian.py         # Bayesian belief fusion + LCB
│   │   └── ranker.py           # Final ranking + reasoning generation
│   └── pipeline/               # Core pipeline (inherited & enhanced)
│       ├── loader.py           # Streaming JSONL/JSON loader
│       ├── prefilter.py        # Stage 0: fast elimination
│       ├── honeypot.py         # Honeypot detection
│       ├── skills.py           # Skill taxonomy (used by agents)
│       ├── career.py           # Career analysis (used by agents)
│       ├── behavioral.py       # Behavioral scoring (used by agents)
│       ├── scorer.py           # Legacy composite scorer
│       └── ranker.py           # Legacy ranker
├── data/
│   ├── candidates.jsonl        # 100K candidate pool
│   └── sample_candidates.json
├── outputs/
│   └── submission.csv          # Final NEXUS output
├── requirements.txt
└── README.md
```

## Compute Constraints & Benchmark Proof

| Constraint | Limit | NEXUS v2 Performance |
|-----------|-------|-------------------|
| Runtime | < 5 min CPU | **~3.47 min (208s)** on 100K |
| Memory | < 16 GB RAM | Streaming loader, < 4 GB |
| GPU | Not required | Not used (Uses CPU ONNX) |
| Network | Not required | Model weights cached locally |
| Honeypots | < 10% | **0%** (21 caught in pre-filter) |

## Scoring Weights

NEXUS replaces fixed weights with dynamic agent-based evaluation:

| Agent | Weight | Role |
|-------|--------|------|
| Advocate | 20% | Positive signal detection |
| Skeptic | 25% | Red flag detection + challenges |
| Forensic | 25% | Objective cross-validation |
| Trajectory | 15% | Career path analysis |
| Availability | 15% | Hireability estimation |

Weights are further modulated by:
- Agent confidence (0-1)
- Evidence credibility (post-debate)
- Agent agreement level

## License

MIT
