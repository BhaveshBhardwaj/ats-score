#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker — NEXUS Architecture

Neural Evidence eXamination & Unified Scoring

Multi-agent adversarial reasoning system for intelligent
candidate discovery and ranking.

Architecture:
  Stage 0: Enhanced Pre-Filter (100K → ~95K)
  Stage 0.5: Fast Triage (95K → ~12K)
  Stage 1: 5-Agent Parallel Evaluation
  Stage 2: Adversarial Debate Protocol
  Stage 3: Bayesian Belief Fusion + LCB Ranking
  Stage 4: Reasoning Generation + CSV Output

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --limit 1000
"""

import argparse
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.loader import load_candidates
from pipeline.prefilter import prefilter
from pipeline.triage import triage_candidates

# NEXUS imports
from agents.advocate import AdvocateAgent
from agents.skeptic import SkepticAgent
from agents.forensic import ForensicAgent
from agents.trajectory import TrajectoryAgent
from agents.availability import AvailabilityAgent
from fusion.debate import run_debate, compute_disagreement_profile, aggregate_evidence
from fusion.bayesian import bayesian_fusion, compute_lcb_score
from fusion.ranker import rank_and_output_nexus


def run_nexus_pipeline(candidate, agents):
    """
    Run the full NEXUS pipeline on a single candidate.
    
    Returns a dict with all NEXUS results or None if candidate
    should be skipped.
    """
    # Stage 1: All agents evaluate independently
    verdicts = [agent.evaluate(candidate) for agent in agents]
    
    # Stage 2: Adversarial debate (only if agents disagree)
    disagreement = compute_disagreement_profile(verdicts)
    
    if disagreement["needs_debate"]:
        verdicts = run_debate(verdicts, agents, candidate)
    
    # Stage 3: Bayesian belief fusion
    all_evidence = aggregate_evidence(verdicts)
    fusion_result = bayesian_fusion(verdicts, all_evidence)
    
    # Stage 4: LCB score for robust ranking
    lcb = compute_lcb_score(fusion_result)
    
    # Stage 4.5: Post-fusion availability multiplier
    # The JD says: "A perfect-on-paper candidate who hasn't logged in for
    # 6 months and has a 5% response rate is not actually available."
    signals = candidate.get("redrob_signals", {})
    
    # Availability boost/penalty
    open_to_work = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.5)
    last_active = signals.get("last_active_date", "")
    notice_days = signals.get("notice_period_days", 60)
    
    # Check recency (simple heuristic: if last_active contains "2026" or "2025", they're recent)
    is_recent = False
    if last_active:
        for year in ("2026", "2025"):
            if year in str(last_active):
                is_recent = True
                break
    
    # Availability multiplier
    if open_to_work and response_rate > 0.5 and is_recent:
        lcb *= 1.08  # Strong availability signal
    elif response_rate < 0.15 or not is_recent:
        lcb *= 0.88  # Effectively unavailable
    
    # Tiered notice period (JD: "sub-30 day notice preferred, 30+ bar gets higher")
    if notice_days <= 30:
        lcb *= 1.04
    elif notice_days <= 60:
        pass  # neutral
    elif notice_days <= 90:
        lcb *= 0.98
    else:
        lcb *= 0.94  # 90+ days is a significant friction
    
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "verdicts": verdicts,
        "fusion_result": fusion_result,
        "lcb_score": lcb,
        "disagreement": disagreement,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NEXUS — Neural Evidence eXamination & Unified Scoring"
    )
    parser.add_argument(
        "--candidates", "-c",
        required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out", "-o",
        default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Process only first N candidates (for testing)"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=100,
        help="Number of top candidates to output (default: 100)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    args = parser.parse_args()
    
    # Validate input
    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)
    
    print("")
    print("=" * 68)
    print("  NEXUS — Neural Evidence eXamination & Unified Scoring")
    print("  Multi-Agent Adversarial Reasoning for Candidate Ranking")
    print("")
    print("  Agents: Advocate | Skeptic | Forensic | Trajectory | Availability")
    print("  Protocol: Pre-Filter → Triage → Evaluate → Debate → Fuse → Rank")
    print("=" * 68)
    print("")
    
    start_time = time.time()
    
    # ── Initialize Agents ─────────────────────────────────────────
    print("[INIT] Initializing NEXUS agents...")
    agents = [
        AdvocateAgent(),
        SkepticAgent(),
        ForensicAgent(),
        TrajectoryAgent(),
        AvailabilityAgent(),
    ]
    print(f"   [OK] {len(agents)} agents ready")
    
    # ── Stage 0: Pre-Filter ───────────────────────────────────────
    print(f"\n[Stage 0] Loading & pre-filtering candidates...")
    print(f"   Source: {candidates_path}")
    if args.limit:
        print(f"   Limit: {args.limit} candidates")
    
    total_count = 0
    filtered_count = 0
    honeypot_count = 0
    passed_candidates = []
    
    stage0_start = time.time()
    
    for candidate in load_candidates(str(candidates_path), limit=args.limit):
        total_count += 1
        
        if total_count % 10000 == 0:
            elapsed = time.time() - stage0_start
            rate = total_count / elapsed
            print(f"   Processed {total_count:,} candidates ({rate:.0f}/sec) — {len(passed_candidates)} passed")
        
        passes, reason = prefilter(candidate)
        
        if not passes:
            filtered_count += 1
            if "Honeypot" in reason:
                honeypot_count += 1
            if args.verbose and total_count <= 20:
                cid = candidate.get("candidate_id", "?")
                print(f"   [FILTERED] {cid}: {reason}")
            continue
        
        passed_candidates.append(candidate)
    
    stage0_time = time.time() - stage0_start
    
    print(f"\n   [OK] Stage 0 complete in {stage0_time:.1f}s")
    print(f"   Total processed: {total_count:,}")
    print(f"   Filtered out: {filtered_count:,} ({filtered_count/max(total_count,1)*100:.1f}%)")
    print(f"   Honeypots: {honeypot_count}")
    print(f"   Pre-filter passed: {len(passed_candidates):,}")
    
    # ── Stage 0.5: Fast Triage ────────────────────────────────────
    # Reduce candidate pool to manageable size for NEXUS
    MAX_NEXUS_CANDIDATES = 5000
    
    if len(passed_candidates) > MAX_NEXUS_CANDIDATES:
        print(f"\n[Stage 0.5] Fast triage: {len(passed_candidates):,} → top {MAX_NEXUS_CANDIDATES:,}...")
        
        triage_start = time.time()
        
        # Use the dedicated triage function (FlashText + FastEmbed Semantic boost)
        triage_results = triage_candidates(passed_candidates, MAX_NEXUS_CANDIDATES)
        
        triage_time = time.time() - triage_start
        
        print(f"   [OK] Triage complete in {triage_time:.1f}s")
        print(f"   Candidates for NEXUS: {len(triage_results):,}")
    else:
        triage_results = passed_candidates
        print(f"\n[Stage 0.5] Triage skipped (pool size {len(passed_candidates):,} ≤ {MAX_NEXUS_CANDIDATES:,})")
    
    # ── Stage 1-3: NEXUS Evaluation ──────────────────────────────
    print(f"\n[Stage 1-3] Running NEXUS multi-agent evaluation...")
    print(f"   5 agents × {len(triage_results):,} candidates")
    
    stage1_start = time.time()
    nexus_results = []
    debate_count = 0
    
    for i, candidate in enumerate(triage_results):
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - stage1_start
            rate = (i + 1) / elapsed
            print(f"   Evaluated {i+1:,}/{len(triage_results):,} ({rate:.0f}/sec) — {debate_count} debates triggered")
        
        result = run_nexus_pipeline(candidate, agents)
        if result:
            nexus_results.append(result)
            if result["disagreement"].get("needs_debate", False):
                debate_count += 1
    
    stage1_time = time.time() - stage1_start
    
    print(f"\n   [OK] Stages 1-3 complete in {stage1_time:.1f}s")
    print(f"   Candidates evaluated: {len(nexus_results):,}")
    print(f"   Debates triggered: {debate_count} ({debate_count/max(len(nexus_results),1)*100:.1f}%)")
    
    # Score statistics
    if nexus_results:
        lcb_scores = [r["lcb_score"] for r in nexus_results]
        print(f"   LCB scores: max={max(lcb_scores):.4f}, min={min(lcb_scores):.4f}, mean={sum(lcb_scores)/len(lcb_scores):.4f}")
        
        uncertainties = [r["fusion_result"]["uncertainty"] for r in nexus_results]
        print(f"   Uncertainty: mean={sum(uncertainties)/len(uncertainties):.4f}")
    
    # ── Stage 4: Final Ranking ────────────────────────────────────
    print(f"\n[Stage 4] Generating final ranking (top {args.top})...")
    
    stage4_start = time.time()
    rows = rank_and_output_nexus(nexus_results, args.out, top_n=args.top)
    stage4_time = time.time() - stage4_start
    
    print(f"   [OK] Stage 4 complete in {stage4_time:.1f}s")
    
    # ── Pipeline Summary ──────────────────────────────────────────
    total_time = time.time() - start_time
    
    print(f"\n{'='*68}")
    print(f"  NEXUS PIPELINE SUMMARY")
    print(f"{'='*68}")
    print(f"  Total runtime:        {total_time:.1f}s")
    print(f"  Candidates in:        {total_count:,}")
    print(f"  Pre-filtered out:     {filtered_count:,}")
    print(f"  Honeypots caught:     {honeypot_count}")
    print(f"  NEXUS evaluated:      {len(nexus_results):,}")
    print(f"  Debates triggered:    {debate_count}")
    print(f"  Output:               {args.out}")
    print(f"  Top candidate score:  {rows[0]['score'] if rows else 'N/A'}")
    print(f"  Min candidate score:  {rows[-1]['score'] if rows else 'N/A'}")
    print(f"{'='*68}\n")
    
    if total_time > 300:
        print(f"  WARNING: Runtime ({total_time:.1f}s) exceeds 5-minute limit!")
    else:
        print(f"  [OK] Runtime ({total_time:.1f}s) within 5-minute constraint")


if __name__ == "__main__":
    main()
