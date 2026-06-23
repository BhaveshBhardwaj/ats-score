#!/usr/bin/env python3
"""
Redrob AI Candidate Ranker -- Main Entry Point

Intelligent candidate discovery and ranking system for the
Redrob Hackathon: Senior AI Engineer -- Founding Team position.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --limit 1000
"""

import argparse
import sys
import time
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.loader import load_candidates
from pipeline.prefilter import prefilter
from pipeline.scorer import score_candidate
from pipeline.ranker import rank_and_output


def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranker -- Intelligent Discovery & Ranking"
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
    print("=" * 62)
    print("  Redrob AI Candidate Ranker")
    print("  Senior AI Engineer -- Founding Team")
    print("")
    print("  Multi-stage pipeline: PreFilter -> Score -> Rank")
    print("=" * 62)
    print("")
    
    start_time = time.time()
    
    # --- Stage 1: Stream, pre-filter, and score ---
    print("[Stage 1] Loading & pre-filtering candidates...")
    print(f"   Source: {candidates_path}")
    if args.limit:
        print(f"   Limit: {args.limit} candidates")
    
    total_count = 0
    filtered_count = 0
    honeypot_count = 0
    scored_candidates = []
    
    stage1_start = time.time()
    
    for candidate in load_candidates(str(candidates_path), limit=args.limit):
        total_count += 1
        
        # Progress reporting
        if total_count % 10000 == 0:
            elapsed = time.time() - stage1_start
            rate = total_count / elapsed
            print(f"   Processed {total_count:,} candidates ({rate:.0f}/sec) -- {len(scored_candidates)} passed pre-filter")
        
        # Pre-filter
        passes, reason = prefilter(candidate)
        
        if not passes:
            filtered_count += 1
            if "Honeypot" in reason:
                honeypot_count += 1
            if args.verbose and total_count <= 20:
                cid = candidate.get("candidate_id", "?")
                print(f"   [FILTERED] {cid}: {reason}")
            continue
        
        # Score the candidate
        scored = score_candidate(candidate)
        scored_candidates.append(scored)
    
    stage1_time = time.time() - stage1_start
    
    print(f"\n   [OK] Stage 1 complete in {stage1_time:.1f}s")
    print(f"   Total candidates processed: {total_count:,}")
    print(f"   Filtered out: {filtered_count:,} ({filtered_count/max(total_count,1)*100:.1f}%)")
    print(f"   Honeypots detected: {honeypot_count}")
    print(f"   Candidates to rank: {len(scored_candidates):,}")
    
    # --- Stage 2: Sort and select top N ---
    print(f"\n[Stage 2] Ranking top {args.top} candidates...")
    
    stage2_start = time.time()
    rows = rank_and_output(scored_candidates, args.out, top_n=args.top)
    stage2_time = time.time() - stage2_start
    
    print(f"   [OK] Stage 2 complete in {stage2_time:.1f}s")
    
    # --- Summary ---
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total runtime:       {total_time:.1f}s")
    print(f"  Candidates in:       {total_count:,}")
    print(f"  Pre-filtered out:    {filtered_count:,}")
    print(f"  Honeypots caught:    {honeypot_count}")
    print(f"  Candidates scored:   {len(scored_candidates):,}")
    print(f"  Output:              {args.out}")
    print(f"  Top candidate score: {rows[0]['score'] if rows else 'N/A'}")
    print(f"  Min candidate score: {rows[-1]['score'] if rows else 'N/A'}")
    print(f"{'='*60}\n")
    
    # Check runtime constraint
    if total_time > 300:
        print(f"  WARNING: Runtime ({total_time:.1f}s) exceeds 5-minute limit!")
    else:
        print(f"  [OK] Runtime ({total_time:.1f}s) within 5-minute constraint")


if __name__ == "__main__":
    main()
