import json
import uuid
import sys
from pathlib import Path
import subprocess
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def create_golden_candidate(index):
    return {
        "candidate_id": f"GOLDEN_000{index}",
        "profile": {
            "current_title": "Senior AI Engineer",
            "years_of_experience": 7,
            "location": "India",
            "summary": "AI Engineer specializing in retrieval and ranking systems."
        },
        "skills": [
            {"name": "Python", "years": 7},
            {"name": "Pinecone", "years": 4},
            {"name": "FAISS", "years": 3},
            {"name": "sentence-transformers", "years": 4},
            {"name": "Elasticsearch", "years": 5},
            {"name": "Learning to Rank", "years": 3},
            {"name": "PyTorch", "years": 5}
        ],
        "career_history": [
            {
                "title": "Machine Learning Engineer",
                "company": "FastGrowing AI Startup",
                "industry": "Internet Software & Services",
                "start_date": "2022-01-01",
                "description": "Deployed embeddings-based retrieval systems (BGE, E5) to real users in production. Scaled FAISS vector database to millions of vectors. Built evaluation frameworks using NDCG."
            },
            {
                "title": "Data Scientist",
                "company": "Another Tech Product",
                "industry": "Information Technology",
                "start_date": "2018-01-01",
                "end_date": "2021-12-31",
                "description": "Developed ranking algorithms and ML models."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.95,
            "last_active_date": "2026-06-25",
            "notice_period_days": 15
        }
    }

def create_poison_candidate(index):
    return {
        "candidate_id": f"POISON_000{index}",
        "profile": {
            "current_title": "AI Intern Data Scientist Machine Learning Engineer AI NLP Search Ranking ML DL",
            "years_of_experience": 1,
            "location": "Bangalore",
            "country": "India",
            "summary": "machine learning AI ML artificial intelligence machine learning NLP ML search ranking machine learning deep learning"
        },
        "skills": [
            {"name": "Machine Learning", "years": 1},
            {"name": "AI", "years": 1},
            {"name": "Python", "years": 1}
        ],
        "career_history": [
            {
                "title": "Intern",
                "company": "Infosys",
                "industry": "IT Services",
                "start_date": "2023-01-01",
                "description": "machine learning deep learning nlp ranking search ai. Researched on a dataset in a lab."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag": False,
            "recruiter_response_rate": 0.05,
            "last_active_date": "2022-01-01",
            "notice_period_days": 90
        }
    }

def main():
    base_path = Path(__file__).parent.parent
    sample_path = base_path / "data" / "sample_candidates.json"
    out_path = base_path / "data" / "eval_candidates.json"
    submission_path = base_path / "evaluation" / "eval_submission.csv"

    print("--- NEXUS SYNTHETIC INJECTION EVALUATION ---")
    
    # 1. Load existing candidates
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except Exception as e:
        print(f"Error loading {sample_path}: {e}")
        return

    print(f"Loaded {len(candidates)} candidates from {sample_path}")

    # 2. Inject candidates
    for i in range(1, 6):
        candidates.append(create_golden_candidate(i))
        candidates.append(create_poison_candidate(i))

    print(f"Injected 5 Golden and 5 Poison candidates. Total: {len(candidates)}")

    # 3. Write to temporary file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f)

    # 4. Run NEXUS pipeline
    print("\nRunning NEXUS Pipeline...")
    rank_script = base_path / "src" / "rank.py"
    
    cmd = [
        sys.executable, str(rank_script),
        "--candidates", str(out_path),
        "--out", str(submission_path),
        "--top", str(len(candidates)) # Score all of them
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Pipeline failed!")
        print(result.stderr)
        return

    # 5. Evaluate results
    print("\n--- EVALUATION RESULTS ---")
    golden_ranks = []
    poison_ranks = []

    try:
        with open(submission_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["candidate_id"]
                rank = int(row["rank"])
                if cid.startswith("GOLDEN"):
                    golden_ranks.append((cid, rank, row["score"]))
                elif cid.startswith("POISON"):
                    poison_ranks.append((cid, rank, row["score"]))
    except Exception as e:
        print(f"Error reading submission.csv: {e}")
        return

    print(f"\nGOLDEN CANDIDATES (Should be in Top 10):")
    for cid, rank, score in golden_ranks:
        status = "PASSED" if rank <= 10 else "FAILED"
        print(f"  {cid}: Rank {rank} (Score: {score}) - {status}")

    total_scored = len(candidates)
    print(f"\nPOISON CANDIDATES (Should be filtered out or ranked very low):")
    if not poison_ranks:
        print("  All POISON candidates were pre-filtered or heavily down-ranked out of the output limit! (PASSED)")
    else:
        for cid, rank, score in poison_ranks:
            status = "PASSED" if rank > total_scored - 100 else "FAILED"
            print(f"  {cid}: Rank {rank} (Score: {score}) - {status}")

if __name__ == "__main__":
    main()
