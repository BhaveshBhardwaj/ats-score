import os
import json
import csv
import sys
import time
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    print("Please install the groq package: pip install groq")
    sys.exit(1)

# JD text as extracted from triage.py
JD_TEXT = """
Senior AI Engineer at a Series A AI-native talent intelligence platform. 
Must have production experience with embeddings-based retrieval systems like 
sentence-transformers, BGE, E5 deployed to real users. Must have operational 
experience with vector databases: Pinecone, Weaviate, Qdrant, Milvus, 
OpenSearch, Elasticsearch, FAISS. Strong Python required. Must have designed 
evaluation frameworks for ranking systems using NDCG, MRR, MAP. 
Nice to have: LLM fine-tuning with LoRA, QLoRA, PEFT. Learning-to-rank models. 
The ideal candidate has 6-8 years experience, 4-5 years in applied ML/AI roles 
at product companies, not services. Has shipped end-to-end ranking, search, or 
recommendation system to real users at meaningful scale. Located in India. 
Disqualifiers: pure research without production deployment, only recent LangChain 
experience, career entirely at consulting firms like TCS Infosys Wipro, 
primarily computer vision or robotics without NLP/IR exposure.
"""

def get_llm_score(client, candidate_json):
    prompt = f"""You are an expert technical recruiter and AI Engineering manager.
Evaluate the following candidate profile against the Job Description.
Provide a strict integer score from 1 to 10 representing their fit.
1 = Terrible fit, reject immediately.
10 = Perfect fit, hire immediately.

Output ONLY a single integer from 1 to 10. Do not include any explanation.

--- JOB DESCRIPTION ---
{JD_TEXT}

--- CANDIDATE PROFILE ---
{json.dumps(candidate_json, indent=2)}
"""

    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict technical evaluator. Output only a single integer."},
                {"role": "user", "content": prompt}
            ],
            model="openai/gpt-oss-120b",
        )
        result = completion.choices[0].message.content.strip()
        score_str = "".join(filter(str.isdigit, result))
        if score_str:
            return int(score_str)
        return 0
    except Exception as e:
        print(f"Error querying Groq: {e}")
        return 0

def main():
    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable.")
        sys.exit(1)
        
    client = Groq(api_key=api_key)

    base_path = Path(__file__).parent.parent
    cand_json = base_path / "data" / "eval_candidates.json"
    out_csv = base_path / "evaluation" / "llm_submission.csv"

    if not cand_json.exists():
        print(f"File not found: {cand_json}")
        return

    print("--- FULL DATASET LLM SCORING (openai/gpt-oss-120b) ---")
    
    with open(cand_json, "r", encoding="utf-8") as f:
        candidates_list = json.load(f)

    print(f"Evaluating all {len(candidates_list)} candidates... (This will take a few minutes)")
    
    results = []
    for i, c in enumerate(candidates_list):
        cid = c.get("candidate_id", "UNKNOWN")
        
        print(f"  [{i+1}/{len(candidates_list)}] Scoring {cid}... ", end="", flush=True)
        llm_score = get_llm_score(client, c)
        print(f"Score: {llm_score}/10")
        
        results.append({
            "candidate_id": cid,
            "llm_score": llm_score
        })
        time.sleep(0.5)

    # Rank them by score descending
    results.sort(key=lambda x: -x["llm_score"])
    
    # Assign ranks
    ranked_results = []
    for rank, row in enumerate(results, start=1):
        ranked_results.append({
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "llm_score": row["llm_score"]
        })

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "llm_score"])
        writer.writeheader()
        writer.writerows(ranked_results)
        
    print(f"\nDone! LLM rankings written to {out_csv}")

if __name__ == "__main__":
    main()
