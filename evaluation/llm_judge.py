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
            model="openai/gpt-oss-120b", # Using the highly capable model for evaluation
        )
        result = completion.choices[0].message.content.strip()
        print(f"(Raw: '{result}') ", end="")
        # Ensure we just get the integer
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
    sub_csv = base_path / "evaluation" / "eval_submission.csv"
    cand_json = base_path / "data" / "eval_candidates.json"

    if not sub_csv.exists() or not cand_json.exists():
        sub_csv = base_path / "outputs" / "submission.csv"
        cand_json = base_path / "data" / "sample_candidates.json"

    print("--- NEXUS LLM-AS-A-JUDGE EVALUATION (Using Groq llama3-70b) ---")
    
    # Load candidates mapping
    with open(cand_json, "r", encoding="utf-8") as f:
        candidates_list = json.load(f)
    candidates = {c["candidate_id"]: c for c in candidates_list}

    # Load submission rankings
    scored_candidates = []
    with open(sub_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored_candidates.append(row)

    if not scored_candidates:
        print("No candidates found.")
        return

    # Sample candidates (Top 5, Middle 5, Bottom 5)
    top_5 = scored_candidates[:5]
    mid_idx = len(scored_candidates) // 2
    middle_5 = scored_candidates[max(0, mid_idx-2):min(len(scored_candidates), mid_idx+3)]
    bottom_5 = scored_candidates[-5:]

    eval_set = [("TOP", c) for c in top_5] + [("MIDDLE", c) for c in middle_5] + [("BOTTOM", c) for c in bottom_5]

    print(f"Evaluating {len(eval_set)} sampled candidates...")
    
    results = []
    for category, row in eval_set:
        cid = row["candidate_id"]
        nexus_rank = row["rank"]
        nexus_score = float(row["score"])
        
        c_data = candidates.get(cid)
        if not c_data:
            continue
            
        print(f"  Scoring {cid} (NEXUS Rank: {nexus_rank})... ", end="", flush=True)
        llm_score = get_llm_score(client, c_data)
        print(f"LLM Score: {llm_score}/10")
        
        results.append({
            "category": category,
            "cid": cid,
            "nexus_rank": int(nexus_rank),
            "nexus_score": nexus_score,
            "llm_score": llm_score
        })
        time.sleep(0.5) # Slight pause to avoid rate limits

    # Print summary correlation
    print("\n--- EVALUATION SUMMARY ---")
    
    def avg_llm(cat):
        scores = [r["llm_score"] for r in results if r["category"] == cat]
        return sum(scores) / len(scores) if scores else 0

    print(f"Average LLM Score for NEXUS TOP candidates:    {avg_llm('TOP'):.1f} / 10")
    print(f"Average LLM Score for NEXUS MIDDLE candidates: {avg_llm('MIDDLE'):.1f} / 10")
    print(f"Average LLM Score for NEXUS BOTTOM candidates: {avg_llm('BOTTOM'):.1f} / 10")
    
    # Basic correlation check
    if avg_llm('TOP') > avg_llm('BOTTOM'):
        print("\nSUCCESS: LLM Judge strongly agrees with NEXUS rankings!")
    else:
        print("\nWARNING: LLM Judge disagrees with NEXUS rankings.")

if __name__ == "__main__":
    main()
