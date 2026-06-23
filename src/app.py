"""
Redrob AI Candidate Ranker -- Streamlit Demo App

Interactive demo for the Intelligent Candidate Discovery & Ranking system.
Upload a small set of candidates (JSON/JSONL) and see ranked results.

Run with:
    streamlit run app.py
"""

import streamlit as st
import json
import csv
import io
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.loader import load_candidates
from pipeline.prefilter import prefilter
from pipeline.scorer import score_candidate
from pipeline.ranker import generate_reasoning


# ---- Page Config ----
st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .score-high { color: #10b981; font-weight: bold; }
    .score-mid { color: #f59e0b; font-weight: bold; }
    .score-low { color: #ef4444; font-weight: bold; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


def score_color(score):
    if score >= 0.7:
        return "score-high"
    elif score >= 0.4:
        return "score-mid"
    return "score-low"


def run_pipeline(candidates_data):
    """Run the full ranking pipeline on a list of candidate dicts."""
    results = {
        "total": 0,
        "filtered": 0,
        "honeypots": 0,
        "scored": [],
    }
    
    progress = st.progress(0, text="Processing candidates...")
    total = len(candidates_data)
    
    for i, candidate in enumerate(candidates_data):
        results["total"] += 1
        progress.progress((i + 1) / total, text=f"Processing {i+1}/{total}...")
        
        # Pre-filter
        passes, reason = prefilter(candidate)
        if not passes:
            results["filtered"] += 1
            if "Honeypot" in reason:
                results["honeypots"] += 1
            continue
        
        # Score
        scored = score_candidate(candidate)
        scored["_candidate"] = candidate  # Keep reference for display
        results["scored"].append(scored)
    
    # Sort by composite score descending
    results["scored"].sort(key=lambda x: (-x["composite_score"], x["candidate_id"]))
    
    progress.empty()
    return results


def display_results(results, top_n=100):
    """Display ranked results."""
    scored = results["scored"][:top_n]
    
    if not scored:
        st.warning("No candidates passed the pre-filter. Try a different dataset.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Processed", results["total"])
    with col2:
        st.metric("Passed Pre-Filter", len(results["scored"]))
    with col3:
        st.metric("Filtered Out", results["filtered"])
    with col4:
        st.metric("Honeypots Caught", results["honeypots"])
    
    st.markdown("---")
    
    # Results table
    rows = []
    for rank, s in enumerate(scored, 1):
        candidate = s.get("_candidate", {})
        profile = candidate.get("profile", {})
        reasoning = generate_reasoning(s)
        
        rows.append({
            "Rank": rank,
            "ID": s["candidate_id"],
            "Score": round(s["composite_score"], 4),
            "Title": profile.get("current_title", "?"),
            "Experience": f"{profile.get('years_of_experience', 0)} yrs",
            "Company": profile.get("current_company", "?"),
            "Location": f"{profile.get('location', '?')}, {profile.get('country', '?')}",
            "Career": round(s["career"]["total_score"], 3),
            "Skills": round(s["skills"]["total_score"], 3),
            "Behavioral": round(s["behavioral"]["total_score"], 3),
            "Reasoning": reasoning,
        })
    
    st.dataframe(
        rows,
        use_container_width=True,
        height=600,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=1, format="%.4f"),
            "Career": st.column_config.ProgressColumn("Career", min_value=0, max_value=1, format="%.3f"),
            "Skills": st.column_config.ProgressColumn("Skills", min_value=0, max_value=1, format="%.3f"),
            "Behavioral": st.column_config.ProgressColumn("Behav", min_value=0, max_value=1, format="%.3f"),
        },
    )
    
    # Generate downloadable CSV
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "candidate_id": row["ID"],
            "rank": row["Rank"],
            "score": f"{row['Score']:.4f}",
            "reasoning": row["Reasoning"],
        })
    
    st.download_button(
        label="Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )
    
    # Detailed view for top candidates
    st.markdown("---")
    st.subheader("Detailed Candidate Profiles (Top 10)")
    
    for s in scored[:10]:
        candidate = s.get("_candidate", {})
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})
        
        with st.expander(
            f"#{scored.index(s)+1} | {s['candidate_id']} | "
            f"{profile.get('current_title', '?')} @ {profile.get('current_company', '?')} | "
            f"Score: {s['composite_score']:.4f}"
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Headline:** {profile.get('headline', 'N/A')}")
                st.markdown(f"**Summary:** {profile.get('summary', 'N/A')[:300]}...")
                st.markdown(f"**Location:** {profile.get('location', '?')}, {profile.get('country', '?')}")
                st.markdown(f"**Experience:** {profile.get('years_of_experience', 0)} years")
                st.markdown(f"**Industry:** {profile.get('current_industry', '?')}")
            
            with col2:
                st.markdown("**Score Breakdown:**")
                st.markdown(f"- Career Fit: {s['career']['total_score']:.3f} -- {s['career']['details']}")
                st.markdown(f"- Skills: {s['skills']['total_score']:.3f} -- {s['skills']['details']}")
                st.markdown(f"- Experience: {s['experience']['total_score']:.3f} -- {s['experience']['details']}")
                st.markdown(f"- Behavioral: {s['behavioral']['total_score']:.3f} -- {s['behavioral']['details']}")
                st.markdown(f"- Location: {s['location']['total_score']:.3f} -- {s['location']['details']}")
            
            # Career history
            st.markdown("**Career History:**")
            for job in candidate.get("career_history", []):
                current = " (Current)" if job.get("is_current") else ""
                st.markdown(
                    f"- **{job.get('title', '?')}** at {job.get('company', '?')}{current} "
                    f"({job.get('duration_months', 0)}mo, {job.get('industry', '?')})"
                )
            
            # Skills
            skills = candidate.get("skills", [])
            if skills:
                skill_text = ", ".join(
                    f"{s.get('name', '?')} ({s.get('proficiency', '?')})"
                    for s in skills
                )
                st.markdown(f"**Skills:** {skill_text}")


# ---- Main App ----
def main():
    st.markdown('<p class="main-header">Redrob AI Candidate Ranker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Candidate Discovery & Ranking for Senior AI Engineer</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.markdown("""
        This system ranks candidates for a **Senior AI Engineer -- Founding Team** 
        position at Redrob AI using a multi-stage pipeline:
        
        1. **Pre-Filter**: Eliminate obviously unfit candidates & honeypots
        2. **Multi-Dimensional Scoring**: Career, Skills, Experience, Behavioral, Location
        3. **Rank & Reason**: Composite scoring with natural language reasoning
        """)
        
        top_n = st.slider("Top N candidates to show", 10, 100, 100)
        
        st.markdown("---")
        st.markdown("**Scoring Weights:**")
        st.markdown("- Career Fit: 35%")
        st.markdown("- Skills Match: 25%")
        st.markdown("- Experience: 15%")
        st.markdown("- Behavioral: 15%")
        st.markdown("- Location: 10%")
    
    # File upload
    st.markdown("### Upload Candidates")
    st.markdown("Upload a JSON or JSONL file with candidate profiles. For a quick test, use the `sample_candidates.json` from the hackathon bundle.")
    
    uploaded_file = st.file_uploader(
        "Choose a candidates file",
        type=["json", "jsonl"],
        help="JSON array or line-delimited JSON (JSONL) format"
    )
    
    if uploaded_file:
        # Parse the file
        try:
            content = uploaded_file.read().decode("utf-8")
            
            if uploaded_file.name.endswith(".json"):
                candidates_data = json.loads(content)
                if not isinstance(candidates_data, list):
                    candidates_data = [candidates_data]
            else:
                candidates_data = []
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if line:
                        candidates_data.append(json.loads(line))
            
            st.success(f"Loaded {len(candidates_data)} candidates from {uploaded_file.name}")
            
            # Run pipeline
            if st.button("Run Ranking Pipeline", type="primary"):
                start = time.time()
                
                with st.spinner("Running multi-stage ranking pipeline..."):
                    results = run_pipeline(candidates_data)
                
                elapsed = time.time() - start
                st.success(f"Pipeline completed in {elapsed:.1f}s")
                
                display_results(results, top_n=top_n)
                
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    else:
        # Show sample data option
        st.info("No file uploaded. Upload a candidates JSON/JSONL file to get started.")


if __name__ == "__main__":
    main()
