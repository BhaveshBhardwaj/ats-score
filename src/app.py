"""
Redrob AI Candidate Ranker — NEXUS Demo App

Interactive Streamlit app showcasing the multi-agent adversarial
evaluation system. Upload candidates and see how 5 agents debate
their fitness for the Senior AI Engineer role.

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

from agents.advocate import AdvocateAgent
from agents.skeptic import SkepticAgent
from agents.forensic import ForensicAgent
from agents.trajectory import TrajectoryAgent
from agents.availability import AvailabilityAgent
from fusion.debate import run_debate, compute_disagreement_profile, aggregate_evidence
from fusion.bayesian import bayesian_fusion, compute_lcb_score
from fusion.ranker import generate_nexus_reasoning


# ---- Page Config ----
st.set_page_config(
    page_title="NEXUS — Multi-Agent Candidate Ranker",
    page_icon="⚡",
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
    .agent-card {
        border-radius: 12px;
        padding: 15px;
        margin: 5px 0;
    }
    .agent-advocate { background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%); color: #1a3a1a; }
    .agent-skeptic { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #3a1a1a; }
    .agent-forensic { background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%); color: #1a1a3a; }
    .agent-trajectory { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #3a2a1a; }
    .agent-availability { background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%); color: #2a1a3a; }
    .score-high { color: #10b981; font-weight: bold; }
    .score-mid { color: #f59e0b; font-weight: bold; }
    .score-low { color: #ef4444; font-weight: bold; }
    .evidence-positive { color: #10b981; }
    .evidence-negative { color: #ef4444; }
    .evidence-challenged { text-decoration: line-through; opacity: 0.6; }
    .stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


def _init_agents():
    """Initialize all NEXUS agents (cached)."""
    return [
        AdvocateAgent(),
        SkepticAgent(),
        ForensicAgent(),
        TrajectoryAgent(),
        AvailabilityAgent(),
    ]


def run_nexus_single(candidate, agents):
    """Run NEXUS on a single candidate."""
    verdicts = [agent.evaluate(candidate) for agent in agents]
    disagreement = compute_disagreement_profile(verdicts)
    
    if disagreement["needs_debate"]:
        verdicts = run_debate(verdicts, agents, candidate)
    
    all_evidence = aggregate_evidence(verdicts)
    fusion = bayesian_fusion(verdicts, all_evidence)
    lcb = compute_lcb_score(fusion)
    
    return {
        "candidate_id": candidate.get("candidate_id", ""),
        "verdicts": verdicts,
        "fusion_result": fusion,
        "lcb_score": lcb,
        "disagreement": disagreement,
    }


def run_nexus_pipeline(candidates_data, agents):
    """Run NEXUS on all candidates."""
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
        progress.progress((i + 1) / total, text=f"NEXUS evaluating {i+1}/{total}...")
        
        passes, reason = prefilter(candidate)
        if not passes:
            results["filtered"] += 1
            if "Honeypot" in reason:
                results["honeypots"] += 1
            continue
        
        nexus_result = run_nexus_single(candidate, agents)
        nexus_result["_candidate"] = candidate
        results["scored"].append(nexus_result)
    
    results["scored"].sort(key=lambda x: (-x["lcb_score"], x["candidate_id"]))
    progress.empty()
    return results


def display_agent_verdict(verdict, agent_name, color_class):
    """Display a single agent's verdict."""
    score_pct = f"{verdict.score:.0%}"
    conf_pct = f"{verdict.confidence:.0%}"
    
    st.markdown(f"""
    <div class="agent-card {color_class}">
        <strong>{agent_name}</strong>: {score_pct} (conf: {conf_pct})<br>
        <small>{verdict.reasoning[:120]}</small>
    </div>
    """, unsafe_allow_html=True)


def display_results(results, top_n=100):
    """Display NEXUS ranked results."""
    scored = results["scored"][:top_n]
    
    if not scored:
        st.warning("No candidates passed the pre-filter.")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Processed", results["total"])
    with col2:
        st.metric("NEXUS Evaluated", len(results["scored"]))
    with col3:
        st.metric("Filtered Out", results["filtered"])
    with col4:
        st.metric("Honeypots", results["honeypots"])
    with col5:
        debates = sum(1 for s in results["scored"] if s["disagreement"].get("needs_debate", False))
        st.metric("Debates", debates)
    
    st.markdown("---")
    
    # Results table
    rows = []
    for rank, s in enumerate(scored, 1):
        fusion = s["fusion_result"]
        reasoning = generate_nexus_reasoning(fusion, s.get("verdicts", []))
        candidate = s.get("_candidate", {})
        profile = candidate.get("profile", {})
        
        ci_low, ci_high = fusion.get("confidence_interval", (0, 1))
        
        rows.append({
            "Rank": rank,
            "ID": s["candidate_id"],
            "LCB Score": round(s["lcb_score"], 4),
            "Posterior": round(fusion["posterior"], 4),
            "Uncertainty": round(fusion["uncertainty"], 4),
            "Agreement": round(fusion["agent_agreement"], 4),
            "Title": profile.get("current_title", "?"),
            "Company": profile.get("current_company", "?"),
            "Exp": f"{profile.get('years_of_experience', 0)} yrs",
            "Reasoning": reasoning,
        })
    
    st.dataframe(
        rows,
        use_container_width=True,
        height=500,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "LCB Score": st.column_config.ProgressColumn("LCB", min_value=0, max_value=1, format="%.4f"),
            "Posterior": st.column_config.ProgressColumn("Posterior", min_value=0, max_value=1, format="%.4f"),
            "Uncertainty": st.column_config.ProgressColumn("Uncert", min_value=0, max_value=1, format="%.4f"),
            "Agreement": st.column_config.ProgressColumn("Agree", min_value=0, max_value=1, format="%.4f"),
        },
    )
    
    # CSV download
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "candidate_id": row["ID"],
            "rank": row["Rank"],
            "score": f"{row['LCB Score']:.4f}",
            "reasoning": row["Reasoning"],
        })
    
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
    )
    
    # ── Detailed Agent Analysis for Top Candidates ────────────────
    st.markdown("---")
    st.subheader("🔍 Multi-Agent Analysis (Top 10)")
    
    agent_names = {
        "advocate": ("🟢 Advocate", "agent-advocate"),
        "skeptic": ("🔴 Skeptic", "agent-skeptic"),
        "forensic": ("🔵 Forensic", "agent-forensic"),
        "trajectory": ("🟠 Trajectory", "agent-trajectory"),
        "availability": ("🟣 Availability", "agent-availability"),
    }
    
    for s in scored[:10]:
        candidate = s.get("_candidate", {})
        profile = candidate.get("profile", {})
        fusion = s["fusion_result"]
        
        ci_low, ci_high = fusion.get("confidence_interval", (0, 1))
        
        with st.expander(
            f"#{scored.index(s)+1} | {s['candidate_id']} | "
            f"{profile.get('current_title', '?')} @ {profile.get('current_company', '?')} | "
            f"LCB: {s['lcb_score']:.4f}"
        ):
            # Agent verdicts
            st.markdown("**Agent Verdicts:**")
            cols = st.columns(5)
            for i, verdict in enumerate(s.get("verdicts", [])):
                name, css_class = agent_names.get(verdict.agent_id, ("Agent", ""))
                with cols[i % 5]:
                    display_agent_verdict(verdict, name, css_class)
            
            # Fusion details
            st.markdown("**Fusion Results:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Posterior", f"{fusion['posterior']:.4f}")
            with col2:
                st.metric("Uncertainty", f"{fusion['uncertainty']:.4f}")
            with col3:
                st.metric("CI", f"[{ci_low:.2f}, {ci_high:.2f}]")
            with col4:
                st.metric("Agreement", f"{fusion['agent_agreement']:.2f}")
            
            # Evidence summary
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Top Positive Evidence:**")
                for ev in fusion.get("top_positive_evidence", []):
                    survived = " ✓" if ev.challenge_survived else ""
                    st.markdown(f"- {ev.details}{survived}")
            
            with col2:
                st.markdown("**❌ Top Negative Evidence:**")
                for ev in fusion.get("top_negative_evidence", []):
                    survived = " ✓" if ev.challenge_survived else ""
                    st.markdown(f"- {ev.details}{survived}")
            
            # Disagreement profile
            disagree = s.get("disagreement", {})
            if disagree.get("needs_debate"):
                st.info(f"⚡ Debate triggered (spread: {disagree.get('spread', 0):.2f})")


# ---- Main App ----
def main():
    st.markdown('<p class="main-header">⚡ NEXUS Candidate Ranker</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Neural Evidence eXamination & Unified Scoring — Multi-Agent Adversarial Reasoning</p>', unsafe_allow_html=True)
    
    agents = _init_agents()
    
    with st.sidebar:
        st.header("NEXUS Architecture")
        st.markdown("""
        **5 Adversarial Agents** evaluate each candidate independently, 
        then **debate** when they disagree.
        
        🟢 **Advocate** — finds reasons TO hire  
        🔴 **Skeptic** — finds reasons NOT to hire  
        🔵 **Forensic** — cross-validates all claims  
        🟠 **Trajectory** — Monte Carlo career simulation  
        🟣 **Availability** — Bayesian hireability estimation  
        
        ---
        
        **Novel Algorithms:**
        - Cross-Evidence Corroboration Matrix (CECM)
        - Adversarial Evidence Challenge Protocol
        - Lower Confidence Bound Ranking
        - Bayesian Belief Fusion
        """)
        
        top_n = st.slider("Top N candidates", 10, 100, 100)
    
    st.markdown("### Upload Candidates")
    
    uploaded_file = st.file_uploader(
        "Choose a candidates file",
        type=["json", "jsonl"],
        help="JSON array or line-delimited JSON (JSONL) format"
    )
    
    if uploaded_file:
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
            
            if st.button("⚡ Run NEXUS Pipeline", type="primary"):
                start = time.time()
                
                with st.spinner("Running multi-agent adversarial evaluation..."):
                    results = run_nexus_pipeline(candidates_data, agents)
                
                elapsed = time.time() - start
                st.success(f"NEXUS pipeline completed in {elapsed:.1f}s")
                
                display_results(results, top_n=top_n)
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload a candidates JSON/JSONL file to run the NEXUS pipeline.")


if __name__ == "__main__":
    main()
