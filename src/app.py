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
    /* Global Theme Overrides */
    .stApp {
        background-color: #0e1117;
        color: #e2e8f0;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        animation: fadeInDown 0.8s ease-out;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-top: -5px;
        margin-bottom: 40px;
        animation: fadeIn 1s ease-out 0.3s both;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Agent Cards (Glassmorphism + Hover) */
    .agent-card {
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .agent-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
    }
    
    /* Specific Agent Colors (Gradients) */
    .agent-advocate { border-left: 4px solid #34d399; }
    .agent-skeptic { border-left: 4px solid #f87171; }
    .agent-forensic { border-left: 4px solid #60a5fa; }
    .agent-trajectory { border-left: 4px solid #fbbf24; }
    .agent-availability { border-left: 4px solid #c084fc; }
    
    /* Agent Headers */
    .agent-header {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Evidence Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-pos { background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-neg { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    
    /* Dataframe overrides */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
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


def display_agent_verdict(verdict, agent_name, color_class, icon):
    """Display a single agent's verdict."""
    score_pct = f"{verdict.score:.0%}"
    conf_pct = f"{verdict.confidence:.0%}"
    
    st.markdown(f"""
    <div class="agent-card {color_class}">
        <div class="agent-header">
            {icon} {agent_name}
        </div>
        <div style="display: flex; gap: 15px; margin-bottom: 10px;">
            <div style="background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 6px; font-size: 0.9em;">
                Score: <strong>{score_pct}</strong>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 6px; font-size: 0.9em;">
                Confidence: <strong>{conf_pct}</strong>
            </div>
        </div>
        <div style="font-size: 0.95em; color: #cbd5e1; line-height: 1.5;">
            {verdict.reasoning[:150]}{"..." if len(verdict.reasoning) > 150 else ""}
        </div>
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
    
    # Use tabs for a cleaner layout
    tab1, tab2, tab3 = st.tabs(["🏆 Leaderboard", "🔍 Detailed Analysis", "📊 Analytics"])
    
    with tab1:
        st.markdown("### Top Ranked Candidates")
        st.dataframe(
            rows,
            use_container_width=True,
            height=600,
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
            type="primary"
        )
    
    with tab2:
        st.markdown("### Multi-Agent Debate Transcripts (Top 10)")
        
        agent_configs = {
            "advocate": ("Advocate", "agent-advocate", "🟢"),
            "skeptic": ("Skeptic", "agent-skeptic", "🔴"),
            "forensic": ("Forensic", "agent-forensic", "🔵"),
            "trajectory": ("Trajectory", "agent-trajectory", "🟠"),
            "availability": ("Availability", "agent-availability", "🟣"),
        }
        
        for s in scored[:10]:
            candidate = s.get("_candidate", {})
            profile = candidate.get("profile", {})
            fusion = s["fusion_result"]
            
            ci_low, ci_high = fusion.get("confidence_interval", (0, 1))
            
            with st.expander(
                f"#{scored.index(s)+1} | {s['candidate_id']} | "
                f"{profile.get('current_title', '?')} @ {profile.get('current_company', '?')} | "
                f"Score: {s['lcb_score']:.4f}"
            ):
                # Fusion metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Bayesian Posterior", f"{fusion['posterior']:.4f}")
                with col2:
                    st.metric("Uncertainty", f"{fusion['uncertainty']:.4f}")
                with col3:
                    st.metric("95% CI", f"[{ci_low:.2f}, {ci_high:.2f}]")
                with col4:
                    st.metric("Agent Agreement", f"{fusion['agent_agreement']:.2f}")
                
                st.markdown("<hr style='margin: 10px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                
                # Evidence row
                ev_col1, ev_col2 = st.columns(2)
                with ev_col1:
                    st.markdown("##### ✅ Positive Evidence")
                    for ev in fusion.get("top_positive_evidence", []):
                        survived = " (Debate Validated)" if ev.challenge_survived else ""
                        st.markdown(f"<span class='badge badge-pos'>+{ev.effective_weight:.2f}</span> {ev.details}{survived}", unsafe_allow_html=True)
                
                with ev_col2:
                    st.markdown("##### ❌ Negative Evidence")
                    for ev in fusion.get("top_negative_evidence", []):
                        survived = " (Debate Validated)" if ev.challenge_survived else ""
                        st.markdown(f"<span class='badge badge-neg'>{ev.effective_weight:.2f}</span> {ev.details}{survived}", unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 15px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                
                # Disagreement/Debate alert
                disagree = s.get("disagreement", {})
                if disagree.get("needs_debate"):
                    st.warning(f"⚡ **Debate Triggered**: Score spread was {disagree.get('spread', 0):.2f}. Agents cross-examined each other's evidence.")
                
                # Agent verdicts
                st.markdown("##### Agent Verdicts")
                cols = st.columns(5)
                for i, verdict in enumerate(s.get("verdicts", [])):
                    name, css_class, icon = agent_configs.get(verdict.agent_id, ("Agent", "", "🤖"))
                    with cols[i % 5]:
                        display_agent_verdict(verdict, name, css_class, icon)
                        
    with tab3:
        st.markdown("### Score Distribution & Pipeline Funnel")
        st.info("Visualizations can be built here using st.bar_chart or plotly depending on user preference.")
        # We can just show a basic bar chart of scores
        scores = [row["LCB Score"] for row in rows]
        if scores:
            st.bar_chart(scores)


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
