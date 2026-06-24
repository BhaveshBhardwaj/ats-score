"""
Fast Triage — Stage 0.5 (NEXUS v2 AI Upgrade)

Uses FlashText (Aho-Corasick algorithm) for hyper-fast keyword extraction
and FastEmbed (ONNX) for local CPU semantic similarity scoring.
"""

from flashtext import KeywordProcessor
import numpy as np
import time

from config import (
    MUST_HAVE_SKILLS, CORE_AI_ML_SKILLS, NLP_IR_SKILLS,
    LLM_FINETUNING_SKILLS, MLOPS_PRODUCTION_SKILLS,
    HIGHLY_RELEVANT_TITLES, MODERATELY_RELEVANT_TITLES,
    IRRELEVANT_TITLES, NEGATIVE_DOMAIN_SKILLS,
    CONSULTING_SERVICES_COMPANIES,
    TECH_PRODUCT_INDUSTRIES,
    IDEAL_EXPERIENCE_MIN, IDEAL_EXPERIENCE_MAX,
    ACCEPTABLE_EXPERIENCE_MIN, ACCEPTABLE_EXPERIENCE_MAX,
)

# ── 1. Initialize FlashText Processors (O(N) extraction) ──────────
_skill_processor = KeywordProcessor(case_sensitive=False)
for kw in MUST_HAVE_SKILLS:
    _skill_processor.add_keyword(kw, "MUST_HAVE")
for kw in CORE_AI_ML_SKILLS:
    _skill_processor.add_keyword(kw, "CORE_AI")
for kw in NLP_IR_SKILLS:
    _skill_processor.add_keyword(kw, "CORE_AI")
for kw in NEGATIVE_DOMAIN_SKILLS:
    _skill_processor.add_keyword(kw, "NEGATIVE")

_title_processor = KeywordProcessor(case_sensitive=False)
for t in HIGHLY_RELEVANT_TITLES:
    _title_processor.add_keyword(t, "HIGH")
for t in MODERATELY_RELEVANT_TITLES:
    _title_processor.add_keyword(t, "MODERATE")
for t in IRRELEVANT_TITLES:
    _title_processor.add_keyword(t, "IRRELEVANT")

_ml_processor = KeywordProcessor(case_sensitive=False)
for kw in ("machine learning", "deep learning", "embedding", "ranking", "retrieval", "recommendation", "nlp", "neural", "transformer", "model"):
    _ml_processor.add_keyword(kw, "ML")
    
_prod_processor = KeywordProcessor(case_sensitive=False)
for kw in ("production", "deployed", "shipped", "scaled", "real users", "a/b test"):
    _prod_processor.add_keyword(kw, "PROD")

# ── 2. Initialize FastEmbed for Semantic Search ───────────────────
try:
    from fastembed import TextEmbedding
    _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    _JD_TEXT = (
        "Senior AI Engineer at a Series A AI-native talent intelligence platform. "
        "Must have production experience with embeddings-based retrieval systems like "
        "sentence-transformers, BGE, E5 deployed to real users. Must have operational "
        "experience with vector databases: Pinecone, Weaviate, Qdrant, Milvus, "
        "OpenSearch, Elasticsearch, FAISS. Strong Python required. Must have designed "
        "evaluation frameworks for ranking systems using NDCG, MRR, MAP. "
        "Nice to have: LLM fine-tuning with LoRA, QLoRA, PEFT. Learning-to-rank models. "
        "The ideal candidate has 6-8 years experience, 4-5 years in applied ML/AI roles "
        "at product companies, not services. Has shipped end-to-end ranking, search, or "
        "recommendation system to real users at meaningful scale. Located in India. "
        "Disqualifiers: pure research without production deployment, only recent LangChain "
        "experience, career entirely at consulting firms like TCS Infosys Wipro, "
        "primarily computer vision or robotics without NLP/IR exposure."
    )
    _jd_embedding = list(_embed_model.embed([_JD_TEXT]))[0]
    print("[INIT] FastEmbed Semantic Engine ready.")
except Exception as e:
    print(f"[INIT Warning] FastEmbed not initialized. Semantic scoring disabled. ({e})")
    _embed_model = None
    _jd_embedding = None

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def fast_triage_score(candidate: dict) -> tuple[float, bool]:
    """
    Compute a fast triage score (0-1) using FlashText.
    Returns (score, semantic_eligible)
    """
    score = 0.0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    # ── 1. Title relevance (FlashText) ───────────────────────────
    current_title = profile.get("current_title", "")
    title_matches = _title_processor.extract_keywords(current_title)
    
    if "HIGH" in title_matches:
        score += 0.25
    elif "MODERATE" in title_matches:
        score += 0.12
    else:
        ai_words = {"ai", "ml", "machine", "learning", "data", "nlp", "search", "ranking"}
        if set(current_title.lower().split()) & ai_words:
            score += 0.10
        if "IRRELEVANT" in title_matches:
            score -= 0.15
    
    # ── 2. Skill relevance (FlashText) ───────────────────────────
    skill_names = " ".join([s.get("name", "") for s in skills])
    skill_matches = _skill_processor.extract_keywords(skill_names)
    
    must_have_count = skill_matches.count("MUST_HAVE")
    core_count = skill_matches.count("CORE_AI")
    negative_count = skill_matches.count("NEGATIVE")
    
    score += min(0.20, must_have_count * 0.05)
    score += min(0.10, core_count * 0.02)
    score -= min(0.15, negative_count * 0.03)
    
    # ── 3. Experience fit ─────────────────────────────────────────
    years = profile.get("years_of_experience", 0)
    if IDEAL_EXPERIENCE_MIN <= years <= IDEAL_EXPERIENCE_MAX:
        score += 0.10
    elif ACCEPTABLE_EXPERIENCE_MIN <= years <= ACCEPTABLE_EXPERIENCE_MAX:
        score += 0.05
    
    # ── 4. Career ML evidence (FlashText scan) ────────────────────
    ml_hits = 0
    production_hits = 0
    
    if career:
        # Concatenate recent job descriptions for one-pass scanning
        recent_desc = " ".join([job.get("description", "") for job in career[:2]])
        
        ml_hits = len(_ml_processor.extract_keywords(recent_desc))
        production_hits = len(_prod_processor.extract_keywords(recent_desc))
        
        # Check recent titles
        recent_titles = " ".join([job.get("title", "") for job in career[:2]])
        if "HIGH" in _title_processor.extract_keywords(recent_titles):
            score += 0.05
    
    score += min(0.15, ml_hits * 0.02)
    score += min(0.10, production_hits * 0.03)
    
    # ── 5. Consulting/Product Flags ───────────────────────────────
    if len(career) >= 2:
        consulting_count = sum(
            1 for j in career
            if any(c in j.get("company", "").lower() for c in CONSULTING_SERVICES_COMPANIES)
        )
        if consulting_count == len(career):
            score -= 0.20
            
    if career:
        product_count = sum(
            1 for j in career[:3]
            if any(ind in j.get("industry", "").lower() for ind in TECH_PRODUCT_INDUSTRIES)
        )
        score += min(0.10, product_count * 0.04)
    
    # ── 6. Product AI Depth — the JD's ideal archetype ─────────────
    # "6-8 years, 4-5 in applied ML/AI at product companies, shipped
    # ranking/search/recommendation to real users at meaningful scale"
    if career:
        product_ai_roles = 0
        for job in career:
            job_title = job.get("title", "").lower()
            job_industry = job.get("industry", "").lower()
            job_desc = job.get("description", "").lower()
            
            has_ai_title = "HIGH" in _title_processor.extract_keywords(job_title)
            has_product_co = any(ind in job_industry for ind in TECH_PRODUCT_INDUSTRIES)
            has_prod_work = len(_prod_processor.extract_keywords(job_desc)) > 0
            
            if has_ai_title and has_product_co:
                product_ai_roles += 1
                if has_prod_work:
                    product_ai_roles += 1  # Double credit for shipping
        
        score += min(0.15, product_ai_roles * 0.05)
        
    return max(0.0, min(1.0, score)), (score > 0.15)


def triage_candidates(candidates: list, max_candidates: int = 15000) -> list:
    """
    Score and filter candidates using FlashText + Semantic FastEmbed.
    """
    scored = []
    semantic_eligible = []
    
    # Step 1: FlashText scoring (Insanely fast for 100K)
    for candidate in candidates:
        base_score, is_eligible = fast_triage_score(candidate)
        scored.append((base_score, candidate))
        if is_eligible:
            semantic_eligible.append(candidate)
            
    # Sort by base score
    scored.sort(key=lambda x: -x[0])
    
    # Step 2: Semantic Score boost (Only for top candidates to save time)
    # We only semantically score up to 2,500 candidates to stay strictly under the 5-min CPU limit
    SEMANTIC_LIMIT = 2500
    top_candidates = [c for _, c in scored[:SEMANTIC_LIMIT]]
    remaining_candidates = [c for _, c in scored[SEMANTIC_LIMIT:max_candidates]]
    
    if _embed_model and _jd_embedding is not None and top_candidates:
        print(f"   [Stage 0.5] Generating embeddings for top {len(top_candidates):,} candidates...")
        # Create summary strings for the model
        summaries = []
        for c in top_candidates:
            title = c.get("profile", {}).get("current_title", "")
            skills = ", ".join([s.get("name", "") for s in c.get("skills", [])][:10])
            desc = c.get("career_history", [{}])[0].get("description", "")[:200] if c.get("career_history") else ""
            summaries.append(f"{title}. Skills: {skills}. Background: {desc}")
        
        # Batch embed
        embeddings = list(_embed_model.embed(summaries))
        
        # Add semantic similarity bonus
        final_scored = []
        for i, c in enumerate(top_candidates):
            base = scored[i][0]
            sim = cosine_similarity(_jd_embedding, embeddings[i])
            # Boost score by up to 0.3 based on semantic match
            bonus = max(0, sim * 0.3)
            final_scored.append((base + bonus, c))
            
        final_scored.sort(key=lambda x: -x[0])
        embedded_results = [c for _, c in final_scored]
        return embedded_results + remaining_candidates
    
    # If no embed model, just return top N from FlashText
    return [c for _, c in scored[:max_candidates]]
