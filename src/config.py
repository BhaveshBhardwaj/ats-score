"""
Configuration module encoding the JD requirements as structured data.

Senior AI Engineer — Founding Team at Redrob AI
All scoring weights, skill taxonomies, and disqualifier rules are derived
directly from the job description text.
"""

# ─────────────────────────────────────────────────────────────
# Skill Taxonomy — semantic groupings of skills by relevance
# ─────────────────────────────────────────────────────────────

# Skills that directly match the JD's "things you absolutely need"
MUST_HAVE_SKILLS = {
    # Embeddings-based retrieval
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embeddings", "embedding", "semantic search",
    "dense retrieval", "vector search", "text embeddings",
    # Vector databases & hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
    "annoy", "scann", "chroma", "chromadb",
    # Ranking & evaluation
    "ndcg", "mrr", "map", "ranking", "learning to rank",
    "information retrieval", "search ranking", "re-ranking",
    "reranking", "bm25", "evaluation framework",
    # Python (strong)
    "python",
}

# Skills in the core AI/ML domain relevant to the role
CORE_AI_ML_SKILLS = {
    "machine learning", "deep learning", "neural networks",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "random forest", "supervised learning", "unsupervised learning",
    "reinforcement learning", "ml", "ai", "artificial intelligence",
    "model training", "model evaluation", "feature engineering",
    "statistical modeling", "bayesian", "regression", "classification",
}

# NLP & Information Retrieval — the JD's primary domain
NLP_IR_SKILLS = {
    "nlp", "natural language processing", "text mining",
    "text classification", "named entity recognition", "ner",
    "sentiment analysis", "tokenization", "word embeddings",
    "word2vec", "glove", "fasttext", "bert", "transformers",
    "huggingface", "hugging face", "spacy", "nltk", "gensim",
    "topic modeling", "lda", "text generation", "question answering",
    "summarization", "language model", "language models",
    "sequence to sequence", "seq2seq", "attention mechanism",
    "transformer architecture",
}

# LLM & Fine-tuning — "things we'd like you to have"
LLM_FINETUNING_SKILLS = {
    "llm", "llms", "large language model", "large language models",
    "fine-tuning", "fine tuning", "finetuning",
    "lora", "qlora", "peft", "rlhf", "instruction tuning",
    "prompt engineering", "prompt tuning", "langchain", "llamaindex",
    "rag", "retrieval augmented generation", "gpt", "chatgpt",
    "claude", "gemini", "openai", "anthropic",
    "fine-tuning llms", "model alignment",
}

# MLOps & Production — signals of shipping to production
MLOPS_PRODUCTION_SKILLS = {
    "mlflow", "weights & biases", "wandb", "bentoml", "mlops",
    "model serving", "model deployment", "docker", "kubernetes",
    "k8s", "ci/cd", "fastapi", "flask", "django",
    "airflow", "luigi", "dagster", "prefect",
    "aws sagemaker", "vertex ai", "azure ml",
    "model monitoring", "a/b testing", "ab testing",
    "feature store", "experiment tracking",
}

# Data Engineering — adjacent and valuable
DATA_ENGINEERING_SKILLS = {
    "spark", "pyspark", "apache spark", "kafka", "apache kafka",
    "airflow", "apache airflow", "data pipelines", "etl",
    "data warehouse", "snowflake", "bigquery", "redshift",
    "dbt", "sql", "postgresql", "mysql", "mongodb",
    "data modeling", "data engineering", "databricks",
    "apache beam", "apache flink", "streaming",
}

# Cloud platforms
CLOUD_SKILLS = {
    "aws", "gcp", "azure", "google cloud", "amazon web services",
    "cloud computing", "ec2", "s3", "lambda",
}

# Skills that indicate WRONG domain (negative signal)
NEGATIVE_DOMAIN_SKILLS = {
    # Mechanical / Civil Engineering
    "solidworks", "creo", "ansys", "autocad", "catia",
    "cad", "fea", "cfd", "mechanical design", "structural analysis",
    "civil engineering", "construction",
    # Pure Accounting / Finance (non-tech)
    "tally", "accounting", "tax", "gst", "gaap", "ind-as",
    "audit", "statutory compliance", "financial reporting",
    "accounts payable", "accounts receivable",
    # Pure HR
    "recruitment", "talent acquisition", "payroll",
    "employee engagement", "performance management",
    "compensation and benefits", "hris",
    # Pure Marketing (non-tech)
    "seo", "sem", "social media marketing", "brand management",
    "content writing", "copywriting", "email marketing",
    "google ads", "facebook ads", "marketing automation",
    # Design (non-tech)
    "photoshop", "illustrator", "indesign", "figma",
    "sketch", "graphic design", "ui design", "ux design",
    "adobe suite", "canva",
    # Sales
    "sales", "crm", "salesforce", "lead generation",
    "cold calling", "business development",
}

# ─────────────────────────────────────────────────────────────
# Consulting / Services companies — disqualifier if entire career
# ─────────────────────────────────────────────────────────────

CONSULTING_SERVICES_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro",
    "accenture", "cognizant", "capgemini", "hcl", "hcl technologies",
    "tech mahindra", "l&t infotech", "lt infotech", "lti",
    "mindtree", "mphasis", "hexaware", "persistent systems",
    "cyient", "zensar", "niit technologies", "coforge",
    "birlasoft", "ltimindtree", "deloitte", "kpmg", "ey",
    "ernst & young", "pwc", "pricewaterhousecoopers",
    "ibm consulting", "ibm global services",
}

# ─────────────────────────────────────────────────────────────
# Title relevance — how relevant is a job title to AI Engineer role
# ─────────────────────────────────────────────────────────────

# Titles that strongly indicate AI/ML engineering background
HIGHLY_RELEVANT_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer", "lead ai engineer",
    "lead ml engineer", "staff ml engineer", "principal ml engineer",
    "nlp engineer", "natural language processing engineer",
    "deep learning engineer", "applied ml engineer",
    "applied scientist", "research engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "ml scientist", "research scientist",
    "search engineer", "ranking engineer", "recommendation engineer",
    "recommendation systems engineer", "recommendations engineer",
    "information retrieval engineer", "retrieval engineer",
    "applied ai engineer", "ml platform engineer",
    "machine learning platform engineer",
    "junior ai engineer", "junior machine learning engineer",
}

# Titles with moderate relevance (could be adjacent)
MODERATELY_RELEVANT_TITLES = {
    "data engineer", "senior data engineer", "backend engineer",
    "software engineer", "senior software engineer",
    "full stack engineer", "platform engineer",
    "analytics engineer", "data analyst",
    "junior ml engineer", "junior data scientist",
    "technical lead", "engineering manager",
}

# Titles that are essentially irrelevant to AI Engineer role
IRRELEVANT_TITLES = {
    "hr manager", "human resources manager", "accountant",
    "marketing manager", "sales executive", "sales manager",
    "operations manager", "project manager", "business analyst",
    "content writer", "graphic designer", "ui designer",
    "ux designer", "civil engineer", "mechanical engineer",
    "electrical engineer", "chemical engineer",
    "customer support", "customer service",
    "financial analyst", "investment analyst",
    "teacher", "professor", "lecturer",
    "admin", "administrator", "receptionist",
    "supply chain manager", "logistics manager",
}

# ─────────────────────────────────────────────────────────────
# Industries relevant to product/tech companies
# ─────────────────────────────────────────────────────────────

TECH_PRODUCT_INDUSTRIES = {
    "software", "technology", "information technology",
    "internet", "saas", "fintech", "edtech", "healthtech",
    "ai", "artificial intelligence", "machine learning", "ai/ml",
    "data analytics", "cloud computing", "cybersecurity",
    "e-commerce", "ecommerce", "gaming", "social media",
    "adtech", "martech", "food delivery", "transportation",
    "logistics technology", "proptech", "insurtech",
    "digital media", "media", "telecommunications",
    "biotech", "pharmaceutical", "automotive",
}

# IT Services is tricky — it's services, but also where lots of
# India-based engineers work. We don't penalize it, but we don't
# give it the product-company bonus either.
SERVICES_INDUSTRIES = {
    "it services", "consulting", "outsourcing", "bpo",
    "staffing", "recruitment",
}

# ─────────────────────────────────────────────────────────────
# Location preferences
# ─────────────────────────────────────────────────────────────

# JD says: Pune/Noida preferred, Hyderabad, Mumbai, Delhi NCR welcome
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "new delhi", "delhi ncr",
    "gurgaon", "gurugram", "ghaziabad", "faridabad",
}

ACCEPTABLE_LOCATIONS = {
    "hyderabad", "mumbai", "bangalore", "bengaluru",
    "chennai", "kolkata", "ahmedabad", "jaipur",
}

PREFERRED_COUNTRY = "india"

# ─────────────────────────────────────────────────────────────
# Experience bands
# ─────────────────────────────────────────────────────────────

IDEAL_EXPERIENCE_MIN = 5.0
IDEAL_EXPERIENCE_MAX = 9.0
ACCEPTABLE_EXPERIENCE_MIN = 3.0
ACCEPTABLE_EXPERIENCE_MAX = 15.0
HARD_MIN_EXPERIENCE = 1.0
HARD_MAX_EXPERIENCE = 20.0

# ─────────────────────────────────────────────────────────────
# Salary expectations (INR LPA) — Series A AI Engineer
# ─────────────────────────────────────────────────────────────

SALARY_IDEAL_MIN = 15.0   # 15 LPA
SALARY_IDEAL_MAX = 50.0   # 50 LPA
SALARY_HARD_MAX = 80.0    # Above this is likely enterprise / FAANG expectation

# ─────────────────────────────────────────────────────────────
# Behavioral signal thresholds
# ─────────────────────────────────────────────────────────────

RESPONSE_RATE_GOOD = 0.5
RESPONSE_RATE_BAD = 0.15
RESPONSE_TIME_GOOD_HOURS = 24
RESPONSE_TIME_BAD_HOURS = 168  # 1 week
NOTICE_PERIOD_IDEAL = 30
NOTICE_PERIOD_OK = 60
NOTICE_PERIOD_BAD = 90
INACTIVE_DAYS_THRESHOLD = 180  # 6 months
INACTIVE_DAYS_WARNING = 90

# ─────────────────────────────────────────────────────────────
# Scoring weights for composite score
# ─────────────────────────────────────────────────────────────

WEIGHTS = {
    "career_fit": 0.35,        # Most important: right background
    "skills_relevance": 0.25,  # Core skills match
    "experience_quality": 0.15, # Years + trajectory
    "behavioral": 0.15,        # Available and responsive
    "location_fit": 0.10,      # Location/logistics
}

# ─────────────────────────────────────────────────────────────
# Production / Shipping keywords in career descriptions
# ─────────────────────────────────────────────────────────────

PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "scaled", "real users",
    "live system", "a/b test", "ab test", "a/b testing",
    "user-facing", "customer-facing", "real-time", "realtime",
    "latency", "throughput", "sla", "uptime", "monitoring",
    "ci/cd", "continuous integration", "continuous deployment",
    "microservice", "api", "rest api", "grpc",
    "million users", "millions of users", "thousands of users",
    "daily active", "monthly active",
}

RESEARCH_ONLY_KEYWORDS = {
    "published paper", "conference paper", "journal paper",
    "academic", "thesis", "dissertation", "research only",
    "no production", "prototype only",
}

# Keywords indicating actual ML/AI work in career descriptions
ML_WORK_KEYWORDS = {
    "machine learning", "deep learning", "neural network",
    "model training", "model serving", "embeddings",
    "recommendation", "ranking", "search", "retrieval",
    "nlp", "natural language", "text classification",
    "feature engineering", "data pipeline", "ml pipeline",
    "inference", "prediction", "classification", "clustering",
    "transformer", "bert", "gpt", "llm",
    "vector", "similarity", "embedding", "fine-tun",
    "sentiment", "ner", "named entity",
    "xgboost", "lightgbm", "learning-to-rank", "learning to rank",
    "a/b test", "click-through", "relevance", "scoring",
    "feature pipeline", "model", "training", "evaluation",
    "offline metric", "online metric", "engagement",
    "personalization", "discovery", "content feed",
}

# Keywords indicating non-AI work
NON_AI_WORK_KEYWORDS = {
    "cad", "solidworks", "creo", "ansys", "mechanical design",
    "structural", "civil", "construction",
    "accounting", "audit", "tax", "financial reporting",
    "gaap", "statutory compliance", "accounts payable",
    "brand design", "packaging design", "adobe suite",
    "seo strategy", "editorial calendar", "copywriting",
    "support agent", "ticket", "escalation",
    "warehouse", "fulfillment", "logistics", "picking",
    "packing", "outbound", "inventory",
    "payroll", "benefits administration",
}
