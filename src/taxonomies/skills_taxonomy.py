# ---------------------------------------------------------------------------
# SKILLS TAXONOMY
# Key: canonical skill name  |  Value: list of aliases / surface forms
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY = {
    # ── Programming Languages ──────────────────────────────────────────────
    "python": ["python", "python3", "python2"],
    "r": [r"\br\b", "r programming", "r language"],
    "sql": ["sql", "mysql", "postgresql", "sqlite", "t-sql", "pl/sql", "nosql"],
    "java": ["java", "java8", "java11"],
    "javascript": ["javascript", "js", "es6", "es2015", "node.js", "nodejs"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp", "c plus plus"],
    "c#": ["c#", "csharp", "c sharp"],
    "go": [r"\bgo\b", "golang"],
    "rust": ["rust"],
    "scala": ["scala"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "bash": ["bash", "shell scripting", "shell script", "zsh"],
    "matlab": ["matlab"],
    "julia": ["julia"],

    # ── ML / AI / Data Science ─────────────────────────────────────────────
    "machine learning": ["machine learning", r"\bml\b", "supervised learning",
                          "unsupervised learning", "semi-supervised"],
    "deep learning": ["deep learning", r"\bdl\b", "neural network", "neural networks",
                       "ann", "dnn"],
    "natural language processing": ["natural language processing", r"\bnlp\b",
                                     "text mining", "text analytics", "language model"],
    "computer vision": ["computer vision", r"\bcv\b", "image recognition",
                         "object detection", "image processing"],
    "reinforcement learning": ["reinforcement learning", r"\brl\b"],
    "feature engineering": ["feature engineering", "feature extraction",
                              "feature selection"],
    "model evaluation": ["model evaluation", "cross validation", "cross-validation",
                          "hyperparameter tuning", "grid search", "model selection"],
    "statistics": ["statistics", "statistical analysis", "statistical modeling",
                    "probability", "bayesian", "hypothesis testing", "regression"],
    "time series": ["time series", "forecasting", "arima", "lstm forecasting"],
    "recommendation systems": ["recommendation system", "recommender system",
                                 "collaborative filtering", "content-based filtering"],

    # ── ML Frameworks & Libraries ──────────────────────────────────────────
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "keras": ["keras"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "xgboost": ["xgboost", "xgb"],
    "lightgbm": ["lightgbm", "lgbm"],
    "catboost": ["catboost"],
    "hugging face": ["hugging face", "huggingface", "transformers library"],
    "openai": ["openai", "gpt", "chatgpt", "gpt-4", "gpt4"],
    "langchain": ["langchain", "lang chain"],
    "spacy": ["spacy", "spaCy"],
    "nltk": ["nltk"],
    "gensim": ["gensim"],
    "opencv": ["opencv", "cv2"],

    # ── Data Engineering & Big Data ────────────────────────────────────────
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "spark": ["apache spark", "pyspark", "spark streaming"],
    "hadoop": ["hadoop", "hdfs", "mapreduce"],
    "kafka": ["kafka", "apache kafka"],
    "airflow": ["airflow", "apache airflow"],
    "dbt": ["dbt", "data build tool"],
    "etl": ["etl", "data pipeline", "data ingestion"],
    "data warehousing": ["data warehouse", "data warehousing", "snowflake",
                          "redshift", "bigquery", "data mart"],

    # ── Cloud Platforms ────────────────────────────────────────────────────
    "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "sagemaker",
             "aws glue", "aws redshift"],
    "gcp": ["gcp", "google cloud", "google cloud platform", "bigquery",
             "vertex ai", "cloud run"],
    "azure": ["azure", "microsoft azure", "azure ml", "azure databricks"],

    # ── MLOps & DevOps ─────────────────────────────────────────────────────
    "docker": ["docker", "containerization", "container"],
    "kubernetes": ["kubernetes", "k8s"],
    "ci/cd": ["ci/cd", "continuous integration", "continuous deployment",
               "github actions", "jenkins", "gitlab ci"],
    "mlflow": ["mlflow"],
    "wandb": ["wandb", "weights and biases", "weights & biases"],
    "git": ["git", "github", "gitlab", "version control"],
    "linux": ["linux", "unix"],

    # ── Databases ──────────────────────────────────────────────────────────
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elastic search"],
    "cassandra": ["cassandra"],
    "postgresql": ["postgresql", "postgres"],

    # ── Web & API ──────────────────────────────────────────────────────────
    "fastapi": ["fastapi", "fast api"],
    "flask": ["flask"],
    "django": ["django"],
    "rest api": ["rest api", "restful", "rest", "api development"],
    "graphql": ["graphql"],

    # ── Visualization ──────────────────────────────────────────────────────
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly", "dash"],
    "looker": ["looker"],

    # ── Soft Skills ────────────────────────────────────────────────────────
    "communication": ["communication", "written communication", "verbal communication",
                       "presentation skills", "public speaking"],
    "teamwork": ["teamwork", "team player", "collaboration", "collaborative"],
    "problem solving": ["problem solving", "problem-solving", "analytical thinking",
                         "critical thinking"],
    "leadership": ["leadership", "team lead", "mentoring", "mentorship"],
    "agile": ["agile", "scrum", "kanban", "sprint"],
    "project management": ["project management", "pmp", "jira", "confluence",
                             "trello", "asana"],
}