"""Skill dictionary used to extract skills from resumes and job descriptions.

Add or remove skills here to adapt the system to a different domain.
SKILL_CATEGORIES is used for grouping/reporting; matching is done on the flat
list of canonical skills plus ALIASES (alias -> canonical skill name).
"""

SKILL_CATEGORIES = {
    "programming": [
        "python", "r programming", "java", "javascript", "typescript",
        "c++", "c#", "scala", "matlab", "bash", "php",
    ],
    "databases": [
        "sql", "mysql", "postgresql", "sqlite", "oracle", "mongodb", "nosql",
        "snowflake", "bigquery", "redshift",
    ],
    "analytics_bi": [
        "excel", "power bi", "tableau", "looker", "google sheets", "dax",
        "power query", "data analysis", "data visualization", "data cleaning",
        "dashboards", "reporting", "statistics", "hypothesis testing",
        "a/b testing", "forecasting", "regression", "etl", "pandas", "numpy",
        "matplotlib", "seaborn", "plotly",
    ],
    "ml_ai": [
        "machine learning", "deep learning", "scikit-learn", "tensorflow",
        "pytorch", "keras", "xgboost", "natural language processing",
        "computer vision", "spacy", "nltk", "hugging face", "llm",
        "feature engineering",
    ],
    "data_engineering_cloud": [
        "spark", "hadoop", "airflow", "kafka", "aws", "azure", "google cloud",
        "docker", "kubernetes", "git", "github", "linux",
    ],
    "web": [
        "html", "css", "react", "angular", "node.js", "django", "flask",
        "fastapi", "streamlit", "rest api",
    ],
    "business": [
        "crm", "salesforce", "recruitment", "talent acquisition", "payroll",
        "negotiation", "sales", "marketing", "seo", "financial modeling",
        "accounting", "sap",
    ],
    "soft_skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "stakeholder management", "project management", "agile", "scrum",
    ],
}

# alias (as it may appear in a resume) -> canonical skill
ALIASES = {
    "powerbi": "power bi",
    "microsoft power bi": "power bi",
    "ms excel": "excel",
    "microsoft excel": "excel",
    "advanced excel": "excel",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "postgres": "postgresql",
    "js": "javascript",
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "gcp": "google cloud",
    "amazon web services": "aws",
    "k8s": "kubernetes",
    "dashboard": "dashboards",
    "data viz": "data visualization",
    "visualization": "data visualization",
    "visualisation": "data visualization",
    "ab testing": "a/b testing",
    "a-b testing": "a/b testing",
    "problem-solving": "problem solving",
    "team work": "teamwork",
    "team player": "teamwork",
    "restful api": "rest api",
    "restful apis": "rest api",
    "rest apis": "rest api",
    "statistical analysis": "statistics",
    "big query": "bigquery",
    "huggingface": "hugging face",
    "large language models": "llm",
    "llms": "llm",
    "data cleansing": "data cleaning",
    "data wrangling": "data cleaning",
}

ALL_SKILLS = sorted({s for group in SKILL_CATEGORIES.values() for s in group})
SKILL_TO_CATEGORY = {s: c for c, group in SKILL_CATEGORIES.items() for s in group}
