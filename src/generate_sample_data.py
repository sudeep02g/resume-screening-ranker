"""Generate SYNTHETIC resumes (and toy labels) so the pipeline can be tested without real personal data.

    python -m src.generate_sample_data

The labels are derived from the generating role, so they are only good for checking that the evaluation code
runs. For your real evaluation, label a sample of real/public resumes by hand (see README).
"""
from __future__ import annotations

import datetime
import random
from pathlib import Path

import pandas as pd

ROLES = {
    "Data Analyst": (20, ["python", "sql", "excel", "power bi", "tableau", "data visualization", "statistics", "data cleaning", "data analysis"],
                     ["pandas", "numpy", "dashboards", "reporting", "etl", "communication", "a/b testing", "google sheets"]),
    "BI Developer": (8, ["sql", "power bi", "dax", "data analysis", "dashboards", "excel", "reporting"],
                     ["tableau", "etl", "snowflake", "power query", "communication", "stakeholder management"]),
    "Data Scientist": (10, ["python", "machine learning", "scikit-learn", "statistics", "pandas", "sql"],
                       ["tensorflow", "pytorch", "natural language processing", "deep learning", "xgboost", "numpy", "matplotlib"]),
    "Data Engineer": (8, ["python", "sql", "spark", "airflow", "etl", "aws"],
                      ["kafka", "docker", "snowflake", "git", "linux", "bigquery"]),
    "Web Developer": (8, ["html", "css", "javascript", "react", "node.js", "git", "rest api"],
                      ["typescript", "mongodb", "docker", "angular", "teamwork"]),
    "HR Executive": (6, ["recruitment", "talent acquisition", "payroll", "communication", "excel"],
                     ["teamwork", "leadership", "negotiation"]),
    "Sales Executive": (6, ["sales", "crm", "salesforce", "negotiation", "communication"],
                        ["marketing", "excel", "teamwork", "leadership"]),
}
ALIAS_VARIANTS = {"power bi": ["Power BI", "PowerBI"], "excel": ["MS Excel", "Excel", "Advanced Excel"],
                  "scikit-learn": ["scikit-learn", "sklearn"], "natural language processing": ["NLP"],
                  "machine learning": ["Machine Learning", "ML"]}
DEGREES = [("B.Tech in Computer Science", 2), ("B.Sc in Statistics", 2), ("Bachelor of Commerce", 2),
           ("MBA", 3), ("M.Tech in Data Science", 3), ("Diploma in Information Technology", 1), ("B.E. in Electronics", 2)]
COMPANIES = ["Northwind Analytics", "BlueLake Systems", "Orbit Retail", "Helio Finance", "Zenith Software", "Pinecone Logistics"]


def _fmt(skill: str, rng: random.Random) -> str:
    return rng.choice(ALIAS_VARIANTS[skill]) if skill in ALIAS_VARIANTS else skill


def build(seed: int = 42):
    rng = random.Random(seed)
    year = datetime.date.today().year
    resumes, labels, n = [], [], 1
    for role, (count, core, extra) in ROLES.items():
        for _ in range(count):
            quality = rng.uniform(0.45, 1.0)
            skills = [s for s in core if rng.random() < quality] + [s for s in extra if rng.random() < 0.4]
            hit = sum(s in skills for s in core) / len(core)
            years = rng.choice([0, 1, 1, 2, 2, 3, 4, 5, 6, 8])
            degree, level = rng.choice(DEGREES)
            grad = year - years - rng.choice([0, 0, 1])
            lines = [f"Candidate R{n:03d}", "Email: candidate@example.com | Phone: +91 98765 43210", ""]
            if years:
                lines.append(f"Summary: {role} with {years} years of experience in analytics and reporting.")
            else:
                lines.append(f"Summary: Fresher aspiring {role}; completed academic projects and an internship.")
            lines += ["", "Skills: " + ", ".join(_fmt(s, rng) for s in skills), "", "Experience:"]
            if years:
                start, mid = year - years, year - years // 2
                if years >= 2:
                    lines.append(f"{role} Associate, {rng.choice(COMPANIES)} ({start} - {mid})")
                    lines.append(f"{role}, {rng.choice(COMPANIES)} ({mid} - Present)")
                else:
                    lines.append(f"{role}, {rng.choice(COMPANIES)} ({start} - Present)")
            else:
                lines.append(f"Intern, {rng.choice(COMPANIES)} (3 months)")
            lines += ["", "Education:", f"{degree}, City University ({grad - 4} - {grad})"]
            resumes.append({"ID": f"R{n:03d}", "Category": role, "Resume_str": "\n".join(lines)})
            # toy labels for the Data Analyst job description
            if role in ("Data Analyst", "BI Developer") and hit >= 0.6:
                label = 2
            elif role in ("Data Analyst", "BI Developer", "Data Scientist", "Data Engineer"):
                label = 1
            else:
                label = 0
            labels.append({"id": f"R{n:03d}", "label": label})
            n += 1
    rng.shuffle(resumes)
    return pd.DataFrame(resumes), pd.DataFrame(labels)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data"
    resumes, labels = build()
    resumes.to_csv(out / "sample_resumes.csv", index=False)
    labels.to_csv(out / "sample_labels_data_analyst.csv", index=False)
    print(f"Wrote {len(resumes)} synthetic resumes to {out}")
