from src.features import extract_education_level, extract_experience_years, extract_skills
from src.parser import redact_pii
from src.ranker import parse_job_description, rank_candidates

import pandas as pd


def test_skill_extraction_handles_aliases_and_boundaries():
    text = "Worked with MS Excel, PowerBI, mysql and C++. Built models with sklearn using Node.js tooling."
    skills = extract_skills(text)
    assert {"excel", "power bi", "mysql", "c++", "scikit-learn", "node.js"} <= skills
    assert "sql" not in skills  # 'mysql' must not count as 'sql'
    assert "javascript" not in skills  # 'node.js' must not trigger the 'js' alias


def test_experience_prefers_explicit_statement():
    assert extract_experience_years("Analyst with 4+ years of professional experience") == 4.0


def test_experience_from_date_ranges_ignores_education():
    text = "B.Tech, ABC University (2014 - 2018)\nAnalyst, X (2020 - 2022)\nAnalyst, Y (2022 - Present)"
    assert extract_experience_years(text, current_year=2026) == 6.0


def test_education_levels():
    assert extract_education_level("Scrum Master, should be fine") == 0
    assert extract_education_level("B.Tech in CS") == 2
    assert extract_education_level("MBA, B.E. Mechanical") == 3
    assert extract_education_level("PhD in Statistics") == 4


def test_pii_redaction():
    cleaned = redact_pii("Reach me at jane@example.com or +91 98765 43210, https://linkedin.com/in/jane")
    assert "@" not in cleaned and "9876" not in cleaned and "linkedin" not in cleaned


def test_job_description_parsing():
    jd = "Requirements:\n- 3+ years of experience\n- Bachelor's degree\n- Python and SQL\nNice to have:\n- Tableau"
    job = parse_job_description(jd)
    assert job.required_skills == ["python", "sql"]
    assert job.nice_skills == ["tableau"]
    assert job.min_years == 3.0 and job.min_education == 2


def test_ranking_orders_best_match_first():
    jd = "Requirements:\n- Python, SQL and Excel\n- 2+ years of experience"
    resumes = pd.DataFrame({
        "id": ["weak", "strong"],
        "text": ["Sales executive with CRM and negotiation. 5 years of experience.",
                 "Analyst skilled in Python, SQL and Excel. 3 years of experience."],
    })
    ranked = rank_candidates(resumes, jd, use_embeddings=False)
    assert ranked.loc[0, "id"] == "strong"
    assert ranked["final_score"].between(0, 100).all()
