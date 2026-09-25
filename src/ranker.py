"""Candidate ranking: skill match + TF-IDF + (optional) semantic embeddings + experience + education."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .features import (
    EDUCATION_LABELS,
    EXPLICIT_YEARS_RE,
    build_features,
    education_levels,
    extract_skills,
)

DEFAULT_WEIGHTS = {"skills": 0.45, "tfidf": 0.15, "semantic": 0.20, "experience": 0.12, "education": 0.08}

_NICE_HEADING = re.compile(
    r"^\s*(?:nice[\s-]*to[\s-]*have|preferred(?:\s+(?:qualifications|skills))?|good[\s-]*to[\s-]*have|"
    r"bonus(?:\s+points)?|desirable)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


# --------------------------------------------------------------------------- #
# Job description
# --------------------------------------------------------------------------- #
@dataclass
class JobProfile:
    text: str
    required_skills: list
    nice_skills: list
    min_years: float | None
    min_education: int | None


def parse_job_description(text: str) -> JobProfile:
    """Turn a JD into required skills, nice-to-have skills, minimum experience and education.

    Skills listed after a heading such as "Nice to have" / "Preferred" are treated as optional.
    """
    m = _NICE_HEADING.search(text)
    required_text, nice_text = (text[: m.start()], text[m.start():]) if m else (text, "")
    required = sorted(extract_skills(required_text))
    nice = sorted(s for s in extract_skills(nice_text) if s not in required)

    years = EXPLICIT_YEARS_RE.search(text)
    min_years = float(years.group(1)) if years else None

    levels = education_levels(required_text)
    min_education = min(levels) if levels else None
    return JobProfile(text, required, nice, min_years, min_education)


# --------------------------------------------------------------------------- #
# Similarity models
# --------------------------------------------------------------------------- #
def tfidf_similarity(jd_text: str, texts) -> np.ndarray:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform([jd_text] + list(texts))
    return cosine_similarity(matrix[0], matrix[1:]).ravel()


def _chunks(text: str, size: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def semantic_similarity(jd_text: str, texts, model_name: str = "all-MiniLM-L6-v2"):
    """Sentence-embedding similarity. Returns None if sentence-transformers is unavailable.

    Resumes are split into ~200-word chunks (the model truncates long inputs); a resume's score is
    the mean of its 3 best chunks against the job description.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    except Exception:
        return None

    all_chunks, owners = [], []
    for i, text in enumerate(texts):
        for chunk in _chunks(text):
            all_chunks.append(chunk)
            owners.append(i)
    embeddings = model.encode([jd_text] + all_chunks, normalize_embeddings=True, show_progress_bar=False)
    sims = np.asarray(embeddings[1:]) @ np.asarray(embeddings[0])
    owners = np.asarray(owners)
    scores = np.zeros(len(texts))
    for i in range(len(texts)):
        best = np.sort(sims[owners == i])[::-1][:3]
        scores[i] = best.mean() if len(best) else 0.0
    return scores


def _minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or x.max() - x.min() < 1e-12:
        return np.clip(x, 0, 1)
    return (x - x.min()) / (x.max() - x.min())


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _explain(skills: set, job: JobProfile, years: float, edu: int) -> str:
    parts = []
    req, nice = set(job.required_skills), set(job.nice_skills)
    if req:
        missing = sorted(req - skills)
        parts.append(f"matches {len(req & skills)}/{len(req)} required skills")
        if missing:
            parts.append("missing: " + ", ".join(missing))
    if nice:
        parts.append(f"{len(nice & skills)}/{len(nice)} nice-to-have")
    if job.min_years is not None:
        parts.append(f"experience {years:g} yrs vs {job.min_years:g} required")
    if job.min_education is not None:
        parts.append(f"education {EDUCATION_LABELS[edu]} vs {EDUCATION_LABELS[job.min_education]} required")
    text = "; ".join(parts)
    return text[:1].upper() + text[1:]


def rank_candidates(
    resumes: pd.DataFrame,
    jd_text: str,
    weights: dict | None = None,
    use_embeddings: bool = True,
) -> pd.DataFrame:
    """Score and rank every resume against a job description.

    `resumes` needs `id` and `text` columns. Components that are not applicable (no skills in the JD,
    no experience requirement, embeddings unavailable ...) are dropped and the remaining weights are
    re-normalised, so scores always run from 0 to 100.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    job = parse_job_description(jd_text)
    df = build_features(resumes)
    texts = df["text"].tolist()
    cand_skills = [set(s) for s in df["skills"]]

    req, nice = set(job.required_skills), set(job.nice_skills)

    def skill_score(cs: set) -> float:
        r = len(req & cs) / len(req) if req else None
        n = len(nice & cs) / len(nice) if nice else None
        if r is None:
            return n or 0.0
        return r if n is None else 0.8 * r + 0.2 * n

    components: dict[str, np.ndarray] = {}
    if req or nice:
        components["skills"] = np.array([skill_score(cs) for cs in cand_skills])
    components["tfidf"] = _minmax(tfidf_similarity(jd_text, texts))
    if use_embeddings:
        sem = semantic_similarity(jd_text, texts)
        if sem is not None:
            components["semantic"] = _minmax(sem)
    if job.min_years is not None:
        components["experience"] = np.clip(df["experience_years"] / max(job.min_years, 1e-9), 0, 1).to_numpy()
    if job.min_education is not None:
        components["education"] = np.clip(df["education_level"] / job.min_education, 0, 1).to_numpy()

    active = {k: weights.get(k, 0.0) for k in components if weights.get(k, 0.0) > 0}
    total = sum(active.values()) or 1.0
    norm_w = {k: v / total for k, v in active.items()}

    out = pd.DataFrame({"id": df["id"]})
    if "category" in df:
        out["category"] = df["category"]
    out["final_score"] = 100 * sum(norm_w[k] * components[k] for k in norm_w)
    for name in ["skills", "tfidf", "semantic", "experience", "education"]:
        out[f"{name}_score"] = components.get(name, np.full(len(df), np.nan))
    for k in norm_w:
        out[f"contrib_{k}"] = 100 * norm_w[k] * components[k]
    out["experience_years"] = df["experience_years"]
    out["education"] = df["education_level"].map(EDUCATION_LABELS)
    out["matched_required"] = [", ".join(sorted(req & cs)) for cs in cand_skills]
    out["missing_required"] = [", ".join(sorted(req - cs)) for cs in cand_skills]
    out["matched_nice"] = [", ".join(sorted(nice & cs)) for cs in cand_skills]
    out["explanation"] = [
        _explain(cs, job, y, e) for cs, y, e in zip(cand_skills, df["experience_years"], df["education_level"])
    ]

    out = out.sort_values(["final_score", "skills_score"], ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.attrs["weights_used"] = norm_w
    out.attrs["job"] = job
    return out
