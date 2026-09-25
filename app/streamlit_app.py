"""Streamlit dashboard for the resume screening & ranking system.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.parser import clean_text, extract_text_from_bytes, load_resumes, redact_pii  # noqa: E402
from src.ranker import DEFAULT_WEIGHTS, rank_candidates  # noqa: E402

st.set_page_config(page_title="Resume Screening & Ranking", layout="wide")
st.title("AI-Powered Resume Screening & Candidate Ranking")
st.caption("Decision-support tool: it ranks candidates against a job description and explains each score. "
           "Contact details are redacted before scoring; a human should always make the final call.")

# ------------------------------- sidebar ------------------------------------ #
with st.sidebar:
    st.header("1. Job description")
    jd_files = sorted((ROOT / "data" / "job_descriptions").glob("*.txt"))
    choice = st.selectbox("Choose a sample JD", ["(paste my own)"] + [f.stem for f in jd_files],
                          index=1 if jd_files else 0)
    default_jd = (ROOT / "data" / "job_descriptions" / f"{choice}.txt").read_text() if choice != "(paste my own)" else ""
    jd_text = st.text_area("Job description text", default_jd, height=220)

    st.header("2. Resumes")
    uploads = st.file_uploader("Upload resumes (CSV, PDF, DOCX, TXT)", type=["csv", "pdf", "docx", "txt"],
                               accept_multiple_files=True)
    st.caption("No upload? The synthetic sample dataset is used.")

    st.header("3. Scoring weights")
    weights = {
        "skills": st.slider("Skill match", 0.0, 1.0, DEFAULT_WEIGHTS["skills"], 0.05),
        "tfidf": st.slider("Keyword similarity (TF-IDF)", 0.0, 1.0, DEFAULT_WEIGHTS["tfidf"], 0.05),
        "semantic": st.slider("Semantic similarity (embeddings)", 0.0, 1.0, DEFAULT_WEIGHTS["semantic"], 0.05),
        "experience": st.slider("Experience", 0.0, 1.0, DEFAULT_WEIGHTS["experience"], 0.05),
        "education": st.slider("Education", 0.0, 1.0, DEFAULT_WEIGHTS["education"], 0.05),
    }
    use_embeddings = st.checkbox("Use sentence embeddings (needs sentence-transformers)", value=False)
    top_n = st.slider("Candidates to show in charts", 5, 30, 10)


def read_resumes(files) -> pd.DataFrame:
    if not files:
        return load_resumes(ROOT / "data" / "sample_resumes.csv")
    frames = []
    for f in files:
        if f.name.lower().endswith(".csv"):
            tmp = ROOT / "outputs" / f"_upload_{f.name}"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(f.getvalue())
            frames.append(load_resumes(tmp))
            tmp.unlink(missing_ok=True)
        else:
            text = redact_pii(clean_text(extract_text_from_bytes(f.name, f.getvalue())))
            frames.append(pd.DataFrame({"id": [Path(f.name).stem], "category": [""], "text": [text]}))
    return pd.concat(frames, ignore_index=True)


if not jd_text.strip():
    st.info("Choose or paste a job description in the sidebar to start.")
    st.stop()

resumes = read_resumes(uploads)
ranked = rank_candidates(resumes, jd_text, weights=weights, use_embeddings=use_embeddings)
job = ranked.attrs["job"]

# ------------------------------- summary ------------------------------------ #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Resumes screened", len(ranked))
c2.metric("Required skills found in JD", len(job.required_skills))
c3.metric("Min. experience", f"{job.min_years:g} yrs" if job.min_years is not None else "n/a")
c4.metric("Top score", f"{ranked['final_score'].max():.1f}")
st.write("**Required skills:** " + (", ".join(job.required_skills) or "none detected")
         + "  \n**Nice to have:** " + (", ".join(job.nice_skills) or "none"))

# ------------------------------- ranking ------------------------------------ #
st.subheader("Ranked candidates")
table_cols = ["rank", "id", "category", "final_score", "skills_score", "tfidf_score", "semantic_score",
              "experience_years", "education", "explanation"]
table = ranked[[c for c in table_cols if c in ranked.columns]].round(2)
table.attrs = {}
st.dataframe(table, width="stretch", hide_index=True)
st.download_button("Download full ranking (CSV)",
                   ranked.drop(columns=[c for c in ranked.columns if c.startswith("contrib_")]).to_csv(index=False),
                   file_name="ranked_candidates.csv", mime="text/csv")

left, right = st.columns(2)
with left:
    st.subheader(f"Score breakdown - top {top_n}")
    contrib_cols = [c for c in ranked.columns if c.startswith("contrib_")]
    chart = ranked.head(top_n).set_index("id")[contrib_cols]
    chart.columns = [c.replace("contrib_", "") for c in chart.columns]
    st.bar_chart(chart)
with right:
    st.subheader(f"Most common missing required skills - top {top_n}")
    missing = (ranked.head(top_n)["missing_required"].str.split(", ").explode().replace("", pd.NA).dropna()
               .value_counts())
    if missing.empty:
        st.write("The top candidates cover every required skill.")
    else:
        st.bar_chart(missing)

# ------------------------------- candidate detail --------------------------- #
st.subheader("Candidate detail")
selected = st.selectbox("Pick a candidate", ranked["id"])
row = ranked[ranked["id"] == selected].iloc[0]
d1, d2 = st.columns(2)
d1.write(f"**Rank {int(row['rank'])} - score {row['final_score']:.1f}/100**")
d1.write(row["explanation"])
d2.write("**Matched required skills:** " + (row["matched_required"] or "none"))
d2.write("**Missing required skills:** " + (row["missing_required"] or "none"))
d2.write("**Matched nice-to-have:** " + (row["matched_nice"] or "none"))
with st.expander("Show (redacted) resume text"):
    st.text(resumes.loc[resumes["id"] == selected, "text"].iloc[0])
