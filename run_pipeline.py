"""Command-line interface: rank resumes against a job description.

Example:
    python run_pipeline.py --jd data/job_descriptions/data_analyst.txt \
        --resumes data/sample_resumes.csv --top 10 \
        --labels data/sample_labels_data_analyst.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_rankings
from src.parser import load_resumes
from src.ranker import rank_candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank resumes against a job description.")
    ap.add_argument("--jd", required=True, help="Path to a job description .txt file")
    ap.add_argument("--resumes", required=True, help="CSV file, or a folder of PDF/DOCX/TXT resumes")
    ap.add_argument("--top", type=int, default=10, help="How many candidates to print")
    ap.add_argument("--out", default=None, help="Where to save the full ranking (CSV)")
    ap.add_argument("--labels", default=None, help="Optional CSV with columns id,label to evaluate the ranking")
    ap.add_argument("--no-embeddings", action="store_true", help="Skip sentence-embedding similarity")
    args = ap.parse_args()

    jd_path = Path(args.jd)
    resumes = load_resumes(args.resumes)
    ranked = rank_candidates(resumes, jd_path.read_text(encoding="utf-8"), use_embeddings=not args.no_embeddings)

    out = Path(args.out) if args.out else Path("outputs") / f"ranked_{jd_path.stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.drop(columns=[c for c in ranked.columns if c.startswith("contrib_")]).to_csv(out, index=False)

    job = ranked.attrs["job"]
    print(f"\nJob: {jd_path.stem} | required skills: {', '.join(job.required_skills)}")
    print("Weights used:", {k: round(v, 2) for k, v in ranked.attrs["weights_used"].items()})
    cols = ["rank", "id", "final_score", "experience_years", "explanation"]
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(ranked[cols].head(args.top).round(1).to_string(index=False))
    print(f"\nFull ranking saved to {out}")

    if args.labels:
        table = evaluate_rankings(ranked, pd.read_csv(args.labels))
        print("\nEvaluation against labels:\n", table.to_string())


if __name__ == "__main__":
    main()
