import argparse
import random
import sys
import textwrap
from pathlib import Path

import pandas as pd

from src.parser import load_resumes
from src.ranker import rank_candidates

# Kaggle categories that could plausibly contain data-analyst-type candidates
RELEVANT = {"INFORMATION-TECHNOLOGY", "BUSINESS-DEVELOPMENT", "FINANCE", "ACCOUNTANT", "CONSULTANT",
            "ENGINEERING", "BANKING"}

RUBRIC = """
RUBRIC (for the Data Analyst job)
  2 = GOOD FIT     several of SQL / Python / Excel / Power BI or Tableau / statistics AND real data-analysis or BI work
  1 = PARTIAL FIT  some relevant skills, or an adjacent role (IT, finance, reporting), but key skills/experience missing
  0 = NOT A FIT    unrelated field, or almost none of the required skills
Judge ONLY skills and experience. Ignore names, gender, age, photos and location.
"""


def build_pool(ranked, n, seed=42):
    rng = random.Random(seed)
    ids = ranked["id"].tolist()
    relevant = ranked.loc[ranked["category"].isin(RELEVANT), "id"].tolist()
    other = ranked.loc[~ranked["category"].isin(RELEVANT), "id"].tolist()
    top_skill = ranked.sort_values("skills_score", ascending=False)["id"].head(40).tolist()
    top_tfidf = ranked.sort_values("tfidf_score", ascending=False)["id"].head(40).tolist()
    picks = []

    def take(pool, k):
        pool = [i for i in pool if i not in picks]
        picks.extend(rng.sample(pool, min(k, len(pool))))

    take(top_skill, round(n * 0.3))
    take(top_tfidf, round(n * 0.3))
    take(relevant, round(n * 0.25))
    take(other, round(n * 0.15))
    take(ids, n - len(picks))
    rng.shuffle(picks)
    return picks[:n]


def show(text, start, chars):
    chunk = " ".join(text[start:start + chars].split())
    print(textwrap.fill(chunk, 110) if chunk else "(end of resume)")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", default="data/job_descriptions/data_analyst.txt")
    ap.add_argument("--resumes", default="data/raw/Resume.csv")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--chars", type=int, default=1500)
    args = ap.parse_args()

    resumes = load_resumes(args.resumes)
    ranked = rank_candidates(resumes, Path(args.jd).read_text(encoding="utf-8"), use_embeddings=False)
    job = ranked.attrs["job"]
    pool = build_pool(ranked, args.n)
    text_by_id = dict(zip(resumes["id"], resumes["text"]))

    labels_path = Path(args.labels)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(labels_path, dtype={"id": str}).to_dict("records") if labels_path.exists() else []
    labelled = {r["id"] for r in rows}
    todo = [i for i in pool if i not in labelled]

    print(f"\nJob: Data Analyst | required skills: {', '.join(job.required_skills)}")
    print(RUBRIC)
    print(f"Already labelled: {len(rows)} | to do now: {len(todo)}")

    for k, rid in enumerate(todo, 1):
        text, pos = text_by_id[rid], 0
        print("\n" + "=" * 100)
        print(f"Resume {k} of {len(todo)}   (id {rid})")
        print("=" * 100)
        show(text, pos, args.chars)
        while True:
            ans = input("\nLabel: 0 / 1 / 2   (m = show more, s = skip, q = save & quit): ").strip().lower()
            if ans in {"0", "1", "2"}:
                rows.append({"id": rid, "label": int(ans)})
                pd.DataFrame(rows).to_csv(labels_path, index=False)
                break
            if ans == "m":
                pos += args.chars
                print()
                show(text, pos, args.chars)
            elif ans == "s":
                break
            elif ans == "q":
                print(f"Saved {len(rows)} labels to {labels_path}. Run the script again to continue.")
                return
            else:
                print("Please type 0, 1, 2, m, s or q.")
    print(f"\nDone! {len(rows)} labels saved to {labels_path}.")


if __name__ == "__main__":
    main()