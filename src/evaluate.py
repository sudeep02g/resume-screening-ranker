"""Evaluate rankings against manually labelled resumes (Precision@K, NDCG@K, Spearman)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

SCORE_COLUMNS = ["final_score", "skills_score", "tfidf_score", "semantic_score"]


def evaluate_rankings(
    ranked: pd.DataFrame,
    labels: pd.DataFrame,
    ks=(3, 5, 10),
    relevant_threshold: int = 2,
    score_columns=SCORE_COLUMNS,
) -> pd.DataFrame:
    """Compare every scoring method against the labels.

    `labels` needs `id` and `label` (e.g. 0 = poor fit, 1 = average, 2 = good fit).
    A candidate counts as "relevant" for Precision@K when label >= relevant_threshold.
    """
    left = ranked.assign(id=ranked["id"].astype(str))
    right = labels[["id", "label"]].assign(id=lambda d: d["id"].astype(str))
    merged = left.merge(right, on="id", how="inner")
    if merged.empty:
        raise ValueError("No overlap between ranked ids and label ids.")

    y = merged["label"].to_numpy(dtype=float)
    rows = []
    for col in score_columns:
        if col not in merged or merged[col].isna().all():
            continue
        s = merged[col].fillna(0).to_numpy(dtype=float)
        order = np.argsort(-s, kind="stable")
        row = {"method": col.replace("_score", "")}
        for k in ks:
            k_eff = min(k, len(y))
            row[f"P@{k}"] = float((y[order][:k_eff] >= relevant_threshold).mean())
        for k in ks:
            row[f"NDCG@{k}"] = float(ndcg_score([y], [s], k=min(k, len(y))))
        row["Spearman"] = float(spearmanr(s, y).correlation)
        rows.append(row)
    return pd.DataFrame(rows).set_index("method").round(3)
