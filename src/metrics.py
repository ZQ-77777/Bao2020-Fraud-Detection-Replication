"""Evaluation metrics used by Bao et al. (2020): AUC and top-k ranking metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score


def topk_metrics(y_true, score, k=0.01):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    n = len(y)
    n_top = max(1, int(round(k * n)))
    order = np.argsort(-s, kind="mergesort")
    top = y[order[:n_top]]
    hits = int(top.sum())
    n_pos = int(y.sum())
    # NDCG@k with binary relevance: DCG = sum_i (2^rel_i - 1) / log2(i + 1)
    disc = 1.0 / np.log2(np.arange(2, n_top + 2))
    dcg = float(np.sum(top * disc))
    ideal = np.zeros(n_top)
    ideal[:min(n_pos, n_top)] = 1
    idcg = float(np.sum(ideal * disc))
    return {
        "n_test": n,
        "n_fraud": n_pos,
        "n_top": n_top,
        "hits": hits,
        "precision_at_k": hits / n_top,
        "sensitivity_at_k": hits / n_pos if n_pos else np.nan,
        "ndcg_at_k": dcg / idcg if idcg > 0 else np.nan,
    }


def evaluate(y_true, score, k=0.01):
    y = np.asarray(y_true).astype(int)
    out = {"auc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan}
    out.update(topk_metrics(y, score, k))
    return out
