"""Reusable metrics for retrieval and ranking evaluation."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from sklearn.metrics import roc_auc_score


RankedItems = Sequence[str]
RelevantItems = Iterable[str]
RankedLists = Sequence[RankedItems]
RelevantLists = Sequence[RelevantItems]


def recall_at_k(ranked_items: RankedItems, relevant_items: RelevantItems, k: int) -> float:
    """Compute recall@k for one ranked list."""

    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    hits = sum(1 for item_id in ranked_items[:k] if item_id in relevant_set)
    return hits / len(relevant_set)


def hit_rate_at_k(ranked_items: RankedItems, relevant_items: RelevantItems, k: int) -> float:
    """Return 1.0 if any relevant item appears in top-k else 0.0."""

    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    return float(any(item_id in relevant_set for item_id in ranked_items[:k]))


def ndcg_at_k(ranked_items: RankedItems, relevant_items: RelevantItems, k: int) -> float:
    """Compute binary-gain nDCG@k."""

    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    dcg = 0.0
    for index, item_id in enumerate(ranked_items[:k], start=1):
        if item_id in relevant_set:
            dcg += 1.0 / math.log2(index + 1)

    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def coverage_at_k(ranked_lists: RankedLists, total_candidate_count: int, k: int) -> float:
    """Compute catalog coverage at top-k."""

    if k <= 0 or total_candidate_count <= 0:
        return 0.0

    recommended_items = {item_id for ranked_items in ranked_lists for item_id in ranked_items[:k]}
    return len(recommended_items) / total_candidate_count


def mean_metric(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
    metric_fn,
    k: int,
) -> float:
    """Average a per-user ranking metric across users."""

    ranked_lists = list(ranked_lists)
    relevant_lists = list(relevant_lists)
    if not ranked_lists:
        return 0.0

    scores = [metric_fn(ranked_items, relevant_items, k) for ranked_items, relevant_items in zip(ranked_lists, relevant_lists)]
    return sum(scores) / len(scores)


def safe_roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Return ROC-AUC only when both classes are present."""

    label_values = list(labels)
    if len(set(label_values)) < 2:
        return None
    return float(roc_auc_score(label_values, list(scores)))
