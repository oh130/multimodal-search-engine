from __future__ import annotations

import math
from typing import Iterable, Sequence


RankedItems = Sequence[str]
RelevantItems = Iterable[str]
RankedLists = Sequence[RankedItems]
RelevantLists = Sequence[RelevantItems]


def _validate_inputs(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
) -> None:
    if len(ranked_lists) != len(relevant_lists):
        raise ValueError("ranked_lists and relevant_lists must have the same length.")


def reciprocal_rank(
    ranked_items: RankedItems,
    relevant_items: RelevantItems,
) -> float:
    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    for index, item_id in enumerate(ranked_items, start=1):
        if item_id in relevant_set:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(
    ranked_items: RankedItems,
    relevant_items: RelevantItems,
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    return float(any(item_id in relevant_set for item_id in ranked_items[:k]))


def hit_rate_scores(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
    k: int,
) -> list[float]:
    _validate_inputs(ranked_lists, relevant_lists)
    return [
        hit_rate_at_k(ranked_items, relevant_items, k)
        for ranked_items, relevant_items in zip(ranked_lists, relevant_lists)
    ]


def mean_hit_rate_at_k(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
    k: int,
) -> float:
    if not ranked_lists:
        return 0.0
    scores = hit_rate_scores(ranked_lists, relevant_lists, k)
    return sum(scores) / len(scores)


def mrr_scores(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
) -> list[float]:
    _validate_inputs(ranked_lists, relevant_lists)
    return [
        reciprocal_rank(ranked_items, relevant_items)
        for ranked_items, relevant_items in zip(ranked_lists, relevant_lists)
    ]


def mean_reciprocal_rank(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
) -> float:
    if not ranked_lists:
        return 0.0
    scores = mrr_scores(ranked_lists, relevant_lists)
    return sum(scores) / len(scores)


def dcg_at_k(
    ranked_items: RankedItems,
    relevant_items: RelevantItems,
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_items)
    if not relevant_set:
        return 0.0

    score = 0.0
    for index, item_id in enumerate(ranked_items[:k], start=1):
        if item_id in relevant_set:
            score += 1.0 / math.log2(index + 1)
    return score


def idcg_at_k(
    relevant_items: RelevantItems,
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    ideal_hits = min(len(set(relevant_items)), k)
    return sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))


def ndcg_at_k(
    ranked_items: RankedItems,
    relevant_items: RelevantItems,
    k: int,
) -> float:
    ideal_dcg = idcg_at_k(relevant_items, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(ranked_items, relevant_items, k) / ideal_dcg


def ndcg_scores(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
    k: int,
) -> list[float]:
    _validate_inputs(ranked_lists, relevant_lists)
    return [
        ndcg_at_k(ranked_items, relevant_items, k)
        for ranked_items, relevant_items in zip(ranked_lists, relevant_lists)
    ]


def mean_ndcg_at_k(
    ranked_lists: RankedLists,
    relevant_lists: RelevantLists,
    k: int,
) -> float:
    if not ranked_lists:
        return 0.0
    scores = ndcg_scores(ranked_lists, relevant_lists, k)
    return sum(scores) / len(scores)


if __name__ == "__main__":
    ranked_lists = [
        ["item_7", "item_3", "item_9", "item_1"],
        ["item_2", "item_6", "item_8", "item_4"],
        ["item_5", "item_1", "item_2", "item_3"],
    ]
    relevant_lists = [
        {"item_3"},
        {"item_4", "item_8"},
        {"item_10"},
    ]
    k = 3

    print(f"HitRate@{k} scores:", hit_rate_scores(ranked_lists, relevant_lists, k))
    print(f"Mean HitRate@{k}:", round(mean_hit_rate_at_k(ranked_lists, relevant_lists, k), 4))
    print("MRR scores:", [round(score, 4) for score in mrr_scores(ranked_lists, relevant_lists)])
    print("Mean MRR:", round(mean_reciprocal_rank(ranked_lists, relevant_lists), 4))
    print(f"nDCG@{k} scores:", [round(score, 4) for score in ndcg_scores(ranked_lists, relevant_lists, k)])
    print(f"Mean nDCG@{k}:", round(mean_ndcg_at_k(ranked_lists, relevant_lists, k), 4))
