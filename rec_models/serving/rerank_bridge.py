"""Post-ranking reranking bridge for service responses."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd


MAX_CONSECUTIVE_CATEGORY = 2
DEFAULT_EXPLORATION_EPSILON = 0.25
DEFAULT_EXPLORATION_RATIO = 0.2
DEFAULT_MAX_EXPLORATION_SLOTS = 2
DEFAULT_NEW_ITEM_WINDOW_DAYS = 7


def _safe_category(row: dict[str, Any]) -> str:
    category = str(row.get("main_category") or row.get("category") or "UNKNOWN").strip()
    return category or "UNKNOWN"


def _sort_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    sortable = candidates.copy()
    sortable["score"] = pd.to_numeric(sortable.get("score"), errors="coerce").fillna(0.0)
    sortable["popularity"] = pd.to_numeric(sortable.get("popularity"), errors="coerce").fillna(0.0)
    sortable["item_age_days"] = pd.to_numeric(sortable.get("item_age_days"), errors="coerce")
    if "is_new_item" in sortable.columns:
        sortable["is_new_item"] = sortable["is_new_item"].fillna(False).astype(bool)
    else:
        sortable["is_new_item"] = False
    return sortable.sort_values(["score", "popularity", "article_id"], ascending=[False, False, True]).reset_index(drop=True)


def _would_break_diversity(staged_rows: list[dict[str, Any]], candidate_row: dict[str, Any]) -> bool:
    if len(staged_rows) < MAX_CONSECUTIVE_CATEGORY:
        return False

    tail_categories = [_safe_category(row) for row in staged_rows[-MAX_CONSECUTIVE_CATEGORY:]]
    return len(set(tail_categories)) == 1 and tail_categories[-1] == _safe_category(candidate_row)


def apply_diversity_guard(candidates: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Prevent 3 consecutive items from the same category when possible."""

    if candidates.empty or top_n <= 0:
        return candidates.head(0).copy()

    remaining = candidates.to_dict(orient="records")
    staged_rows: list[dict[str, Any]] = []

    while remaining and len(staged_rows) < top_n:
        selected_index: int | None = None
        for index, row in enumerate(remaining):
            if not _would_break_diversity(staged_rows, row):
                selected_index = index
                break

        if selected_index is None:
            selected_index = 0

        staged_rows.append(remaining.pop(selected_index))

    return pd.DataFrame(staged_rows)


def _compute_exploration_slots(top_n: int, requested_slots: int | None = None) -> int:
    if top_n <= 2:
        return 0
    if requested_slots is not None:
        return max(0, min(requested_slots, DEFAULT_MAX_EXPLORATION_SLOTS, top_n))

    ratio_slots = max(1, int(round(top_n * DEFAULT_EXPLORATION_RATIO)))
    return min(ratio_slots, DEFAULT_MAX_EXPLORATION_SLOTS, max(top_n - 1, 0))


def _pick_reason(row: dict[str, Any], exploration_reason: str | None = None) -> str:
    if exploration_reason is not None:
        return exploration_reason
    return str(row.get("reason", "ranking_score"))


def select_exploration_candidates(
    remaining_candidates: pd.DataFrame,
    already_selected: pd.DataFrame,
    exploration_slots: int,
    epsilon: float = DEFAULT_EXPLORATION_EPSILON,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Select exploration candidates using epsilon-greedy priorities."""

    if remaining_candidates.empty or exploration_slots <= 0:
        return remaining_candidates.head(0).copy()

    rng = rng or random.Random()
    selected_rows: list[dict[str, Any]] = []
    selected_ids = set(already_selected.get("article_id", pd.Series(dtype=str)).astype(str))
    category_counts = already_selected.get("main_category", already_selected.get("category", pd.Series(dtype=str))).astype(str).value_counts()

    candidates = remaining_candidates.copy()
    candidates = candidates.loc[~candidates["article_id"].astype(str).isin(selected_ids)].copy()
    if candidates.empty:
        return candidates

    candidates["item_age_days"] = pd.to_numeric(candidates.get("item_age_days"), errors="coerce")
    if "is_new_item" in candidates.columns:
        candidates["is_new_item"] = candidates["is_new_item"].fillna(False).astype(bool)
    else:
        candidates["is_new_item"] = False
    candidates["category_key"] = candidates.apply(lambda row: _safe_category(row.to_dict()), axis=1)
    candidates["category_penalty"] = candidates["category_key"].map(category_counts.to_dict()).fillna(0.0)

    for _ in range(min(exploration_slots, len(candidates))):
        if candidates.empty:
            break

        use_exploration = rng.random() < epsilon
        chosen: pd.DataFrame

        if use_exploration:
            new_item_mask = candidates["is_new_item"] | (
                candidates["item_age_days"].notna() & candidates["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS)
            )
            new_item_candidates = candidates.loc[
                new_item_mask
            ].sort_values(["score", "popularity"], ascending=[False, False])
            if not new_item_candidates.empty:
                chosen = new_item_candidates.head(1).copy()
                chosen.loc[:, "reason"] = "new_item_boost"
            else:
                diverse_candidates = candidates.sort_values(
                    ["category_penalty", "score", "popularity"],
                    ascending=[True, False, False],
                )
                if not diverse_candidates.empty:
                    chosen = diverse_candidates.head(1).copy()
                    chosen.loc[:, "reason"] = "mab_exploration"
                else:
                    chosen = candidates.sort_values(["popularity", "score"], ascending=[False, False]).head(1).copy()
                    chosen.loc[:, "reason"] = "mab_exploration"
        else:
            chosen = candidates.sort_values(["score", "popularity"], ascending=[False, False]).head(1).copy()
            chosen.loc[:, "reason"] = chosen["reason"].fillna("ranking_score")

        chosen.loc[:, "is_exploration"] = chosen["reason"].isin({"mab_exploration", "new_item_boost"})
        selected_rows.extend(chosen.to_dict(orient="records"))

        chosen_id = str(chosen.iloc[0]["article_id"])
        chosen_category = str(chosen.iloc[0]["category_key"])
        selected_ids.add(chosen_id)
        category_counts[chosen_category] = int(category_counts.get(chosen_category, 0)) + 1

        candidates = candidates.loc[~candidates["article_id"].astype(str).eq(chosen_id)].copy()
        candidates["category_penalty"] = candidates["category_key"].map(category_counts.to_dict()).fillna(0.0)

    return pd.DataFrame(selected_rows)


def _exploration_positions(top_n: int, slot_count: int) -> list[int]:
    if slot_count <= 0 or top_n <= 0:
        return []
    if slot_count == 1:
        return [min(max(top_n // 3, 1), top_n - 1)]
    return sorted({min(max(top_n // 3, 1), top_n - 1), min(max((top_n * 2) // 3, 2), top_n - 1)})


def inject_exploration_slots(
    primary_ranked: pd.DataFrame,
    exploration_candidates: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Inject exploration rows into the ranked list at stable positions."""

    if top_n <= 0:
        return primary_ranked.head(0).copy()

    base_rows = primary_ranked.to_dict(orient="records")
    exploration_rows = exploration_candidates.to_dict(orient="records")
    if not exploration_rows:
        return pd.DataFrame(base_rows[:top_n])

    positions = _exploration_positions(top_n=top_n, slot_count=len(exploration_rows))
    merged_rows: list[dict[str, Any]] = []
    base_index = 0
    exploration_index = 0

    for result_index in range(top_n):
        if exploration_index < len(exploration_rows) and result_index in positions:
            merged_rows.append(exploration_rows[exploration_index])
            exploration_index += 1
            continue

        if base_index < len(base_rows):
            merged_rows.append(base_rows[base_index])
            base_index += 1
            continue

        if exploration_index < len(exploration_rows):
            merged_rows.append(exploration_rows[exploration_index])
            exploration_index += 1

    return pd.DataFrame(merged_rows[:top_n])


def rerank_recommendations(
    scored_candidates: pd.DataFrame,
    top_n: int,
    exploration_slots: int | None = None,
    epsilon: float = DEFAULT_EXPLORATION_EPSILON,
    random_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Apply service-level reranking with diversity, freshness, and exploration."""

    if scored_candidates.empty or top_n <= 0:
        return []

    ordered = _sort_candidates(scored_candidates)
    effective_top_n = min(top_n, len(ordered))
    effective_exploration_slots = min(_compute_exploration_slots(effective_top_n, exploration_slots), effective_top_n)
    primary_slots = max(effective_top_n - effective_exploration_slots, 0)

    primary_ranked = apply_diversity_guard(ordered, top_n=max(primary_slots, effective_top_n if effective_exploration_slots == 0 else primary_slots))
    remaining_ids = set(primary_ranked.get("article_id", pd.Series(dtype=str)).astype(str))
    remaining_candidates = ordered.loc[~ordered["article_id"].astype(str).isin(remaining_ids)].copy()

    rng = random.Random(random_seed)
    exploration_ranked = select_exploration_candidates(
        remaining_candidates=remaining_candidates,
        already_selected=primary_ranked,
        exploration_slots=effective_exploration_slots,
        epsilon=epsilon,
        rng=rng,
    )
    combined = inject_exploration_slots(primary_ranked=primary_ranked, exploration_candidates=exploration_ranked, top_n=effective_top_n)
    final_ranked = apply_diversity_guard(combined, top_n=effective_top_n)

    recommendations: list[dict[str, Any]] = []
    for row in final_ranked.to_dict(orient="records"):
        recommendations.append(
            {
                "product_id": str(row.get("article_id", "")),
                "score": float(row.get("score", 0.0)),
                "reason": _pick_reason(row),
                "is_exploration": bool(row.get("is_exploration", False)),
            }
        )
    return recommendations
