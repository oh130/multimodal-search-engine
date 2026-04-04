"""Service-side candidate generation helpers.

This module provides a lightweight candidate source for serving so the baseline
ranking model can be connected end-to-end before a production retrieval model
is wired in.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
ARTICLE_FEATURES_PATH = BASE_DIR / "data" / "processed" / "articles_feature.csv"
ITEM_FEATURES_PATH = BASE_DIR / "data" / "processed" / "item_features.csv"
DEFAULT_CANDIDATE_POOL_SIZE = 100
DEFAULT_NEW_ITEM_WINDOW_DAYS = 7
DEFAULT_SIGNAL_LOOKUP_LIMIT_MULTIPLIER = 4
LOOKUP_COLUMNS = (
    "article_id",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_master_name",
    "department_name",
    "section_name",
    "garment_group_name",
    "category",
    "main_category",
    "color",
    "popularity",
    "item_age_days",
    "is_new_item",
)


@dataclass(frozen=True)
class ServingFeatureStore:
    catalog: pd.DataFrame
    article_records: dict[str, dict[str, Any]]
    category_to_ids: dict[str, tuple[str, ...]]
    main_category_to_ids: dict[str, tuple[str, ...]]
    color_to_ids: dict[str, tuple[str, ...]]
    popular_article_ids: tuple[str, ...]
    popularity_max: float


def normalize_article_id(article_id: Any) -> str:
    """Normalize article ids to the zero-padded training format."""

    text = str(article_id).strip()
    if text.isdigit():
        return text.zfill(10)
    return text


def _safe_text(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "UNKNOWN"


def _build_lookup_map(catalog: pd.DataFrame, column: str) -> dict[str, tuple[str, ...]]:
    grouped = catalog.groupby(column, sort=False)["article_id"].apply(tuple)
    return {str(key): value for key, value in grouped.items()}


def _build_article_records(catalog: pd.DataFrame, popularity_max: float) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in catalog.loc[:, LOOKUP_COLUMNS].to_dict(orient="records"):
        article_id = str(row["article_id"])
        popularity = float(row.get("popularity", 0.0) or 0.0)
        item_age_days = row.get("item_age_days")
        numeric_item_age_days = float(item_age_days) if pd.notna(item_age_days) else math.nan
        is_new_item = bool(row.get("is_new_item", False))
        records[article_id] = {
            **row,
            "article_id": article_id,
            "category": _safe_text(row.get("category")),
            "main_category": _safe_text(row.get("main_category")),
            "color": _safe_text(row.get("color")),
            "popularity": popularity,
            "item_age_days": numeric_item_age_days,
            "is_new_item": is_new_item,
            "normalized_popularity": popularity / popularity_max,
            "fresh_boost": 0.5 if is_new_item else 0.0,
        }
    return records


@lru_cache(maxsize=1)
def load_serving_artifacts() -> ServingFeatureStore:
    """Load and cache serving-time article metadata and lookup maps."""

    if not ARTICLE_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Article feature file not found: {ARTICLE_FEATURES_PATH}")

    catalog = pd.read_csv(ARTICLE_FEATURES_PATH, dtype=str).fillna("UNKNOWN")
    catalog["article_id"] = catalog["article_id"].map(normalize_article_id)

    if ITEM_FEATURES_PATH.exists():
        popularity = pd.read_csv(ITEM_FEATURES_PATH, dtype={"article_id": str})
        popularity["article_id"] = popularity["article_id"].map(normalize_article_id)
        popularity["popularity"] = pd.to_numeric(popularity.get("popularity"), errors="coerce").fillna(0.0)
        catalog = catalog.merge(
            popularity.loc[:, ["article_id", "popularity"]],
            on="article_id",
            how="left",
        )
    else:
        LOGGER.warning("Item popularity file not found: %s", ITEM_FEATURES_PATH)
        catalog["popularity"] = 0.0

    catalog["popularity"] = pd.to_numeric(catalog["popularity"], errors="coerce").fillna(0.0)
    catalog["item_age_days"] = pd.to_numeric(catalog.get("item_age_days"), errors="coerce")
    catalog["is_new_item"] = catalog["item_age_days"].le(DEFAULT_NEW_ITEM_WINDOW_DAYS).fillna(False)
    for column in ("category", "main_category", "color"):
        catalog[column] = catalog[column].map(_safe_text)

    catalog = catalog.sort_values(["popularity", "article_id"], ascending=[False, True]).reset_index(drop=True)
    category_rank = catalog.groupby("main_category", dropna=False).cumcount()
    catalog["cold_start_bonus"] = category_rank.rsub(9).clip(lower=0) / 10.0

    popularity_max = max(float(catalog["popularity"].max()), 1.0)
    article_records = _build_article_records(catalog=catalog, popularity_max=popularity_max)
    for article_id, cold_start_bonus in zip(catalog["article_id"].astype(str), catalog["cold_start_bonus"], strict=False):
        article_records[article_id]["cold_start_bonus"] = float(cold_start_bonus)

    LOGGER.info("Loaded serving article artifacts with %s catalog rows", len(catalog))
    return ServingFeatureStore(
        catalog=catalog,
        article_records=article_records,
        category_to_ids=_build_lookup_map(catalog, "category"),
        main_category_to_ids=_build_lookup_map(catalog, "main_category"),
        color_to_ids=_build_lookup_map(catalog, "color"),
        popular_article_ids=tuple(catalog["article_id"].astype(str)),
        popularity_max=popularity_max,
    )


def get_cached_feature_store() -> ServingFeatureStore:
    """Return the singleton-style feature store used by serving."""

    return load_serving_artifacts()


@lru_cache(maxsize=1)
def load_article_catalog() -> pd.DataFrame:
    """Compatibility accessor used by evaluation helpers."""

    return get_cached_feature_store().catalog


def _build_recent_signal_sets(
    feature_store: ServingFeatureStore,
    recent_clicks: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """Extract metadata signal sets from recently clicked items."""

    if not recent_clicks:
        return set(), set(), set()

    categories: set[str] = set()
    main_categories: set[str] = set()
    colors: set[str] = set()
    for article_id in recent_clicks:
        record = feature_store.article_records.get(article_id)
        if record is None:
            continue
        categories.add(str(record["category"]))
        main_categories.add(str(record["main_category"]))
        colors.add(str(record["color"]))
    return categories, main_categories, colors


def _accumulate_scores(
    candidate_scores: dict[str, float],
    candidate_matches: dict[str, bool],
    article_ids: tuple[str, ...],
    increment: float,
    limit: int | None = None,
) -> None:
    iterable = article_ids if limit is None else article_ids[:limit]
    for article_id in iterable:
        candidate_scores[article_id] = candidate_scores.get(article_id, 0.0) + increment
        candidate_matches[article_id] = True


def _materialize_candidates(
    selected_ids: list[str],
    feature_store: ServingFeatureStore,
    candidate_scores: dict[str, float],
    recent_matches: dict[str, bool],
    session_matches: dict[str, bool],
    candidate_reason: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for article_id in selected_ids:
        record = feature_store.article_records[article_id]
        rows.append(
            {
                **{column: record.get(column) for column in LOOKUP_COLUMNS},
                "candidate_score": candidate_scores[article_id],
                "candidate_reason": candidate_reason,
                "matches_recent_click_signal": recent_matches.get(article_id, False),
                "matches_session_interest": session_matches.get(article_id, False),
            }
        )
    return pd.DataFrame(rows)


def _cold_start_candidates(
    feature_store: ServingFeatureStore,
    recent_click_set: set[str],
    candidate_pool_size: int,
) -> pd.DataFrame:
    selected_ids: list[str] = []
    candidate_scores: dict[str, float] = {}

    for article_id in feature_store.popular_article_ids:
        if article_id in recent_click_set:
            continue
        record = feature_store.article_records[article_id]
        candidate_scores[article_id] = float(record["normalized_popularity"]) + float(record["cold_start_bonus"]) + float(record["fresh_boost"])
        selected_ids.append(article_id)
        if len(selected_ids) >= candidate_pool_size:
            break

    return _materialize_candidates(
        selected_ids=selected_ids,
        feature_store=feature_store,
        candidate_scores=candidate_scores,
        recent_matches={},
        session_matches={},
        candidate_reason="cold_start_popularity",
    )


def generate_candidates(
    user_id: str,
    top_k: int,
    recent_clicks: list[str] | None = None,
    session_interest: dict[str, Any] | None = None,
    candidate_pool_size: int | None = None,
) -> pd.DataFrame:
    """Generate ranking candidates using catalog metadata and popularity.

    This keeps a cold-start-safe fallback in place until a dedicated candidate
    retrieval service is connected.
    """

    del user_id

    feature_store = get_cached_feature_store()
    recent_clicks = [normalize_article_id(article_id) for article_id in (recent_clicks or []) if str(article_id).strip()]
    recent_click_set = set(recent_clicks)
    session_interest = session_interest or {}

    candidate_pool_size = candidate_pool_size or max(DEFAULT_CANDIDATE_POOL_SIZE, top_k * 10)
    cold_start = not recent_clicks and not session_interest
    signal_lookup_limit = max(candidate_pool_size * DEFAULT_SIGNAL_LOOKUP_LIMIT_MULTIPLIER, candidate_pool_size)

    if cold_start:
        filtered = _cold_start_candidates(
            feature_store=feature_store,
            recent_click_set=recent_click_set,
            candidate_pool_size=candidate_pool_size,
        )
        LOGGER.info(
            "Generated %s candidates for ranking (top_k=%s, cold_start=%s)",
            len(filtered),
            top_k,
            cold_start,
        )
        return filtered.reset_index(drop=True)

    candidate_scores: dict[str, float] = {}
    recent_matches: dict[str, bool] = {}
    session_matches: dict[str, bool] = {}

    if recent_clicks:
        categories, main_categories, colors = _build_recent_signal_sets(feature_store, recent_clicks)
        for category in categories:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.category_to_ids.get(category, ()),
                3.0,
                limit=signal_lookup_limit,
            )
        for main_category in main_categories:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.main_category_to_ids.get(main_category, ()),
                2.0,
                limit=signal_lookup_limit,
            )
        for color in colors:
            _accumulate_scores(
                candidate_scores,
                recent_matches,
                feature_store.color_to_ids.get(color, ()),
                1.0,
                limit=signal_lookup_limit,
            )

    for category, weight in session_interest.items():
        normalized_weight = float(weight) if weight is not None else 0.0
        normalized_category = _safe_text(category)
        _accumulate_scores(
            candidate_scores,
            session_matches,
            feature_store.category_to_ids.get(normalized_category, ()),
            normalized_weight * 4.0,
            limit=signal_lookup_limit,
        )
        _accumulate_scores(
            candidate_scores,
            session_matches,
            feature_store.main_category_to_ids.get(normalized_category, ()),
            normalized_weight * 2.0,
            limit=signal_lookup_limit,
        )

    for article_id, signal_score in list(candidate_scores.items()):
        if article_id in recent_click_set:
            del candidate_scores[article_id]
            recent_matches.pop(article_id, None)
            session_matches.pop(article_id, None)
            continue

        record = feature_store.article_records[article_id]
        candidate_scores[article_id] = signal_score + float(record["normalized_popularity"]) + float(record["fresh_boost"])

    if len(candidate_scores) < candidate_pool_size:
        for article_id in feature_store.popular_article_ids:
            if article_id in recent_click_set or article_id in candidate_scores:
                continue
            record = feature_store.article_records[article_id]
            candidate_scores[article_id] = float(record["normalized_popularity"]) + float(record["fresh_boost"])
            if len(candidate_scores) >= candidate_pool_size:
                break

    selected_ids = sorted(
        candidate_scores,
        key=lambda article_id: (
            -candidate_scores[article_id],
            -float(feature_store.article_records[article_id]["popularity"]),
            article_id,
        ),
    )[:candidate_pool_size]
    filtered = _materialize_candidates(
        selected_ids=selected_ids,
        feature_store=feature_store,
        candidate_scores=candidate_scores,
        recent_matches=recent_matches,
        session_matches=session_matches,
        candidate_reason="candidate_retrieval",
    )

    LOGGER.info(
        "Generated %s candidates for ranking (top_k=%s, cold_start=%s)",
        len(filtered),
        top_k,
        cold_start,
    )
    return filtered.reset_index(drop=True)
