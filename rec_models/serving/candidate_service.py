"""Service-side candidate generation helpers.

This module provides a lightweight candidate source for serving so the baseline
ranking model can be connected end-to-end before a production retrieval model
is wired in.
"""

from __future__ import annotations

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


def normalize_article_id(article_id: Any) -> str:
    """Normalize article ids to the zero-padded training format."""

    text = str(article_id).strip()
    if text.isdigit():
        return text.zfill(10)
    return text


@lru_cache(maxsize=1)
def load_article_catalog() -> pd.DataFrame:
    """Load article metadata and popularity used for candidate generation."""

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
    catalog = catalog.sort_values(["popularity", "article_id"], ascending=[False, True]).reset_index(drop=True)
    return catalog


def _build_recent_signal_sets(
    catalog: pd.DataFrame,
    recent_clicks: list[str],
) -> tuple[set[str], set[str], set[str]]:
    """Extract metadata signal sets from recently clicked items."""

    if not recent_clicks:
        return set(), set(), set()

    clicked_items = catalog[catalog["article_id"].isin(recent_clicks)]
    categories = set(clicked_items["category"].dropna().astype(str))
    main_categories = set(clicked_items["main_category"].dropna().astype(str))
    colors = set(clicked_items["color"].dropna().astype(str))
    return categories, main_categories, colors


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

    catalog = load_article_catalog().copy()
    recent_clicks = [normalize_article_id(article_id) for article_id in (recent_clicks or []) if str(article_id).strip()]
    recent_click_set = set(recent_clicks)
    session_interest = session_interest or {}

    candidate_pool_size = candidate_pool_size or max(DEFAULT_CANDIDATE_POOL_SIZE, top_k * 10)
    cold_start = not recent_clicks and not session_interest

    catalog["candidate_reason"] = "candidate_retrieval"
    catalog["matches_recent_click_signal"] = False
    catalog["matches_session_interest"] = False
    signal_score = pd.Series(0.0, index=catalog.index, dtype=float)

    if recent_clicks:
        categories, main_categories, colors = _build_recent_signal_sets(catalog, recent_clicks)
        if categories:
            category_matches = catalog["category"].isin(categories)
            signal_score += category_matches.astype(float) * 3.0
            catalog["matches_recent_click_signal"] |= category_matches
        if main_categories:
            main_category_matches = catalog["main_category"].isin(main_categories)
            signal_score += main_category_matches.astype(float) * 2.0
            catalog["matches_recent_click_signal"] |= main_category_matches
        if colors:
            color_matches = catalog["color"].isin(colors)
            signal_score += color_matches.astype(float) * 1.0
            catalog["matches_recent_click_signal"] |= color_matches

    for category, weight in session_interest.items():
        normalized_weight = float(weight) if weight is not None else 0.0
        category_matches = catalog["category"].eq(str(category))
        main_category_matches = catalog["main_category"].eq(str(category))
        signal_score += category_matches.astype(float) * normalized_weight * 4.0
        signal_score += main_category_matches.astype(float) * normalized_weight * 2.0
        catalog["matches_session_interest"] |= category_matches | main_category_matches

    catalog["candidate_score"] = signal_score + (catalog["popularity"] / max(catalog["popularity"].max(), 1.0))
    if cold_start:
        catalog["candidate_reason"] = "cold_start_popularity"
        catalog["candidate_score"] += (
            catalog.groupby("main_category", dropna=False)["popularity"].rank(method="first", ascending=False).rsub(10).clip(lower=0) / 10.0
        )

    if "item_age_days" in catalog.columns:
        fresh_boost = catalog["item_age_days"].apply(
            lambda value: 0.5 if pd.notna(value) and float(value) <= DEFAULT_NEW_ITEM_WINDOW_DAYS else 0.0
        )
        catalog["candidate_score"] += fresh_boost

    filtered = catalog.loc[~catalog["article_id"].isin(recent_click_set)].copy()
    filtered = filtered.sort_values(
        ["candidate_score", "popularity", "article_id"],
        ascending=[False, False, True],
    ).head(candidate_pool_size)

    if "item_age_days" not in filtered.columns or filtered["item_age_days"].isna().all():
        filtered["item_age_days"] = math.nan
        filtered["is_new_item"] = False

    LOGGER.info(
        "Generated %s candidates for ranking (top_k=%s, cold_start=%s)",
        len(filtered),
        top_k,
        cold_start,
    )
    return filtered.reset_index(drop=True)
