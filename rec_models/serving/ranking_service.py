"""Service-side ranking inference for recommendation candidates."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

try:
    from rec_models.ranking.infer import (
        DEFAULT_CHECKPOINT_DIR,
        _extract_scores,
        load_artifacts,
        prepare_inference_features,
    )
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from ranking.infer import (  # type: ignore[no-redef]
        DEFAULT_CHECKPOINT_DIR,
        _extract_scores,
        load_artifacts,
        prepare_inference_features,
    )


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CUSTOMER_FEATURES_PATH = BASE_DIR / "data" / "processed" / "customer_features.csv"
LEAKAGE_COLUMNS = {"label", "customer_id", "article_id", "price", "sales_channel_id"}
DEFAULT_NUMERIC_VALUE = np.nan
DEFAULT_CATEGORICAL_VALUE = "UNKNOWN"


def cast_numeric_features_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility shim for ranking pipeline artifacts saved from train.py."""

    return frame.astype("float64")


def _register_legacy_joblib_symbols() -> None:
    """Expose legacy symbols expected by persisted sklearn transformers."""

    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "cast_numeric_features_to_float"):
        setattr(main_module, "cast_numeric_features_to_float", cast_numeric_features_to_float)


@lru_cache(maxsize=1)
def load_ranking_pipeline(checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> tuple[Pipeline, dict[str, Any]]:
    """Load and cache the persisted ranking pipeline and metadata."""

    _register_legacy_joblib_symbols()
    model, metadata = load_artifacts(checkpoint_dir=checkpoint_dir)
    return model, metadata


@lru_cache(maxsize=1)
def load_customer_features() -> pd.DataFrame:
    """Load and cache customer profile features for ranking."""

    if not CUSTOMER_FEATURES_PATH.exists():
        LOGGER.warning("Customer feature file not found: %s", CUSTOMER_FEATURES_PATH)
        return pd.DataFrame(columns=["customer_id", "age", "age_bucket", "fashion_news_frequency", "club_member_status"])

    customer_features = pd.read_csv(CUSTOMER_FEATURES_PATH, dtype=str).fillna(DEFAULT_CATEGORICAL_VALUE)
    customer_features["customer_id"] = customer_features["customer_id"].astype(str)
    customer_features["age"] = pd.to_numeric(customer_features.get("age"), errors="coerce")
    return customer_features.set_index("customer_id", drop=False)


def _safe_get_text(value: Any) -> str:
    if value is None:
        return DEFAULT_CATEGORICAL_VALUE
    text = str(value).strip()
    return text if text else DEFAULT_CATEGORICAL_VALUE


def _resolve_user_features(user_id: str) -> dict[str, Any]:
    """Return serving-time user features aligned to the ranking contract."""

    customer_features = load_customer_features()
    if user_id in customer_features.index:
        record = customer_features.loc[user_id].to_dict()
        return {
            "customer_id": str(record.get("customer_id", user_id)),
            "age": pd.to_numeric(record.get("age"), errors="coerce"),
            "age_bucket": _safe_get_text(record.get("age_bucket")),
            "fashion_news_frequency": _safe_get_text(record.get("fashion_news_frequency")),
            "club_member_status": _safe_get_text(record.get("club_member_status")),
        }

    LOGGER.info("Customer features not found for user_id=%s. Using cold-start defaults.", user_id)
    return {
        "customer_id": user_id,
        "age": DEFAULT_NUMERIC_VALUE,
        "age_bucket": DEFAULT_CATEGORICAL_VALUE,
        "fashion_news_frequency": DEFAULT_CATEGORICAL_VALUE,
        "club_member_status": DEFAULT_CATEGORICAL_VALUE,
    }


@lru_cache(maxsize=1)
def get_ranking_feature_columns() -> tuple[str, ...]:
    """Cache serving-time ranking feature columns derived from model metadata."""

    _, metadata = load_ranking_pipeline()
    feature_columns = tuple(column for column in metadata.get("feature_columns", []) if column not in LEAKAGE_COLUMNS)
    if not feature_columns:
        raise ValueError("Ranking metadata does not contain usable feature_columns.")
    return feature_columns


def build_ranking_features(
    user_id: str,
    candidate_items: pd.DataFrame,
    session_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build serving-time ranking features for candidate items."""

    del session_context

    if candidate_items.empty:
        return pd.DataFrame()

    feature_columns = list(get_ranking_feature_columns())
    user_features = _resolve_user_features(user_id)
    age_bucket = _safe_get_text(user_features.get("age_bucket"))
    club_member_status = _safe_get_text(user_features.get("club_member_status"))
    fashion_news_frequency = _safe_get_text(user_features.get("fashion_news_frequency"))

    safe_text_frame = candidate_items.reindex(
        columns=[
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
        ],
        fill_value=DEFAULT_CATEGORICAL_VALUE,
    ).copy()
    for column in safe_text_frame.columns:
        safe_text_frame[column] = safe_text_frame[column].map(_safe_get_text)

    features = pd.DataFrame(index=candidate_items.index)
    features["age"] = user_features.get("age", DEFAULT_NUMERIC_VALUE)
    features["age_bucket"] = age_bucket
    features["fashion_news_frequency"] = fashion_news_frequency
    features["club_member_status"] = club_member_status
    for column in safe_text_frame.columns:
        features[column] = safe_text_frame[column]

    features["age_category"] = age_bucket + "_" + features["category"]
    features["age_color"] = age_bucket + "_" + features["color"]
    features["member_category"] = club_member_status + "_" + features["category"]
    features["fashion_category"] = fashion_news_frequency + "_" + features["category"]
    return prepare_inference_features(features, feature_columns=feature_columns)


def score_candidates(
    user_id: str,
    candidate_items: pd.DataFrame,
    session_context: dict[str, Any] | None = None,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
) -> pd.DataFrame:
    """Score ranking candidates and preserve recommendation metadata."""

    if candidate_items.empty:
        return candidate_items.copy()

    model, _ = load_ranking_pipeline(checkpoint_dir=checkpoint_dir)
    ranking_features = build_ranking_features(
        user_id=user_id,
        candidate_items=candidate_items,
        session_context=session_context,
    )
    scores = _extract_scores(model=model, features=ranking_features)

    result = candidate_items.copy()
    result["score"] = scores
    cold_start_mask = result.get("candidate_reason", pd.Series(index=result.index, dtype=object)).eq("cold_start_popularity")
    session_interest_mask = result.get("matches_session_interest", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    recent_click_mask = result.get("matches_recent_click_signal", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    result["reason"] = np.select(
        [cold_start_mask, session_interest_mask, recent_click_mask],
        ["cold_start_popularity", "session_interest_match", "recent_click_similarity"],
        default="ranking_score",
    )
    result["is_exploration"] = False
    return result
