"""Dataset and feature encoding utilities for Two-Tower candidate training.

This module defines a stable training contract for retrieval experiments:

- positive user-item interactions are extracted from processed ranking data
- user/item categorical vocabularies are fitted once and reused
- train/validation splits are reproducible
- batches can be collated into numpy arrays or torch tensors

The current implementation keeps the first version intentionally simple:
- one example == one positive user-item pair
- in-batch negatives are expected to be used during training
- explicit negative sampling helpers are provided for later extensions
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is optional during code-only development
    torch = None

    class Dataset:  # type: ignore[no-redef]
        """Small fallback to keep the module importable without torch."""

        pass


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "candidate_train_data_test.csv.gz"
DEFAULT_ITEM_FEATURES_PATH = BASE_DIR / "data" / "processed" / "item_features.csv"
DEFAULT_ITEM_FEATURES_TEST_PATH = BASE_DIR / "data" / "processed" / "item_features_test.csv"
TARGET_COLUMN = "label"
USER_ID_COLUMN = "customer_id"
ITEM_ID_COLUMN = "article_id"
PADDING_TOKEN = "__PAD__"
UNKNOWN_TOKEN = "__UNK__"

DEFAULT_USER_CATEGORICAL_COLUMNS = (
    "age_bucket",
    "fashion_news_frequency",
    "club_member_status",
    "preferred_garment_group",
    "preferred_colour_master",
    "preferred_main_category",
    "preferred_season",
    "price_band",
    "activity_segment",
)
DEFAULT_USER_NUMERIC_COLUMNS = (
    "age",
    "purchase_count",
    "purchase_count_7d",
    "purchase_count_30d",
    "purchase_count_90d",
    "total_spend",
    "spend_7d",
    "spend_30d",
    "spend_90d",
    "avg_price",
    "min_price",
    "max_price",
    "recency_days",
    "purchase_span_days",
    "online_ratio",
    "offline_ratio",
)

DEFAULT_ITEM_CATEGORICAL_COLUMNS = (
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
    "dominant_age_bucket",
    "dominant_season",
    "item_price_band",
    "popularity_segment",
)
DEFAULT_ITEM_NUMERIC_COLUMNS: tuple[str, ...] = (
    "item_purchase_count",
    "item_purchase_count_7d",
    "item_purchase_count_30d",
    "item_purchase_count_90d",
    "item_total_spend",
    "item_avg_price",
    "item_min_price",
    "item_max_price",
    "item_days_since_last_purchase",
    "item_freshness_days",
    "item_online_ratio",
    "item_offline_ratio",
)
@dataclass(slots=True, frozen=True)
class FeatureSchema:
    """Column contract shared by dataset, model, and inference code."""

    user_id_column: str = USER_ID_COLUMN
    item_id_column: str = ITEM_ID_COLUMN
    target_column: str = TARGET_COLUMN
    user_categorical_columns: tuple[str, ...] = DEFAULT_USER_CATEGORICAL_COLUMNS
    user_numeric_columns: tuple[str, ...] = DEFAULT_USER_NUMERIC_COLUMNS
    item_categorical_columns: tuple[str, ...] = DEFAULT_ITEM_CATEGORICAL_COLUMNS
    item_numeric_columns: tuple[str, ...] = DEFAULT_ITEM_NUMERIC_COLUMNS


@dataclass(slots=True)
class Vocabulary:
    """Simple categorical vocabulary with stable special tokens."""

    token_to_index: dict[str, int]
    index_to_token: list[str]

    @classmethod
    def build(cls, values: Iterable[Any]) -> Vocabulary:
        tokens = [PADDING_TOKEN, UNKNOWN_TOKEN]
        seen = {PADDING_TOKEN, UNKNOWN_TOKEN}
        for value in values:
            normalized = normalize_categorical_value(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(normalized)
        return cls(
            token_to_index={token: index for index, token in enumerate(tokens)},
            index_to_token=tokens,
        )

    def encode(self, value: Any) -> int:
        normalized = normalize_categorical_value(value)
        return self.token_to_index.get(normalized, self.token_to_index[UNKNOWN_TOKEN])

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_to_index": self.token_to_index,
            "index_to_token": self.index_to_token,
        }


@dataclass(slots=True)
class TwoTowerFeatureEncoder:
    """Encodes user/item rows into indexable categorical and numeric arrays."""

    schema: FeatureSchema = field(default_factory=FeatureSchema)
    user_vocabularies: dict[str, Vocabulary] = field(default_factory=dict)
    item_vocabularies: dict[str, Vocabulary] = field(default_factory=dict)

    def fit(self, data: pd.DataFrame) -> TwoTowerFeatureEncoder:
        """Fit per-column vocabularies on the provided interaction data."""

        user_table, item_table = build_entity_tables(data, schema=self.schema)

        self.user_vocabularies = {
            column: Vocabulary.build(user_table[column].tolist())
            for column in self.schema.user_categorical_columns
        }
        self.item_vocabularies = {
            column: Vocabulary.build(item_table[column].tolist())
            for column in self.schema.item_categorical_columns
        }
        LOGGER.info(
            "Fitted Two-Tower feature encoder | user_vocab_columns=%s item_vocab_columns=%s users=%s items=%s",
            len(self.user_vocabularies),
            len(self.item_vocabularies),
            len(user_table),
            len(item_table),
        )
        return self

    def encode_user_row(self, row: pd.Series | dict[str, Any]) -> dict[str, np.ndarray]:
        row_mapping = row if isinstance(row, dict) else row.to_dict()
        categorical = np.asarray(
            [self.user_vocabularies[column].encode(row_mapping.get(column)) for column in self.schema.user_categorical_columns],
            dtype=np.int64,
        )
        numeric = np.asarray(
            [normalize_numeric_value(row_mapping.get(column)) for column in self.schema.user_numeric_columns],
            dtype=np.float32,
        )
        return {
            "categorical": categorical,
            "numeric": numeric,
        }

    def encode_item_row(self, row: pd.Series | dict[str, Any]) -> dict[str, np.ndarray]:
        row_mapping = row if isinstance(row, dict) else row.to_dict()
        categorical = np.asarray(
            [self.item_vocabularies[column].encode(row_mapping.get(column)) for column in self.schema.item_categorical_columns],
            dtype=np.int64,
        )
        numeric = np.asarray(
            [normalize_numeric_value(row_mapping.get(column)) for column in self.schema.item_numeric_columns],
            dtype=np.float32,
        )
        return {
            "categorical": categorical,
            "numeric": numeric,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": asdict(self.schema),
            "user_vocabularies": {column: vocab.to_dict() for column, vocab in self.user_vocabularies.items()},
            "item_vocabularies": {column: vocab.to_dict() for column, vocab in self.item_vocabularies.items()},
        }


@dataclass(slots=True)
class DatasetArtifacts:
    """Pre-split data and encoder required for Two-Tower training."""

    train_dataset: TwoTowerPairDataset
    validation_dataset: TwoTowerPairDataset
    encoder: TwoTowerFeatureEncoder
    metadata: dict[str, Any]


def normalize_categorical_value(value: Any) -> str:
    """Normalize text-like values into a stable vocabulary token."""

    if value is None:
        return UNKNOWN_TOKEN
    text = str(value).strip()
    return text if text else UNKNOWN_TOKEN


def normalize_numeric_value(value: Any, default: float = 0.0) -> float:
    """Normalize numeric values for dense features."""

    if isinstance(value, bool):
        return float(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y"}:
            return 1.0
        if normalized in {"false", "no", "n"}:
            return 0.0

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


def normalize_item_id(value: Any) -> str:
    """Normalize article ids to the zero-padded serving format."""

    text = str(value).strip()
    if text.isdigit():
        return text.zfill(10)
    return text


def _resolve_item_feature_path() -> Path | None:
    for path in (DEFAULT_ITEM_FEATURES_PATH, DEFAULT_ITEM_FEATURES_TEST_PATH):
        if path.exists():
            return path
    return None


def enrich_with_item_features(
    data: pd.DataFrame,
    item_feature_path: Path | None = None,
    numeric_columns: Sequence[str] = DEFAULT_ITEM_NUMERIC_COLUMNS,
) -> pd.DataFrame:
    """Attach item-level numeric features used by retrieval experiments.

    The processed ranking dataset does not currently contain serving-time item
    aggregates such as popularity or freshness. This helper merges those
    features from `item_features(.csv)` when available so retrieval training can
    use the same item signals already exposed elsewhere in the project.
    """

    missing_numeric_columns = [column for column in numeric_columns if column not in data.columns]
    if not missing_numeric_columns:
        return data

    resolved_item_feature_path = item_feature_path or _resolve_item_feature_path()
    if resolved_item_feature_path is None:
        LOGGER.warning(
            "Item feature file not found. Continuing without additional item numeric features."
        )
        enriched = data.copy()
        for column in missing_numeric_columns:
            if column not in enriched.columns:
                enriched[column] = 0.0
        return enriched

    item_features = pd.read_csv(resolved_item_feature_path, dtype={"article_id": str}).fillna("")
    item_features["article_id"] = item_features["article_id"].map(normalize_item_id)

    available_numeric_columns = [column for column in missing_numeric_columns if column in item_features.columns]
    if not available_numeric_columns:
        enriched = data.copy()
        for column in missing_numeric_columns:
            if column not in enriched.columns:
                enriched[column] = 0.0
        return enriched

    merge_columns = ["article_id", *available_numeric_columns]
    enriched = data.merge(
        item_features.loc[:, merge_columns].drop_duplicates("article_id"),
        on="article_id",
        how="left",
        suffixes=("", "_item_feature"),
    )
    for column in missing_numeric_columns:
        if column not in enriched.columns:
            enriched[column] = 0.0

    LOGGER.info(
        "Merged item numeric features from %s | columns=%s",
        resolved_item_feature_path,
        available_numeric_columns,
    )
    return enriched


def load_candidate_training_data(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load processed interaction data for candidate training."""

    resolved_path = csv_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Candidate training data not found: {resolved_path}")

    data = pd.read_csv(resolved_path)
    required_columns = {USER_ID_COLUMN, ITEM_ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Candidate training data is missing required columns: {sorted(missing_columns)}")

    data[USER_ID_COLUMN] = data[USER_ID_COLUMN].astype(str)
    data[ITEM_ID_COLUMN] = data[ITEM_ID_COLUMN].map(normalize_item_id)
    enriched = enrich_with_item_features(data)
    LOGGER.info("Loaded candidate training data from %s | rows=%s", resolved_path, len(enriched))
    return enriched


def filter_positive_interactions(data: pd.DataFrame, schema: FeatureSchema | None = None) -> pd.DataFrame:
    """Keep one row per positive user-item interaction."""

    resolved_schema = schema or FeatureSchema()
    positive_mask = pd.to_numeric(data[resolved_schema.target_column], errors="coerce").fillna(0).eq(1)
    positives = data.loc[positive_mask].copy()
    positives = positives.drop_duplicates([resolved_schema.user_id_column, resolved_schema.item_id_column], keep="first")
    if positives.empty:
        raise ValueError("No positive interactions found for Two-Tower training.")
    LOGGER.info("Filtered positive interactions | rows=%s", len(positives))
    return positives.reset_index(drop=True)


def build_entity_tables(data: pd.DataFrame, schema: FeatureSchema | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deduplicated user and item feature tables from interaction rows."""

    resolved_schema = schema or FeatureSchema()
    user_columns = [resolved_schema.user_id_column, *resolved_schema.user_categorical_columns, *resolved_schema.user_numeric_columns]
    item_columns = [resolved_schema.item_id_column, *resolved_schema.item_categorical_columns, *resolved_schema.item_numeric_columns]

    user_table = (
        data.loc[:, [column for column in user_columns if column in data.columns]]
        .drop_duplicates(subset=[resolved_schema.user_id_column], keep="first")
        .reset_index(drop=True)
    )
    item_table = (
        data.loc[:, [column for column in item_columns if column in data.columns]]
        .drop_duplicates(subset=[resolved_schema.item_id_column], keep="first")
        .reset_index(drop=True)
    )

    for column in resolved_schema.user_categorical_columns:
        if column not in user_table.columns:
            user_table[column] = UNKNOWN_TOKEN
    for column in resolved_schema.user_numeric_columns:
        if column not in user_table.columns:
            user_table[column] = 0.0
    for column in resolved_schema.item_categorical_columns:
        if column not in item_table.columns:
            item_table[column] = UNKNOWN_TOKEN
    for column in resolved_schema.item_numeric_columns:
        if column not in item_table.columns:
            item_table[column] = 0.0

    return user_table, item_table


def stable_holdout_split(
    positive_interactions: pd.DataFrame,
    validation_ratio: float = 0.2,
    schema: FeatureSchema | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible user-level train/validation split.

    Users are hashed into buckets so the same user consistently lands in the
    same split without relying on pandas or sklearn randomness.
    """

    resolved_schema = schema or FeatureSchema()
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")

    user_ids = positive_interactions[resolved_schema.user_id_column].astype(str).unique().tolist()
    threshold = int(validation_ratio * 10_000)

    validation_users = {
        user_id
        for user_id in user_ids
        if int(hashlib.blake2b(user_id.encode("utf-8"), digest_size=4).hexdigest(), 16) % 10_000 < threshold
    }
    if not validation_users:
        validation_users = set(user_ids[: max(1, int(len(user_ids) * validation_ratio))])

    validation = positive_interactions.loc[
        positive_interactions[resolved_schema.user_id_column].astype(str).isin(validation_users)
    ].copy()
    train = positive_interactions.loc[
        ~positive_interactions[resolved_schema.user_id_column].astype(str).isin(validation_users)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            "Stable holdout split produced an empty train or validation split. "
            "Use more data or adjust validation_ratio."
        )
    LOGGER.info(
        "Created stable holdout split | train_rows=%s validation_rows=%s train_users=%s validation_users=%s",
        len(train),
        len(validation),
        train[resolved_schema.user_id_column].nunique(),
        validation[resolved_schema.user_id_column].nunique(),
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


class TwoTowerPairDataset(Dataset):
    """Dataset of positive user-item pairs for retrieval training."""

    def __init__(
        self,
        interactions: pd.DataFrame,
        encoder: TwoTowerFeatureEncoder,
        schema: FeatureSchema | None = None,
    ) -> None:
        self.schema = schema or encoder.schema
        self.encoder = encoder
        self.interactions = interactions.reset_index(drop=True).copy()
        self.interactions[self.schema.user_id_column] = self.interactions[self.schema.user_id_column].astype(str)
        self.interactions[self.schema.item_id_column] = self.interactions[self.schema.item_id_column].map(normalize_item_id)

        user_table, item_table = build_entity_tables(self.interactions, schema=self.schema)
        self.user_records = user_table.set_index(self.schema.user_id_column, drop=False).to_dict(orient="index")
        self.item_records = item_table.set_index(self.schema.item_id_column, drop=False).to_dict(orient="index")
        self.user_to_positive_items = (
            self.interactions.groupby(self.schema.user_id_column, sort=False)[self.schema.item_id_column].apply(list).to_dict()
        )
        self.all_item_ids = tuple(self.item_records.keys())
        self.item_to_main_category = {
            str(item_id): normalize_categorical_value(record.get("main_category"))
            for item_id, record in self.item_records.items()
        }
        self.category_to_item_ids = self._build_category_to_item_ids()

    def __len__(self) -> int:
        return len(self.interactions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.interactions.iloc[index]
        user_id = str(row[self.schema.user_id_column])
        item_id = str(row[self.schema.item_id_column])
        user_record = self.user_records[user_id]
        item_record = self.item_records[item_id]
        user_features = self.encoder.encode_user_row(user_record)
        item_features = self.encoder.encode_item_row(item_record)

        return {
            "customer_id": user_id,
            "article_id": item_id,
            "user_categorical": user_features["categorical"],
            "user_numeric": user_features["numeric"],
            "item_categorical": item_features["categorical"],
            "item_numeric": item_features["numeric"],
        }

    def _build_category_to_item_ids(self) -> dict[str, tuple[str, ...]]:
        category_to_ids: dict[str, list[str]] = {}
        for item_id, category in self.item_to_main_category.items():
            category_to_ids.setdefault(category, []).append(str(item_id))
        return {category: tuple(item_ids) for category, item_ids in category_to_ids.items()}

    def sample_negative_item_ids(
        self,
        customer_id: str,
        sample_size: int,
        positive_item_id: str | None = None,
        hard_negative_ratio: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Sample negative items not seen as positives for the given user."""

        if sample_size <= 0:
            return []

        rng = rng or np.random.default_rng()
        positive_items = set(self.user_to_positive_items.get(str(customer_id), []))
        hard_sample_size = min(sample_size, max(0, int(round(sample_size * hard_negative_ratio))))
        sampled_negatives: list[str] = []

        if positive_item_id is not None and hard_sample_size > 0:
            main_category = self.item_to_main_category.get(str(positive_item_id), UNKNOWN_TOKEN)
            hard_candidates = [
                item_id
                for item_id in self.category_to_item_ids.get(main_category, ())
                if item_id not in positive_items and item_id != str(positive_item_id)
            ]
            if hard_candidates:
                hard_count = min(hard_sample_size, len(hard_candidates))
                hard_sample = rng.choice(hard_candidates, size=hard_count, replace=False)
                sampled_negatives.extend(str(item_id) for item_id in hard_sample.tolist())

        remaining_sample_size = max(0, sample_size - len(sampled_negatives))
        if remaining_sample_size <= 0:
            return sampled_negatives

        candidates = [
            item_id
            for item_id in self.all_item_ids
            if item_id not in positive_items and item_id not in sampled_negatives
        ]
        if not candidates:
            return sampled_negatives
        sample_count = min(remaining_sample_size, len(candidates))
        sampled = rng.choice(candidates, size=sample_count, replace=False)
        sampled_negatives.extend(str(item_id) for item_id in sampled.tolist())
        return sampled_negatives

    def encode_item_id_batch(self, item_ids: Sequence[str]) -> dict[str, np.ndarray]:
        """Encode many item ids into stacked categorical/numeric arrays."""

        encoded_rows = [self.encoder.encode_item_row(self.item_records[str(item_id)]) for item_id in item_ids]
        if not encoded_rows:
            return {
                "categorical": np.empty((0, len(self.schema.item_categorical_columns)), dtype=np.int64),
                "numeric": np.empty((0, len(self.schema.item_numeric_columns)), dtype=np.float32),
            }
        return {
            "categorical": np.stack([row["categorical"] for row in encoded_rows]).astype(np.int64),
            "numeric": np.stack([row["numeric"] for row in encoded_rows]).astype(np.float32),
        }


def collate_two_tower_batch(examples: Sequence[dict[str, Any]], as_torch: bool = False) -> dict[str, Any]:
    """Collate dataset samples into a batch for training or debugging."""

    if not examples:
        raise ValueError("Cannot collate an empty Two-Tower batch.")

    batch = {
        "customer_id": [example["customer_id"] for example in examples],
        "article_id": [example["article_id"] for example in examples],
        "user_categorical": np.stack([example["user_categorical"] for example in examples]).astype(np.int64),
        "user_numeric": np.stack([example["user_numeric"] for example in examples]).astype(np.float32),
        "item_categorical": np.stack([example["item_categorical"] for example in examples]).astype(np.int64),
        "item_numeric": np.stack([example["item_numeric"] for example in examples]).astype(np.float32),
    }
    if as_torch:
        if torch is None:
            raise ImportError("torch is not installed. Install torch before requesting tensor collation.")
        return {
            **batch,
            "user_categorical": torch.as_tensor(batch["user_categorical"], dtype=torch.long),
            "user_numeric": torch.as_tensor(batch["user_numeric"], dtype=torch.float32),
            "item_categorical": torch.as_tensor(batch["item_categorical"], dtype=torch.long),
            "item_numeric": torch.as_tensor(batch["item_numeric"], dtype=torch.float32),
        }
    return batch


def build_two_tower_datasets(
    csv_path: Path = DEFAULT_DATA_PATH,
    validation_ratio: float = 0.2,
    schema: FeatureSchema | None = None,
) -> DatasetArtifacts:
    """Build train/validation datasets and the fitted encoder in one call."""

    resolved_schema = schema or FeatureSchema()
    raw_data = load_candidate_training_data(csv_path)
    positives = filter_positive_interactions(raw_data, schema=resolved_schema)
    train_data, validation_data = stable_holdout_split(
        positives,
        validation_ratio=validation_ratio,
        schema=resolved_schema,
    )

    encoder = TwoTowerFeatureEncoder(schema=resolved_schema).fit(train_data)
    train_dataset = TwoTowerPairDataset(train_data, encoder=encoder, schema=resolved_schema)
    validation_dataset = TwoTowerPairDataset(validation_data, encoder=encoder, schema=resolved_schema)

    metadata = {
        "data_path": str(csv_path.expanduser().resolve()),
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "train_users": train_data[resolved_schema.user_id_column].nunique(),
        "validation_users": validation_data[resolved_schema.user_id_column].nunique(),
        "train_items": train_data[resolved_schema.item_id_column].nunique(),
        "validation_items": validation_data[resolved_schema.item_id_column].nunique(),
        "encoder": encoder.metadata(),
    }
    return DatasetArtifacts(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        encoder=encoder,
        metadata=metadata,
    )


def save_dataset_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Persist dataset/encoder metadata for reproducible training."""

    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
