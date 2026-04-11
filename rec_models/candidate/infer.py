"""Inference utilities for the Two-Tower candidate retrieval model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - inference requires torch
    torch = None

try:
    from rec_models.candidate.dataset import (
        DEFAULT_DATA_PATH,
        DEFAULT_ITEM_CATEGORICAL_COLUMNS,
        DEFAULT_ITEM_NUMERIC_COLUMNS,
        DEFAULT_USER_CATEGORICAL_COLUMNS,
        DEFAULT_USER_NUMERIC_COLUMNS,
        FeatureSchema,
        TwoTowerFeatureEncoder,
        Vocabulary,
        build_entity_tables,
        load_candidate_training_data,
        normalize_item_id,
    )
    from rec_models.candidate.model import TowerConfig, TwoTowerConfig, TwoTowerModel
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.dataset import (  # type: ignore[no-redef]
        DEFAULT_DATA_PATH,
        DEFAULT_ITEM_CATEGORICAL_COLUMNS,
        DEFAULT_ITEM_NUMERIC_COLUMNS,
        DEFAULT_USER_CATEGORICAL_COLUMNS,
        DEFAULT_USER_NUMERIC_COLUMNS,
        FeatureSchema,
        TwoTowerFeatureEncoder,
        Vocabulary,
        build_entity_tables,
        load_candidate_training_data,
        normalize_item_id,
    )
    from candidate.model import TowerConfig, TwoTowerConfig, TwoTowerModel  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "candidate"
DEFAULT_MODEL_ARTIFACT = "two_tower.pt"
DEFAULT_TOP_K = 300


def _require_torch() -> None:
    if torch is None:
        raise ImportError("torch is required for Two-Tower inference.")


def _tower_config_from_dict(raw: dict[str, Any]) -> Any:
    return TowerConfig(
        categorical_cardinalities=list(raw.get("categorical_cardinalities", [])),
        numeric_dim=int(raw.get("numeric_dim", 0)),
        embedding_dim=int(raw.get("embedding_dim", 32)),
        hidden_dims=tuple(raw.get("hidden_dims", (128, 64))),
        dropout=float(raw.get("dropout", 0.1)),
    )


def _rebuild_encoder(encoder_metadata: dict[str, Any]) -> TwoTowerFeatureEncoder:
    schema_dict = encoder_metadata.get("schema", {})
    schema = FeatureSchema(
        user_id_column=schema_dict.get("user_id_column", "customer_id"),
        item_id_column=schema_dict.get("item_id_column", "article_id"),
        target_column=schema_dict.get("target_column", "label"),
        user_categorical_columns=tuple(schema_dict.get("user_categorical_columns", DEFAULT_USER_CATEGORICAL_COLUMNS)),
        user_numeric_columns=tuple(schema_dict.get("user_numeric_columns", DEFAULT_USER_NUMERIC_COLUMNS)),
        item_categorical_columns=tuple(schema_dict.get("item_categorical_columns", DEFAULT_ITEM_CATEGORICAL_COLUMNS)),
        item_numeric_columns=tuple(schema_dict.get("item_numeric_columns", DEFAULT_ITEM_NUMERIC_COLUMNS)),
    )

    encoder = TwoTowerFeatureEncoder(schema=schema)
    encoder.user_vocabularies = {
        column: Vocabulary(
            token_to_index={str(token): int(index) for token, index in vocab["token_to_index"].items()},
            index_to_token=[str(token) for token in vocab["index_to_token"]],
        )
        for column, vocab in encoder_metadata.get("user_vocabularies", {}).items()
    }
    encoder.item_vocabularies = {
        column: Vocabulary(
            token_to_index={str(token): int(index) for token, index in vocab["token_to_index"].items()},
            index_to_token=[str(token) for token in vocab["index_to_token"]],
        )
        for column, vocab in encoder_metadata.get("item_vocabularies", {}).items()
    }
    return encoder


def load_two_tower_artifacts(
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    model_artifact: str = DEFAULT_MODEL_ARTIFACT,
    map_location: str = "cpu",
) -> tuple[TwoTowerModel, TwoTowerFeatureEncoder, dict[str, Any]]:
    """Load the trained model, encoder, and saved metadata."""

    _require_torch()
    resolved_dir = checkpoint_dir.expanduser().resolve()
    model_path = resolved_dir / model_artifact
    if not model_path.exists():
        raise FileNotFoundError(f"Two-Tower model artifact not found: {model_path}")

    payload = torch.load(model_path, map_location=map_location)
    model_config_dict = payload.get("model_config", {})
    config = TwoTowerConfig(
        user_tower=_tower_config_from_dict(model_config_dict.get("user_tower", {})),
        item_tower=_tower_config_from_dict(model_config_dict.get("item_tower", {})),
        output_dim=int(model_config_dict.get("output_dim", 64)),
        l2_normalize=bool(model_config_dict.get("l2_normalize", True)),
    )
    model = TwoTowerModel(config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    dataset_metadata = payload.get("dataset_metadata", {})
    encoder = _rebuild_encoder(dataset_metadata.get("encoder", {}))
    return model, encoder, dataset_metadata


def _to_tensor(array: np.ndarray, dtype: Any, device: str) -> Any:
    _require_torch()
    return torch.as_tensor(array, dtype=dtype, device=device)


def encode_user_records(
    model: TwoTowerModel,
    encoder: TwoTowerFeatureEncoder,
    user_records: list[dict[str, Any]],
    device: str = "cpu",
) -> np.ndarray:
    """Encode a batch of user records into retrieval embeddings."""

    if not user_records:
        return np.empty((0, model.config.output_dim), dtype=np.float32)

    encoded = [encoder.encode_user_row(record) for record in user_records]
    categorical = _to_tensor(np.stack([row["categorical"] for row in encoded]), torch.long, device)
    numeric = _to_tensor(np.stack([row["numeric"] for row in encoded]).astype(np.float32), torch.float32, device)

    with torch.no_grad():
        embeddings = model.encode_user(categorical, numeric)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def encode_item_records(
    model: TwoTowerModel,
    encoder: TwoTowerFeatureEncoder,
    item_records: list[dict[str, Any]],
    device: str = "cpu",
) -> np.ndarray:
    """Encode a batch of item records into retrieval embeddings."""

    if not item_records:
        return np.empty((0, model.config.output_dim), dtype=np.float32)

    encoded = [encoder.encode_item_row(record) for record in item_records]
    categorical = _to_tensor(np.stack([row["categorical"] for row in encoded]), torch.long, device)
    numeric_np = np.stack([row["numeric"] for row in encoded]).astype(np.float32) if encoded[0]["numeric"].size else np.zeros((len(encoded), 0), dtype=np.float32)
    numeric = _to_tensor(numeric_np, torch.float32, device)

    with torch.no_grad():
        embeddings = model.encode_item(categorical, numeric)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def build_item_embedding_index(
    items: pd.DataFrame,
    model: TwoTowerModel,
    encoder: TwoTowerFeatureEncoder,
    schema: FeatureSchema | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Build a dense item embedding matrix for retrieval."""

    resolved_schema = schema or encoder.schema
    _, item_table = build_entity_tables(items, schema=resolved_schema)
    item_records = item_table.to_dict(orient="records")
    item_ids = [normalize_item_id(record[resolved_schema.item_id_column]) for record in item_records]
    embeddings = encode_item_records(model, encoder, item_records, device=device)
    return embeddings, item_ids, item_records


def retrieve_top_k(
    user_embedding: np.ndarray,
    item_embeddings: np.ndarray,
    item_ids: Sequence[str],
    top_k: int,
    exclude_item_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the highest-scoring items for one user embedding."""

    if item_embeddings.size == 0 or top_k <= 0:
        return []

    exclude_item_ids = exclude_item_ids or set()
    scores = np.matmul(item_embeddings, user_embedding.astype(np.float32))
    ranking = np.argsort(-scores)

    results: list[dict[str, Any]] = []
    for index in ranking:
        item_id = str(item_ids[index])
        if item_id in exclude_item_ids:
            continue
        results.append(
            {
                "article_id": item_id,
                "score": float(scores[index]),
            }
        )
        if len(results) >= top_k:
            break
    return results


def retrieve_candidates_for_users(
    user_rows: pd.DataFrame,
    items: pd.DataFrame,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    top_k: int = DEFAULT_TOP_K,
    model_artifact: str = DEFAULT_MODEL_ARTIFACT,
    device: str = "cpu",
    exclude_seen_items: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Run retrieval for each distinct user in the provided frame."""

    model, encoder, dataset_metadata = load_two_tower_artifacts(
        checkpoint_dir=checkpoint_dir,
        model_artifact=model_artifact,
        map_location=device,
    )
    schema_dict = dataset_metadata.get("encoder", {}).get("schema", {})
    schema = FeatureSchema(
        user_id_column=schema_dict.get("user_id_column", "customer_id"),
        item_id_column=schema_dict.get("item_id_column", "article_id"),
        target_column=schema_dict.get("target_column", "label"),
        user_categorical_columns=tuple(schema_dict.get("user_categorical_columns", DEFAULT_USER_CATEGORICAL_COLUMNS)),
        user_numeric_columns=tuple(schema_dict.get("user_numeric_columns", DEFAULT_USER_NUMERIC_COLUMNS)),
        item_categorical_columns=tuple(schema_dict.get("item_categorical_columns", DEFAULT_ITEM_CATEGORICAL_COLUMNS)),
        item_numeric_columns=tuple(schema_dict.get("item_numeric_columns", DEFAULT_ITEM_NUMERIC_COLUMNS)),
    )

    item_embeddings, item_ids, _ = build_item_embedding_index(
        items=items,
        model=model,
        encoder=encoder,
        schema=schema,
        device=device,
    )
    user_table, _ = build_entity_tables(user_rows, schema=schema)
    user_records = user_table.to_dict(orient="records")
    user_embeddings = encode_user_records(model, encoder, user_records, device=device)

    results: dict[str, list[dict[str, Any]]] = {}
    positives_by_user = (
        user_rows.groupby(schema.user_id_column, sort=False)[schema.item_id_column]
        .apply(lambda values: {normalize_item_id(value) for value in values})
        .to_dict()
    )
    for user_record, user_embedding in zip(user_records, user_embeddings, strict=False):
        user_id = str(user_record[schema.user_id_column])
        results[user_id] = retrieve_top_k(
            user_embedding=user_embedding,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            top_k=top_k,
            exclude_item_ids=positives_by_user.get(user_id, set()) if exclude_seen_items else None,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve top-k candidates using a trained Two-Tower model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Input interaction data used to source user/item features.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory containing two_tower.pt.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Retrieval cutoff K.")
    parser.add_argument("--user-id", type=str, help="Optional single user id to inspect.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_candidate_training_data(args.data)
    if args.user_id is not None:
        user_rows = data.loc[data["customer_id"].astype(str).eq(args.user_id)].copy()
        if user_rows.empty:
            raise ValueError(f"user_id not found in input data: {args.user_id}")
    else:
        user_rows = data.drop_duplicates("customer_id", keep="first").head(10).copy()

    results = retrieve_candidates_for_users(
        user_rows=user_rows,
        items=data,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
        device=args.device,
    )

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved retrieval results to {output_path}")
        return

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
