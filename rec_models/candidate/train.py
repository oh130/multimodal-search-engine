"""Training entry point for the Two-Tower candidate retrieval model."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - training requires torch
    torch = None
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]

try:
    from rec_models.candidate.dataset import (
        DEFAULT_DATA_PATH,
        DatasetArtifacts,
        TwoTowerPairDataset,
        build_two_tower_datasets,
        collate_two_tower_batch,
    )
    from rec_models.candidate.model import TwoTowerConfig, TwoTowerModel, build_two_tower_config_from_metadata
    from rec_models.common.metrics import mean_metric, recall_at_k
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from candidate.dataset import (  # type: ignore[no-redef]
        DEFAULT_DATA_PATH,
        DatasetArtifacts,
        TwoTowerPairDataset,
        build_two_tower_datasets,
        collate_two_tower_batch,
    )
    from candidate.model import TwoTowerConfig, TwoTowerModel, build_two_tower_config_from_metadata  # type: ignore[no-redef]
    from common.metrics import mean_metric, recall_at_k  # type: ignore[no-redef]


LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_VALIDATION_K = 300
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "checkpoints" / "candidate"
DEFAULT_MODEL_ARTIFACT = "two_tower.pt"
DEFAULT_METADATA_ARTIFACT = "two_tower_metadata.json"


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise ImportError("torch is required to train the Two-Tower model.")


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters and runtime settings for retrieval training."""

    data_path: str
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    validation_ratio: float = 0.2
    validation_k: int = DEFAULT_VALIDATION_K
    device: str = "cpu"
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    model_artifact: str = DEFAULT_MODEL_ARTIFACT
    metadata_artifact: str = DEFAULT_METADATA_ARTIFACT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Two-Tower candidate retrieval model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Processed interaction CSV for retrieval training.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="AdamW weight decay.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="User-level validation holdout ratio.")
    parser.add_argument("--validation-k", type=int, default=DEFAULT_VALIDATION_K, help="Recall@K cutoff for validation.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Directory for model artifacts.")
    parser.add_argument("--device", type=str, help="Training device override, e.g. cpu or cuda.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


def resolve_device(device_override: str | None = None) -> str:
    _require_torch()
    if device_override:
        return device_override
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloader(dataset: TwoTowerPairDataset, batch_size: int, shuffle: bool) -> DataLoader:
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda rows: collate_two_tower_batch(rows, as_torch=True),
    )


def batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = dict(batch)
    for key in ("user_categorical", "user_numeric", "item_categorical", "item_numeric"):
        moved[key] = moved[key].to(device)
    return moved


def retrieval_loss(logits: Tensor) -> Tensor:
    """In-batch softmax loss with diagonal positives."""

    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def train_one_epoch(
    model: TwoTowerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Run one training epoch and return average loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        prepared = batch_to_device(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            user_categorical=prepared["user_categorical"],
            user_numeric=prepared["user_numeric"],
            item_categorical=prepared["item_categorical"],
            item_numeric=prepared["item_numeric"],
        )
        loss = retrieval_loss(outputs["logits"])
        loss.backward()
        optimizer.step()

        batch_size = prepared["user_categorical"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if total_examples == 0:
        return 0.0
    return total_loss / total_examples


def _stack_encoded_rows(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([row[key] for row in rows]) if rows else np.empty((0, 0), dtype=np.float32)


def build_item_embedding_index(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Encode all known items once for validation retrieval."""

    model.eval()
    merged_records = dict(dataset_artifacts.train_dataset.item_records)
    merged_records.update(dataset_artifacts.validation_dataset.item_records)

    item_ids = list(merged_records.keys())
    encoded_rows = [dataset_artifacts.encoder.encode_item_row(merged_records[item_id]) for item_id in item_ids]
    if not encoded_rows:
        return np.empty((0, dataset_artifacts.metadata["encoder"]["schema"].get("item_numeric_columns", 0))), []

    item_categorical = torch.as_tensor(_stack_encoded_rows(encoded_rows, "categorical"), dtype=torch.long, device=device)
    item_numeric_np = _stack_encoded_rows(encoded_rows, "numeric").astype(np.float32)
    item_numeric = torch.as_tensor(item_numeric_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        embeddings = model.encode_item(item_categorical=item_categorical, item_numeric=item_numeric)
    return embeddings.detach().cpu().numpy().astype(np.float32), item_ids


def evaluate_recall(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    device: str,
    k: int,
) -> float:
    """Compute validation Recall@K against the known item universe."""

    validation_dataset = dataset_artifacts.validation_dataset
    if len(validation_dataset) == 0:
        return 0.0

    item_embeddings, item_ids = build_item_embedding_index(model, dataset_artifacts, device=device)
    if len(item_ids) == 0:
        return 0.0

    item_matrix = torch.as_tensor(item_embeddings, dtype=torch.float32, device=device)
    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    model.eval()
    with torch.no_grad():
        for customer_id, positive_item_ids in validation_dataset.user_to_positive_items.items():
            user_record = validation_dataset.user_records.get(customer_id)
            if user_record is None:
                continue

            encoded_user = dataset_artifacts.encoder.encode_user_row(user_record)
            user_categorical = torch.as_tensor(
                encoded_user["categorical"][None, :],
                dtype=torch.long,
                device=device,
            )
            user_numeric = torch.as_tensor(
                encoded_user["numeric"][None, :],
                dtype=torch.float32,
                device=device,
            )
            user_embedding = model.encode_user(user_categorical=user_categorical, user_numeric=user_numeric)
            scores = torch.matmul(user_embedding, item_matrix.transpose(0, 1)).squeeze(0)
            top_k = min(k, scores.shape[0])
            top_indices = torch.topk(scores, k=top_k, dim=0).indices.detach().cpu().tolist()
            ranked_lists.append([item_ids[index] for index in top_indices])
            relevant_lists.append([str(item_id) for item_id in positive_item_ids])

    return mean_metric(ranked_lists, relevant_lists, recall_at_k, k)


def save_artifacts(
    model: TwoTowerModel,
    dataset_artifacts: DatasetArtifacts,
    training_config: TrainingConfig,
    best_validation_recall: float,
) -> dict[str, Path]:
    """Persist the trained Two-Tower weights and training metadata."""

    checkpoint_dir = Path(training_config.checkpoint_dir).expanduser().resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / training_config.model_artifact
    metadata_path = checkpoint_dir / training_config.metadata_artifact

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(build_two_tower_config_from_metadata(dataset_artifacts.metadata)),
        "dataset_metadata": dataset_artifacts.metadata,
        "training_config": asdict(training_config),
    }
    torch.save(payload, model_path)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_validation_recall": best_validation_recall,
        "training_config": asdict(training_config),
        "dataset_metadata": dataset_artifacts.metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
    }


def train_two_tower(training_config: TrainingConfig) -> dict[str, Any]:
    """Train the retrieval model end-to-end and return summary metrics."""

    _require_torch()
    LOGGER.info("Building Two-Tower datasets from %s", training_config.data_path)
    dataset_artifacts = build_two_tower_datasets(
        csv_path=Path(training_config.data_path),
        validation_ratio=training_config.validation_ratio,
    )
    model_config: TwoTowerConfig = build_two_tower_config_from_metadata(dataset_artifacts.metadata)
    model = TwoTowerModel(model_config).to(training_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_loader = build_dataloader(
        dataset_artifacts.train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
    )

    best_validation_recall = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer=optimizer, device=training_config.device)
        validation_recall = evaluate_recall(
            model,
            dataset_artifacts=dataset_artifacts,
            device=training_config.device,
            k=training_config.validation_k,
        )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                f"validation_recall@{training_config.validation_k}": validation_recall,
            }
        )
        LOGGER.info(
            "Epoch %s/%s | train_loss=%.6f | validation_recall@%s=%.6f",
            epoch,
            training_config.epochs,
            train_loss,
            training_config.validation_k,
            validation_recall,
        )
        best_validation_recall = max(best_validation_recall, validation_recall)

    saved_paths = save_artifacts(
        model=model,
        dataset_artifacts=dataset_artifacts,
        training_config=training_config,
        best_validation_recall=best_validation_recall,
    )
    return {
        "best_validation_recall": best_validation_recall,
        "history": history,
        "saved_paths": {key: str(path) for key, path in saved_paths.items()},
    }


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)
    device = resolve_device(args.device)

    training_config = TrainingConfig(
        data_path=str(args.data.expanduser().resolve()),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_ratio=args.validation_ratio,
        validation_k=args.validation_k,
        device=device,
        checkpoint_dir=str(args.checkpoint_dir.expanduser().resolve()),
    )
    results = train_two_tower(training_config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
