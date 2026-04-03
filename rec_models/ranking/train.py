"""Baseline ranking model training pipeline.

This module trains a maintainable Stage-2 ranking baseline from a tabular CSV.
It keeps preprocessing and model inference bundled in a single scikit-learn
pipeline so later serving code can load one artifact and score candidates with a
stable feature contract.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder


LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "label"
IDENTIFIER_COLUMNS = ("customer_id", "article_id")
LEAKAGE_COLUMNS = {
    "label",
    "customer_id",
    "article_id",
    "price",
    "sales_channel_id",
}
DEFAULT_VALIDATION_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
PIPELINE_ARTIFACT_NAME = "ranking_baseline.joblib"
METADATA_ARTIFACT_NAME = "ranking_baseline_metadata.json"


@dataclass(slots=True)
class TrainingArtifacts:
    """Metadata persisted alongside the trained ranking pipeline."""

    target_column: str
    identifier_columns: list[str]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    model_artifact: str
    created_at_utc: str
    validation_size: float
    random_state: int


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for CLI execution."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_training_data(csv_path: Path) -> pd.DataFrame:
    """Load ranking training data from CSV.

    Args:
        csv_path: Path to the CSV file used for baseline ranking training.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the target column is missing.
    """

    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found: {csv_path}")

    LOGGER.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Training data must include a '{TARGET_COLUMN}' column. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    LOGGER.info("Loaded %s rows and %s columns", len(df), len(df.columns))
    return df


def build_training_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split a training DataFrame into features and target.

    Identifier columns are excluded from baseline features because they are not
    intended to be used directly as raw model inputs in this stage.
    """

    present_excluded_columns = [column for column in df.columns if column in LEAKAGE_COLUMNS]
    for column in present_excluded_columns:
        if column in {"price", "sales_channel_id"}:
            LOGGER.warning("Dropping potential leakage column: %s", column)

    feature_columns = [column for column in df.columns if column not in LEAKAGE_COLUMNS]

    if not feature_columns:
        raise ValueError("No usable feature columns found after excluding leakage and identifier columns.")

    features = df.loc[:, feature_columns].copy()
    target = df[TARGET_COLUMN].copy()

    LOGGER.info(
        "Using %s feature columns (excluded %s leakage/id columns) | total columns=%s | excluded=%s",
        len(feature_columns),
        len(present_excluded_columns),
        len(df.columns),
        present_excluded_columns,
    )
    return features, target, feature_columns


def infer_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from a feature frame."""

    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns


def cast_numeric_features_to_float(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric features to float before numeric imputation.

    This keeps missing numeric values compatible with a float fill value even
    when the source columns are integer-typed.
    """

    return frame.astype("float64")


def build_numeric_preprocessor(numeric_columns: list[str]) -> tuple[str, Pipeline, list[str]] | None:
    """Build the numeric preprocessing branch."""

    if not numeric_columns:
        return None

    numeric_pipeline = Pipeline(
        steps=[
            ("to_float", FunctionTransformer(cast_numeric_features_to_float, validate=False)),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]
    )
    return ("numeric", numeric_pipeline, numeric_columns)


def build_categorical_preprocessor(
    categorical_columns: list[str],
) -> tuple[str, Pipeline, list[str]] | None:
    """Build the categorical preprocessing branch."""

    if not categorical_columns:
        return None

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ("categorical", categorical_pipeline, categorical_columns)


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for baseline ranking features."""

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    numeric_transformer = build_numeric_preprocessor(numeric_columns)
    if numeric_transformer is not None:
        transformers.append(numeric_transformer)

    categorical_transformer = build_categorical_preprocessor(categorical_columns)
    if categorical_transformer is not None:
        transformers.append(categorical_transformer)

    if not transformers:
        raise ValueError("Preprocessor requires at least one numeric or categorical feature column.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """Build the end-to-end baseline ranking pipeline."""

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    classifier = LogisticRegression(max_iter=1000, solver="lbfgs")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def split_train_validation(
    features: pd.DataFrame,
    target: pd.Series,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/validation splits with a safe stratification fallback."""

    stratify_target: pd.Series | None = None
    target_value_counts = target.value_counts(dropna=False)
    if len(target_value_counts) > 1 and target_value_counts.min() >= 2:
        stratify_target = target

    return train_test_split(
        features,
        target,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_target,
    )


def compute_validation_auc(model: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> float | None:
    """Compute validation ROC-AUC when the validation target has both classes."""

    if y_valid.nunique(dropna=False) < 2:
        LOGGER.warning("Validation ROC-AUC skipped because validation labels contain only one class.")
        return None

    probabilities = model.predict_proba(x_valid)[:, 1]
    return float(roc_auc_score(y_valid, probabilities))


def save_artifacts(model: Pipeline, metadata: TrainingArtifacts, output_dir: Path) -> dict[str, Path]:
    """Persist the trained model and metadata inside the checkpoint directory."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / PIPELINE_ARTIFACT_NAME
    metadata_path = output_dir / METADATA_ARTIFACT_NAME

    LOGGER.info("Saving ranking pipeline to %s", pipeline_path)
    joblib.dump(model, pipeline_path)

    LOGGER.info("Saving ranking metadata to %s", metadata_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")

    return {"pipeline_path": pipeline_path, "metadata_path": metadata_path}


def train_ranker(
    csv_path: Path,
    output_dir: Path = DEFAULT_CHECKPOINT_DIR,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
    """Train the baseline ranker and save checkpoint artifacts.

    Args:
        csv_path: Path to the training CSV.
        output_dir: Directory where artifacts are saved.
        validation_size: Fraction reserved for validation.
        random_state: Random seed used for splitting.

    Returns:
        Summary dictionary describing the training run and saved artifacts.
    """

    df = load_training_data(csv_path)
    features, target, feature_columns = build_training_matrices(df)
    if target.nunique(dropna=False) < 2:
        raise ValueError("Ranking training requires at least two target classes in the label column.")

    numeric_columns, categorical_columns = infer_feature_types(features)

    LOGGER.info(
        "Feature types resolved: %s numeric, %s categorical",
        len(numeric_columns),
        len(categorical_columns),
    )

    x_train, x_valid, y_train, y_valid = split_train_validation(
        features=features,
        target=target,
        validation_size=validation_size,
        random_state=random_state,
    )

    LOGGER.info(
        "Training baseline ranker on %s rows, validating on %s rows",
        len(x_train),
        len(x_valid),
    )
    model = build_model_pipeline(numeric_columns=numeric_columns, categorical_columns=categorical_columns)
    model.fit(x_train, y_train)

    validation_auc = compute_validation_auc(model=model, x_valid=x_valid, y_valid=y_valid)
    if validation_auc is not None:
        LOGGER.info("Validation ROC-AUC: %.6f", validation_auc)

    metadata = TrainingArtifacts(
        target_column=TARGET_COLUMN,
        identifier_columns=list(IDENTIFIER_COLUMNS),
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        model_artifact=PIPELINE_ARTIFACT_NAME,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        validation_size=validation_size,
        random_state=random_state,
    )
    saved_paths = save_artifacts(model=model, metadata=metadata, output_dir=output_dir)

    summary: dict[str, Any] = {
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "feature_count": int(len(feature_columns)),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "validation_auc": validation_auc,
        "pipeline_path": str(saved_paths["pipeline_path"]),
        "metadata_path": str(saved_paths["metadata_path"]),
    }
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for baseline ranking training."""

    parser = argparse.ArgumentParser(description="Train the baseline ranking model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the ranking training CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory for ranking artifacts.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for the train/validation split.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for baseline ranking training."""

    args = parse_args()
    configure_logging(verbose=args.verbose)

    summary = train_ranker(
        csv_path=args.data,
        output_dir=args.output_dir,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    LOGGER.info("Training finished: %s", summary)


if __name__ == "__main__":
    main()
