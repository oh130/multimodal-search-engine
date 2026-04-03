"""Baseline ranking inference utilities.

This module loads the persisted ranking pipeline and metadata, aligns candidate
features to the training feature contract, and returns candidate scores while
preserving useful identifier columns for downstream recommendation stages.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
PIPELINE_ARTIFACT_NAME = "ranking_baseline.joblib"
METADATA_ARTIFACT_NAME = "ranking_baseline_metadata.json"


def configure_logging(verbose: bool = False) -> None:
    """Configure logging for CLI execution."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_artifacts(checkpoint_dir: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load the saved ranking model and metadata from the checkpoint directory."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    pipeline_path = checkpoint_dir / PIPELINE_ARTIFACT_NAME
    metadata_path = checkpoint_dir / METADATA_ARTIFACT_NAME

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Ranking pipeline artifact not found: {pipeline_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Ranking metadata artifact not found: {metadata_path}")

    LOGGER.info("Loading ranking pipeline from %s", pipeline_path)
    model = joblib.load(pipeline_path)

    LOGGER.info("Loading ranking metadata from %s", metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, metadata


def prepare_inference_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align inference features to the feature columns seen during training.

    Missing feature columns are added with null values so the saved preprocessing
    pipeline can apply the same imputers used during training. Extra columns are
    safely ignored.
    """

    aligned = df.copy()
    missing_columns = [column for column in feature_columns if column not in aligned.columns]
    if missing_columns:
        LOGGER.warning(
            "Inference input is missing %s expected feature columns. Filling with nulls.",
            len(missing_columns),
        )
        for column in missing_columns:
            aligned[column] = np.nan

    return aligned.loc[:, feature_columns]


def _extract_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Extract ranking scores from a fitted classifier with graceful fallbacks."""

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]
        return probabilities.ravel()

    if hasattr(model, "decision_function"):
        raw_scores = np.asarray(model.decision_function(features), dtype=float).ravel()
        if raw_scores.size == 0:
            return np.asarray([], dtype=float)
        score_min = float(raw_scores.min())
        score_max = float(raw_scores.max())
        if score_max > score_min:
            return (raw_scores - score_min) / (score_max - score_min)
        return np.zeros_like(raw_scores, dtype=float)

    LOGGER.warning("Model does not expose predict_proba or decision_function. Falling back to predict.")
    return np.asarray(model.predict(features), dtype=float).ravel()


def score_candidates(candidates: pd.DataFrame, checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> pd.DataFrame:
    """Score candidate rows using the saved baseline ranker.

    Args:
        candidates: Candidate DataFrame with arbitrary extra columns.
        checkpoint_dir: Directory containing saved ranking artifacts.

    Returns:
        DataFrame containing preserved identifiers when available and a `score`
        column suitable for later ranking stages.
    """

    model, metadata = load_artifacts(checkpoint_dir=checkpoint_dir)

    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Ranking metadata does not contain feature_columns.")

    identifier_columns = metadata.get("identifier_columns", [])
    aligned_features = prepare_inference_features(candidates, feature_columns=feature_columns)
    scores = _extract_scores(model=model, features=aligned_features)

    preserved_columns = [column for column in identifier_columns if column in candidates.columns]
    result = candidates.loc[:, preserved_columns].copy() if preserved_columns else pd.DataFrame(index=candidates.index)
    result["score"] = scores
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch candidate scoring from CSV."""

    parser = argparse.ArgumentParser(description="Score ranking candidates with the baseline ranker.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a candidate CSV file.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory containing ranking artifacts.",
    )
    parser.add_argument("--output", type=Path, help="Optional output CSV path for scored candidates.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point for batch candidate scoring."""

    args = parse_args()
    configure_logging(verbose=args.verbose)

    candidate_path = args.input.expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidate_path}")

    LOGGER.info("Loading candidate rows from %s", candidate_path)
    candidates = pd.read_csv(candidate_path)
    scored = score_candidates(candidates=candidates, checkpoint_dir=args.checkpoint_dir)

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
        LOGGER.info("Saved scored candidates to %s", output_path)
    else:
        LOGGER.info("Scored %s candidate rows", len(scored))
        print(scored.head().to_string(index=False))


if __name__ == "__main__":
    main()
