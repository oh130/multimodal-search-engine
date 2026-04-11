"""Offline evaluator for the current ranking baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rec_models.common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score
    from rec_models.evaluation.data_utils import (
        EvaluationContext,
        build_evaluation_context,
        load_evaluation_data,
    )
    from rec_models.ranking.infer import DEFAULT_CHECKPOINT_DIR, _extract_scores, prepare_inference_features
    from rec_models.serving.ranking_service import load_ranking_pipeline
except ImportError:  # pragma: no cover - supports running from rec_models/ as cwd
    from common.metrics import hit_rate_at_k, mean_metric, ndcg_at_k, safe_roc_auc_score  # type: ignore[no-redef]
    from evaluation.data_utils import (  # type: ignore[no-redef]
        EvaluationContext,
        build_evaluation_context,
        load_evaluation_data,
    )
    from ranking.infer import DEFAULT_CHECKPOINT_DIR, _extract_scores, prepare_inference_features  # type: ignore[no-redef]
    from serving.ranking_service import load_ranking_pipeline  # type: ignore[no-redef]


DEFAULT_TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the baseline ranking model offline.")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed ranking/evaluation data.")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Cutoff K for grouped ranking metrics.")
    parser.add_argument("--max-users", type=int, help="Optional cap for smoke checks or faster iteration.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory containing ranking artifacts.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional output path for JSON metrics.")
    return parser.parse_args()


def evaluate_ranking_model(
    data: pd.DataFrame,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    top_k: int = DEFAULT_TOP_K,
    max_users: int | None = None,
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Evaluate pointwise AUC and grouped ranking metrics for the saved ranker."""

    model, metadata = load_ranking_pipeline(checkpoint_dir=checkpoint_dir)
    feature_columns = metadata.get("feature_columns", [])
    if not feature_columns:
        raise ValueError("Ranking metadata does not contain feature_columns.")

    features = prepare_inference_features(data, feature_columns=feature_columns)
    scores = _extract_scores(model=model, features=features)

    evaluation_context = context or build_evaluation_context(data, max_users=max_users)
    user_set = set(evaluation_context.sampled_user_ids)

    scored = data.copy()
    scored["customer_id"] = scored["customer_id"].astype(str)
    scored["article_id"] = scored["article_id"].astype(str)
    scored["score"] = scores
    scored["is_positive"] = evaluation_context.positive_mask.astype(bool)
    scored = scored.loc[scored["customer_id"].isin(user_set)].copy()

    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []
    for user_id in evaluation_context.sampled_user_ids:
        user_rows = scored.loc[scored["customer_id"].eq(user_id)]
        if user_rows.empty:
            continue

        ranked_items = (
            user_rows.sort_values(["score", "article_id"], ascending=[False, True])["article_id"].astype(str).tolist()
        )
        ranked_lists.append(ranked_items)
        relevant_lists.append(evaluation_context.ground_truth_by_user.get(user_id, []))

    auc = safe_roc_auc_score(scored["is_positive"].astype(int).tolist(), scored["score"].astype(float).tolist())
    return {
        "rows_evaluated": int(len(scored)),
        "users_evaluated": len(ranked_lists),
        "auc": auc,
        f"HitRate@{top_k}": mean_metric(ranked_lists, relevant_lists, hit_rate_at_k, top_k),
        f"NDCG@{top_k}": mean_metric(ranked_lists, relevant_lists, ndcg_at_k, top_k),
    }


def print_report(metrics: dict[str, Any], top_k: int) -> None:
    """Pretty-print ranking metrics."""

    print("Ranking Model")
    print(f"{'rows evaluated':<18} {metrics['rows_evaluated']}")
    print(f"{'users evaluated':<18} {metrics['users_evaluated']}")
    auc = metrics["auc"]
    auc_text = f"{auc:.6f}" if auc is not None else "n/a"
    print(f"{'AUC':<18} {auc_text}")
    print(f"{f'HitRate@{top_k}':<18} {metrics[f'HitRate@{top_k}']:.6f}")
    print(f"{f'NDCG@{top_k}':<18} {metrics[f'NDCG@{top_k}']:.6f}")


def main() -> None:
    args = parse_args()
    data = load_evaluation_data(args.data)
    metrics = evaluate_ranking_model(
        data=data,
        checkpoint_dir=args.checkpoint_dir,
        top_k=args.top_k,
        max_users=args.max_users,
    )
    print_report(metrics, top_k=args.top_k)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nSaved JSON metrics to {output_path}")


if __name__ == "__main__":
    main()
