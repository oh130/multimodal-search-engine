from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from metrics import mean_hit_rate_at_k, mean_ndcg_at_k, mean_reciprocal_rank


DEFAULT_TRANSACTIONS_FILE = Path("data/raw/transactions_train.csv")
DEFAULT_ARTICLE_FEATURES_FILE = Path("data/processed/article_features.csv")
DEFAULT_ARTICLE_FEATURES_FALLBACK_FILE = Path("data/processed/articles_feature.csv")
DEFAULT_OUTPUT_FILE = Path("data/processed/baseline_a_predictions.csv")


@dataclass(frozen=True)
class Interaction:
    customer_id: str
    article_id: str
    timestamp: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metadata + popularity baseline and evaluate it offline.",
    )
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS_FILE)
    parser.add_argument("--article-features", type=Path, default=DEFAULT_ARTICLE_FEATURES_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optionally limit transaction rows for quick testing.",
    )
    return parser.parse_args()


def resolve_article_feature_path(preferred_path: Path) -> Path:
    if preferred_path.exists():
        return preferred_path
    if DEFAULT_ARTICLE_FEATURES_FALLBACK_FILE.exists():
        return DEFAULT_ARTICLE_FEATURES_FALLBACK_FILE
    raise FileNotFoundError(
        f"Missing article feature files: {preferred_path} and {DEFAULT_ARTICLE_FEATURES_FALLBACK_FILE}"
    )


def load_article_features(file_path: Path) -> dict[str, dict[str, str]]:
    article_features: dict[str, dict[str, str]] = {}
    with file_path.open(newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            article_id = row["article_id"].strip()
            if not article_id:
                continue
            article_features[article_id] = {
                "category": row.get("category", "").strip() or "UNKNOWN",
                "main_category": row.get("main_category", "").strip() or "UNKNOWN",
                "color": row.get("color", "").strip() or "UNKNOWN",
                "prod_name": row.get("prod_name", "").strip() or "UNKNOWN",
            }
    return article_features


def load_interactions(file_path: Path, max_rows: int | None = None) -> dict[str, list[Interaction]]:
    interactions_by_user: dict[str, list[Interaction]] = defaultdict(list)
    with file_path.open(newline="", encoding="utf-8-sig") as infile:
        reader = csv.DictReader(infile)
        for index, row in enumerate(reader, start=1):
            if max_rows is not None and index > max_rows:
                break
            customer_id = row["customer_id"].strip()
            article_id = row["article_id"].strip()
            timestamp = row["t_dat"].strip()
            if not customer_id or not article_id:
                continue
            interactions_by_user[customer_id].append(
                Interaction(customer_id=customer_id, article_id=article_id, timestamp=timestamp)
            )

    for user_interactions in interactions_by_user.values():
        user_interactions.sort(key=lambda interaction: interaction.timestamp)
    return interactions_by_user


def split_train_validation(
    interactions_by_user: dict[str, list[Interaction]],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    history_by_user: dict[str, list[str]] = {}
    target_by_user: dict[str, str] = {}

    for customer_id, interactions in interactions_by_user.items():
        unique_sequence: list[str] = []
        for interaction in interactions:
            if not unique_sequence or unique_sequence[-1] != interaction.article_id:
                unique_sequence.append(interaction.article_id)

        if len(unique_sequence) < 2:
            continue

        history_by_user[customer_id] = unique_sequence[:-1]
        target_by_user[customer_id] = unique_sequence[-1]

    return history_by_user, target_by_user


def build_popularity_indexes(
    history_by_user: dict[str, list[str]],
    article_features: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    global_counter: Counter[str] = Counter()
    category_to_articles: dict[str, list[str]] = defaultdict(list)
    main_category_to_articles: dict[str, list[str]] = defaultdict(list)
    color_to_articles: dict[str, list[str]] = defaultdict(list)

    for article_ids in history_by_user.values():
        for article_id in article_ids:
            if article_id not in article_features:
                continue
            global_counter[article_id] += 1

    for article_id, feature in article_features.items():
        category_to_articles[feature["category"]].append(article_id)
        main_category_to_articles[feature["main_category"]].append(article_id)
        color_to_articles[feature["color"]].append(article_id)

    def sort_articles(article_ids: list[str]) -> list[str]:
        return sorted(article_ids, key=lambda article_id: (-global_counter[article_id], article_id))

    global_popular = sort_articles(list(article_features.keys()))
    category_popular = {key: sort_articles(article_ids) for key, article_ids in category_to_articles.items()}
    main_category_popular = {key: sort_articles(article_ids) for key, article_ids in main_category_to_articles.items()}
    color_popular = {key: sort_articles(article_ids) for key, article_ids in color_to_articles.items()}
    return global_popular, category_popular, main_category_popular, color_popular


def deduplicate_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def recommend_for_user(
    history_article_ids: list[str],
    article_features: dict[str, dict[str, str]],
    global_popular: list[str],
    category_popular: dict[str, list[str]],
    main_category_popular: dict[str, list[str]],
    color_popular: dict[str, list[str]],
    top_k: int,
) -> list[str]:
    history_set = set(history_article_ids)
    history_features = [article_features[article_id] for article_id in history_article_ids if article_id in article_features]

    preferred_categories = deduplicate_preserve_order(feature["category"] for feature in reversed(history_features))
    preferred_main_categories = deduplicate_preserve_order(feature["main_category"] for feature in reversed(history_features))
    preferred_colors = deduplicate_preserve_order(feature["color"] for feature in reversed(history_features))

    candidates: list[str] = []

    for category in preferred_categories:
        candidates.extend(category_popular.get(category, []))
    for main_category in preferred_main_categories:
        candidates.extend(main_category_popular.get(main_category, []))
    for color in preferred_colors:
        candidates.extend(color_popular.get(color, []))
    candidates.extend(global_popular)

    recommendations: list[str] = []
    for article_id in deduplicate_preserve_order(candidates):
        if article_id in history_set:
            continue
        recommendations.append(article_id)
        if len(recommendations) >= top_k:
            break
    return recommendations


def write_predictions(
    output_file: Path,
    predictions_by_user: dict[str, list[str]],
    targets_by_user: dict[str, str],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["customer_id", "predictions", "target"])
        writer.writeheader()
        for customer_id, predictions in predictions_by_user.items():
            writer.writerow(
                {
                    "customer_id": customer_id,
                    "predictions": "|".join(predictions),
                    "target": targets_by_user[customer_id],
                }
            )


def main() -> None:
    args = parse_args()
    article_feature_path = resolve_article_feature_path(args.article_features)

    article_features = load_article_features(article_feature_path)
    interactions_by_user = load_interactions(args.transactions, max_rows=args.max_rows)
    history_by_user, target_by_user = split_train_validation(interactions_by_user)

    (
        global_popular,
        category_popular,
        main_category_popular,
        color_popular,
    ) = build_popularity_indexes(history_by_user, article_features)

    predictions_by_user: dict[str, list[str]] = {}
    ranked_lists: list[list[str]] = []
    relevant_lists: list[list[str]] = []

    for customer_id, history_article_ids in history_by_user.items():
        target_article_id = target_by_user[customer_id]
        if target_article_id not in article_features:
            continue

        predictions = recommend_for_user(
            history_article_ids=history_article_ids,
            article_features=article_features,
            global_popular=global_popular,
            category_popular=category_popular,
            main_category_popular=main_category_popular,
            color_popular=color_popular,
            top_k=args.top_k,
        )
        predictions_by_user[customer_id] = predictions
        ranked_lists.append(predictions)
        relevant_lists.append([target_article_id])

    if not ranked_lists:
        raise ValueError("No evaluation rows were generated. Check data paths and history length.")

    write_predictions(args.output, predictions_by_user, target_by_user)

    print(f"users_evaluated: {len(ranked_lists)}")
    print(f"predictions_file: {args.output}")
    print(f"HitRate@{args.top_k}: {mean_hit_rate_at_k(ranked_lists, relevant_lists, args.top_k):.4f}")
    print(f"MRR: {mean_reciprocal_rank(ranked_lists, relevant_lists):.4f}")
    print(f"nDCG@{args.top_k}: {mean_ndcg_at_k(ranked_lists, relevant_lists, args.top_k):.4f}")


if __name__ == "__main__":
    main()
