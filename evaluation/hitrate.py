from metrics import hit_rate_scores, mean_hit_rate_at_k


if __name__ == "__main__":
    ranked_lists = [
        ["item_7", "item_3", "item_9", "item_1"],
        ["item_2", "item_6", "item_8", "item_4"],
        ["item_5", "item_1", "item_2", "item_3"],
    ]
    relevant_lists = [
        {"item_3"},
        {"item_4", "item_8"},
        {"item_10"},
    ]
    k = 3

    scores = hit_rate_scores(ranked_lists, relevant_lists, k)
    mean_score = mean_hit_rate_at_k(ranked_lists, relevant_lists, k)

    print(f"HitRate@{k} scores:", scores)
    print(f"Mean HitRate@{k}:", round(mean_score, 4))
