from metrics import mean_ndcg_at_k, ndcg_scores


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

    scores = ndcg_scores(ranked_lists, relevant_lists, k)
    mean_score = mean_ndcg_at_k(ranked_lists, relevant_lists, k)

    print(f"nDCG@{k} scores:", [round(score, 4) for score in scores])
    print(f"Mean nDCG@{k}:", round(mean_score, 4))
