from metrics import mean_reciprocal_rank, mrr_scores


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

    scores = mrr_scores(ranked_lists, relevant_lists)
    mrr = mean_reciprocal_rank(ranked_lists, relevant_lists)

    print("Reciprocal ranks:", scores)
    print("MRR:", round(mrr, 4))
