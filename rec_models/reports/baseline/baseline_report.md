# Recommendation Baseline Report

## Dataset

- generated_at_utc: `2026-04-11T13:58:53.530292+00:00`
- source_data: `/home/jiwon/projects/multimodal-search-engine/data/processed/train_data_test.csv`
- rows: `18070`
- users: `2954`
- items: `13168`
- positive_rows: `9035`
- negative_rows: `9035`
- top_k: `50`
- candidate_k: `300`
- max_users: `30`

## Metrics

| Component | Metric | Value |
| --- | --- | ---: |
| Candidate | Recall@300 | 0.361821 |
| Ranking | AUC | 0.947862 |
| Ranking | HitRate@50 | 1.000000 |
| Ranking | NDCG@50 | 0.971311 |
| Recommender | HitRate@50 | 1.000000 |
| Recommender | NDCG@50 | 0.933807 |
| Recommender | Coverage@50 | 0.014961 |
| Popularity baseline | HitRate@50 | 1.000000 |
| Popularity baseline | NDCG@50 | 0.995353 |
| Popularity baseline | Coverage@50 | 0.014961 |
