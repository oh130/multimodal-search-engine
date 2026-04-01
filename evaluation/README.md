# Evaluation Dashboard

This folder contains offline ranking metrics, A/B test utilities, and a Streamlit dashboard for visualizing experiment results.

## Files

- `metrics.py`: HitRate@K, MRR, nDCG@K
- `ab_test.py`: p-value and confidence interval utilities for A/B tests
- `streamlit_app.py`: Streamlit dashboard for ranking and A/B evaluation
- `sample_ranking.csv`: sample ranking evaluation input
- `sample_ab.csv`: sample A/B evaluation input

## Run

```powershell
python -m pip install streamlit pandas altair
python -m streamlit run .\evaluation\streamlit_app.py
```

## CSV format

Ranking CSV:

```csv
query_id,ranked_items,relevant_items
q1,"item_7|item_3|item_9|item_1","item_3"
```

A/B CSV:

```csv
group,value
control,0
treatment,1
```
