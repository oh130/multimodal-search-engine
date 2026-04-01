# 평가 대시보드

이 폴더에는 오프라인 랭킹 지표 계산, A/B 테스트 통계 유틸, 그리고 실험 결과를 시각화하는 Streamlit 대시보드가 포함되어 있습니다.

## 파일 구성

- `metrics.py`: `HitRate@K`, `MRR`, `nDCG@K` 계산
- `ab_test.py`: A/B 테스트용 `p-value`, 신뢰구간 계산
- `streamlit_app.py`: 랭킹/A-B 평가 결과 시각화 대시보드
- `sample_ranking.csv`: 랭킹 평가용 예시 입력 파일
- `sample_ab.csv`: A/B 테스트용 예시 입력 파일

## 실행 방법

```powershell
python -m pip install streamlit pandas altair
python -m streamlit run .\evaluation\streamlit_app.py
```

## CSV 형식

랭킹 평가 CSV:

```csv
query_id,ranked_items,relevant_items
q1,"item_7|item_3|item_9|item_1","item_3"
```

A/B 테스트 CSV:

```csv
group,value
control,0
treatment,1
```
