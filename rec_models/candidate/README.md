# Candidate Generation

Two-Tower 기반 후보 생성 모델을 구현한다.

## 목표
전체 상품 중 사용자가 선호할 가능성이 높은 후보 상품 top-k를 빠르게 추출한다.

## 주요 기능
- user tower / item tower 구현
- embedding 학습
- negative sampling 반영
- item embedding export
- Recall@K 평가

## 입력 예시
- user feature: 최근 클릭, 최근 구매, 선호 카테고리, 가격대, persona
- item feature: 카테고리, 브랜드, 가격, 속성

## 출력
- user embedding
- item embedding
- candidate top-k item list

## 실험 자동화

반복 실험은 `run_experiments.py`로 돌린다. 하이퍼파라미터 조합을 CLI 인자로 넘기면 각 조합별 학습, `Recall@K` 평가, 요약 CSV/JSON 저장까지 자동으로 수행한다.

실험 이름은 다음 규칙으로 관리하는 것을 권장한다.

- `two_tower_v1_grid_search`
- `two_tower_v2_epoch_extend`
- `two_tower_v3_feature_update`
- `two_tower_v4_negative_sampling`

즉 `모델명 + 버전 + 실험 목적` 형식으로 맞추면 결과 디렉터리와 문서 해석이 덜 꼬인다.

negative sampling 실험은 다음 추가 인자로 제어할 수 있다.

- `--negatives-per-positive`
- `--hard-negative-ratios`
- `--sampled-negative-weights`

예시:

```bash
python -m rec_models.candidate.run_experiments \
  --data data/processed/train_data_test.csv \
  --epochs 5,10,20 \
  --batch-sizes 128,256 \
  --learning-rates 1e-3,5e-4 \
  --weight-decays 1e-5 \
  --top_k 300 \
  --max-users 100 \
  --experiment-name two_tower_v1_grid_search
```

실행 전 조합만 확인하려면:

```bash
python -m rec_models.candidate.run_experiments \
  --data data/processed/train_data_test.csv \
  --epochs 5,10 \
  --batch-sizes 128,256 \
  --dry-run
```
