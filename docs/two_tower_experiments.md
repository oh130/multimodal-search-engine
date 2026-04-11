# Two-Tower 실험 기록

## Run 1 - 2026-04-12

### 목적

- Two-Tower 후보 생성 모델 1차 학습
- 학습, 체크포인트 저장, Recall@300 비교까지 동작 확인

### 학습 설정

- 데이터: `data/processed/train_data_test.csv`
- 실행 환경: `docker compose exec -T rec-models`
- epoch: `5`
- batch size: `256`
- learning rate: `1e-3`
- weight decay: `1e-5`
- validation metric: `Recall@300`

### 학습 명령

```bash
docker compose exec -T rec-models python -m rec_models.candidate.train \
  --data /app/data/processed/train_data_test.csv \
  --epochs 5 \
  --batch-size 256
```

### 학습 결과

- 최고 validation `Recall@300`: `0.070264`
- 저장 파일:
  - `/app/data/checkpoints/candidate/two_tower.pt`
  - `/app/data/checkpoints/candidate/two_tower_metadata.json`

### epoch별 결과

| epoch | train loss | validation Recall@300 |
| --- | ---: | ---: |
| 1 | 5.542009 | 0.050920 |
| 2 | 5.539835 | 0.047627 |
| 3 | 5.526739 | 0.047883 |
| 4 | 5.502003 | 0.057484 |
| 5 | 5.480359 | 0.070264 |

### 비교 평가 명령

```bash
docker compose exec -T rec-models python -m rec_models.candidate.evaluator \
  --data /app/data/processed/train_data_test.csv \
  --top_k 300 \
  --mode compare \
  --max-users 100
```

### 비교 결과

- 평가 유저 수: `100`
- 기존 baseline `Recall@300`: `0.251665`
- Two-Tower `Recall@300`: `0.030333`
- 차이: `-0.221332`

### 해석

- 학습과 평가는 정상 동작함
- 현재 1차 Two-Tower 성능은 baseline보다 낮음
- epoch가 끝날수록 validation recall이 올라가서 추가 학습은 해볼 가치가 있음

### 다음 할 일

- epoch `20`으로 재학습
- batch size `128`, `256` 비교
- 성능이 계속 낮으면 feature / negative sampling 개선 검토

## Run 2 - 2026-04-12

### 목적

- epoch를 늘렸을 때 Recall@300이 얼마나 개선되는지 확인

### 학습 설정

- 데이터: `data/processed/train_data_test.csv`
- 실행 환경: `docker compose exec -T rec-models`
- epoch: `20`
- batch size: `256`
- 나머지 설정: Run 1과 동일

### 학습 명령

```bash
docker compose exec -T rec-models python -m rec_models.candidate.train --data /app/data/processed/train_data_test.csv --epochs 20 --batch-size 256
```

### 비교 평가 명령

```bash
docker compose exec -T rec-models python -m rec_models.candidate.evaluator --data /app/data/processed/train_data_test.csv --top_k 300 --mode compare --max-users 100
```

### 비교 결과

- 평가 유저 수: `100`
- 기존 baseline `Recall@300`: `0.251665`
- Two-Tower `Recall@300`: `0.175214`
- 차이: `-0.076451`

### 해석

- Run 1보다 성능이 크게 개선됨
- 아직 baseline보다는 낮음
- 다음 실험은 `batch_size=128` 비교가 우선
