# multimodal-search-engine
2026-1. Capstone Design I

# Data Pipeline

이 디렉토리는 추천 모델 학습에 필요한 전처리 데이터를 생성하는 파이프라인을 담당한다.  
원본 H&M 데이터셋의 `articles.csv`, `customers.csv`, `transactions_train.csv`를 기반으로 다음 4단계의 처리를 수행한다.

1. 고객 메타데이터 전처리
2. 상품 메타데이터 전처리
3. 거래 기반 상품 통계 생성
4. 구매 이력 기반 학습용 데이터 생성

최종적으로 추천 모델 학습에 사용할 `train_data.csv` 또는 테스트용 `train_data_test.csv`를 생성한다.

---

## Directory Structure

```text
data_pipeline/
├── build_article_features.py
├── build_customer_features.py
├── build_item_features.py
├── build_ranking_train_data.py
└── README.md
```

## Required Raw Files

전처리 데이터가 하나도 없는 상태에서 시작한다면 아래 raw 파일 3개가 먼저 존재해야 한다.

- `data/raw/customers.csv`
- `data/raw/articles.csv`
- `data/raw/transactions_train.csv`

## Execution Order

`data/processed/`가 비어 있고 raw 파일만 있는 상태라면, 저장소 루트에서 아래 순서로 실행한다.

1. `python data_pipeline/build_customer_features.py`
2. `python data_pipeline/build_article_features.py`
3. `python data_pipeline/build_item_features.py`
4. `python data_pipeline/build_ranking_train_data.py`

이 순서를 지키는 이유는 다음과 같다.

- `build_customer_features.py`는 `customer_features.csv`를 생성한다.
- `build_article_features.py`는 `articles_feature.csv`를 생성한다.
- `build_item_features.py`는 거래 이력으로부터 `item_features` 계열 파일을 생성한다.
- `build_ranking_train_data.py`는 `customer_features.csv`와 `articles_feature.csv`를 입력으로 사용한다.

## Mode Guide

`build_item_features.py`와 `build_ranking_train_data.py`는 각각 파일 내부의 `MODE` 값을 통해 `test`와 `production`을 전환한다.

- `MODE = "test"`
  - 로컬 smoke test, 파이프라인 연결 확인, 빠른 검증용
- `MODE = "production"`
  - 실제 서비스/학습용 전체 데이터 생성용

`build_customer_features.py`와 `build_article_features.py`는 모드가 없고 항상 표준 파일을 생성한다.

## Test Run

빠른 로컬 검증이 목적이면 아래처럼 사용한다.

- `build_item_features.py`의 `MODE = "test"`
- `build_ranking_train_data.py`의 `MODE = "test"`

실행 명령:

```bash
python data_pipeline/build_customer_features.py
python data_pipeline/build_article_features.py
python data_pipeline/build_item_features.py
python data_pipeline/build_ranking_train_data.py
```

생성되는 주요 파일:

- `data/processed/customer_features.csv`
- `data/processed/articles_feature.csv`
- `data/processed/item_features_test.csv`
- `data/processed/train_data_test.csv`

## Production Run

실제 서비스/학습용 데이터를 재생성하려면 아래처럼 사용한다.

- `build_item_features.py`의 `MODE = "production"`
- `build_ranking_train_data.py`의 `MODE = "production"`

실행 명령:

```bash
python data_pipeline/build_customer_features.py
python data_pipeline/build_article_features.py
python data_pipeline/build_item_features.py
python data_pipeline/build_ranking_train_data.py
```

생성되는 주요 파일:

- `data/processed/customer_features.csv`
- `data/processed/articles_feature.csv`
- `data/processed/item_features.csv`
- `data/processed/train_data_production.csv`

## Service / Training Outputs

현재 코드베이스 기준으로 주요 소비 경로는 아래 파일들을 기대한다.

- 검색 엔진: `data/processed/articles_feature.csv`
- 추천 candidate 서빙: `data/processed/articles_feature.csv`, `data/processed/item_features.csv`
- 랭킹 서빙: `data/processed/customer_features.csv`
- 테스트용 랭킹 학습 데이터: `data/processed/train_data_test.csv`
- 실제 랭킹 학습 데이터: `data/processed/train_data_production.csv`

## Important Notes

- 저장소 루트에서 실행해야 한다.
- `build_item_features.py`와 `build_ranking_train_data.py`의 `MODE`는 서로 독립적이다.
- 추천/검색 서비스를 실제로 붙여보려면 `build_item_features.py`는 `production`이어야 한다. `test` 모드에서는 `item_features_test.csv`만 생성된다.
- 전체 학습 데이터를 만들려면 `build_ranking_train_data.py`도 `production`이어야 한다.
- 로컬 확인만 할 때는 뒤 두 파일만 `test`로 두고 실행하면 충분하다.
