# multimodal-search-engine
2026-1. Capstone Design I

# Data Pipeline

이 디렉토리는 추천 모델 학습에 필요한 전처리 데이터를 생성하는 파이프라인을 담당한다.  
원본 H&M 데이터셋의 `articles.csv`, `customers.csv`, `transactions_train.csv`를 기반으로 다음 3단계의 처리를 수행한다.

1. 상품 메타데이터 전처리
2. 고객 메타데이터 전처리
3. 구매 이력 기반 학습용 데이터 생성

최종적으로 추천 모델 학습에 사용할 `train_data.csv` 또는 테스트용 `train_data_test.csv`를 생성한다.

---

## Directory Structure

```text
data_pipeline/
├── articles_extract.py
├── customer_extract.py
├── make_train_data.py
└── README.md