# Multimodal Search Engine

CLIP 기반의 멀티모달 검색 엔진입니다.
텍스트와 이미지를 같은 임베딩 공간에 매핑한 뒤, FAISS HNSW 인덱스를 이용해 유사한 상품을 검색합니다.

이 검색 엔진은 다음 두 가지 모드를 지원합니다.

* `test` 모드: 코드 내부 더미 데이터로 검색
* `production` 모드: H&M Kaggle dataset 기반 검색 (현재 기능이 불완전할 수 있음)

현재 작성한 코드는 기본값이 `test`입니다. 전처리된 데이터 기반으로 검색하고 싶다면 production 모드를 사용하면 됩니다.

---

## API 규약

### 검색

* 포트: `8002`
* `POST /search`

#### 요청 형식

```json
{
  "query": "blue jacket",
  "image_base64": null,
  "top_k": 10
}
```

* `query`: 검색어 문자열
* `image_base64`: base64로 인코딩된 이미지 문자열 또는 `null`
* `top_k`: 반환할 결과 개수

#### 응답 형식

```json
{
  "search_type": "text",
  "results": [
    {
      "product_id": "100001",
      "name": "Blue Denim Jacket",
      "score": 0.9231,
      "price": 79.9
    }
  ],
  "latency_ms": 12.34,
  "total_count": 1
}
```

#### 응답 필수 필드

* `search_type`: `"text"` | `"image"` | `"hybrid"`
* `results`: 검색 결과 배열
* `latency_ms`: 응답 시간(ms)
* `total_count`: 반환된 결과 개수

각 `results` 항목은 다음 필드를 포함합니다.

* `product_id`
* `name`
* `score`
* `price`

---

## 디렉토리 구조

```text
search_engine_sample/
├── app.py
├── search_engine.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 각 파일의 역할

### `search_engine.py`
* OpenAI CLIP 모델을 이용한 텍스트/이미지 임베딩
* 상품 메타데이터 로딩
* H&M 데이터셋 또는 더미 데이터 기반 인덱스 구축
* FAISS HNSW 검색
* 검색 결과를 API 응답 형식으로 변환

### `app.py`
* `POST /search` 요청 처리
* 검색 엔진 호출
* JSON 응답 반환
* 서버 실행 진입점 제공

### `Dockerfile`
* Python 환경 설정
* 필요한 패키지 설치
* 소스 코드 복사
* FastAPI 서버 실행

### `requirements.txt`

프로젝트 실행에 필요한 Python 패키지 목록

예시 패키지:

* `fastapi`
* `uvicorn`
* `torch`
* `transformers`
* `faiss-cpu`
* `numpy`
* `pandas`
* `Pillow`

---

## 실행 방법

### 1. 디렉토리 이동

```bash
cd search_engine
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 로컬 실행

### 검색 엔진 확인

```bash
python search_engine.py
```

이 명령은 검색 엔진이 정상적으로 인덱스를 만들고 샘플 검색을 수행하는지 확인합니다.

### API 서버 실행

```bash
python app.py
```

또는 다음과 같이 실행할 수 있습니다.

```bash
uvicorn app:app --host 0.0.0.0 --port 8002
```

실행 후 다음 주소로 접근합니다.

* `http://localhost:8002/search`

---

## 모드 설정

기본값은 `test` 모드입니다.

```python
mode: str = "test",        # 기본
# mode: str = "production",  # 최종용
```

### `test` 모드
* 코드 내부의 더미 데이터로 검색 (초안 구현 완료)

### `production` 모드
* `data/` 디렉토리의 CSV 파일과 상품 이미지를 활용하여 검색

---

## Docker 실행

### 빌드

프로젝트 루트에서 실행합니다.

```bash
docker build -t multimodal-search -f search_engine_sample/Dockerfile .
```

### 실행

기본 `test` 모드:

```bash
docker run --rm -p 8002:8002 -e MODE=test multimodal-search
```

`production` 모드:

```bash
docker run --rm -p 8002:8002 -e MODE=production multimodal-search
```

---

## 요청 예시

### 텍스트만 검색

```bash
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"summer dress","image_base64":null,"top_k":10}'
```

### 이미지 + 텍스트 검색

```json
{
  "query": "black shoes",
  "image_base64": "<base64 encoded image>",
  "top_k": 10
}
```

### 이미지만 검색

```json
{
  "query": "",
  "image_base64": "<base64 encoded image>",
  "top_k": 10
}
```

---

## 동작 흐름

1. 상품 데이터 로딩
2. CLIP 임베딩 생성
3. HNSW 인덱스 구축
4. 사용자 쿼리 입력
5. 텍스트/이미지/혼합 임베딩 계산
6. 유사 상품 Top-K 반환

---

## 참고 사항
* `search_type`은 입력 형태에 따라 자동으로 결정됩니다.
* `results`에는 `product_id`, `name`, `score`, `price`가 포함됩니다.
* 검색 품질은 CLIP 임베딩과 인덱스 파라미터에 따라 달라질 수 있습니다.
* `production` 모드에서는 H&M 데이터셋 파일 경로가 올바르게 배치되어 있어야 합니다.

