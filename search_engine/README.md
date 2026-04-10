# multimodal-search-engine
2026-1. Capstone Design I

이 저장소는 CLIP 기반 멀티모달 검색 엔진을 구현하기 위한 코드 모음이다.
텍스트와 이미지를 같은 임베딩 공간에 매핑한 뒤, FAISS HNSW 인덱스를 이용해 빠르게 검색한다.

## 프로젝트 구성

* `data_embedding.py`

  * 텍스트와 이미지를 CLIP 임베딩 공간으로 변환하는 코드
  * 학습 또는 전처리 단계에서 사용
  * 검색 엔진에 넣을 임베딩을 생성하는 역할

* `search_engine.py`

  * FAISS HNSW 인덱스를 이용한 검색 엔진 코드
  * 텍스트 검색, 이미지 검색, 혼합 검색을 지원
  * 이미 임베딩된 벡터를 입력받아 검색 결과를 반환

## 구현 내용

### 1. 임베딩 생성

`data_embedding.py`는 CLIP 기반 텍스트 인코더와 이미지 인코더를 사용하여 입력 데이터를 임베딩 벡터로 변환한다.

### 2. 검색 엔진

`search_engine.py`는 FAISS의 HNSW 인덱스를 사용하여 벡터 검색을 수행한다.

지원하는 검색 방식은 다음과 같다.

* 텍스트 입력 검색
* 이미지 입력 검색
* 텍스트 + 이미지 혼합 검색

### 3. 성능 목표

* 검색 응답 시간: 200ms 이내
* 벡터 검색 방식: FAISS HNSW

## 사용 흐름

1. 원본 데이터 준비
2. `data_embedding.py`로 텍스트/이미지 임베딩 생성
3. 생성된 임베딩을 `search_engine.py`의 FAISS HNSW 인덱스에 저장
4. 쿼리 입력 시 텍스트, 이미지, 혼합 임베딩을 이용해 검색 수행

## 실행 환경

예시 환경:

* Python 3.10 이상
* PyTorch
* torchvision
* transformers
* faiss
* numpy

### 설치 예시

```bash
pip install torch torchvision transformers faiss-cpu numpy pillow
```

GPU 버전을 사용할 경우 환경에 맞는 FAISS/GPU 패키지를 설치한다.

## 파일 실행 예시

### 임베딩 생성

```bash
python data_embedding.py
```

### 검색 엔진 실행

```bash
python search_engine.py
```

## 참고

이 프로젝트는 검색 품질과 응답 속도를 함께 고려한 멀티모달 검색 시스템을 목표로 한다.
추가로 실제 서비스 수준으로 확장할 때는 다음 항목을 보완할 수 있다.

* 배치 검색 최적화
* 인덱스 파라미터 튜닝 (`efSearch`, `efConstruction`, `M`)
* 캐싱
* 검색 로그 및 평가 지표(MRR, NDCG@10) 측정

## 작성 파일 요약

* `data_embedding.py`: CLIP 임베딩 생성
* `search_engine.py`: FAISS HNSW 기반 검색 엔진
* `README.md`: 전체 작업 내용 및 업로드 방법 설명
