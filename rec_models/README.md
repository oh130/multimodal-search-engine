# multimodal-search-engine
2026-1. Capstone Design I

# Recommendation Models

이 디렉토리는 추천 시스템의 핵심 모델 파트를 담당한다.

## 역할
- Candidate Generation 모델(Two-Tower) 구현
- Ranking 모델(DeepFM 또는 Wide&Deep) 구현
- 세션 기반 추천 로직 지원
- 서비스 추론용 추천 파이프라인 제공

## 디렉토리 구조

- `common/`
  - 공통 설정, metric, loss, util, feature schema
- `candidate/`
  - Two-Tower 기반 후보 생성 모델
- `ranking/`
  - DeepFM 기반 랭킹 모델
- `session/`
  - 세션 기반 단기 관심 인코딩
- `serving/`
  - API 서버에서 사용할 추천 추론 로직
- `checkpoints/`
  - 학습된 모델 가중치 저장

## 전체 파이프라인

1. User / Item feature 구성
2. Two-Tower로 candidate item top-k 추출
3. Ranking 모델로 후보 재정렬
4. 필요 시 session feature 반영
5. 최종 추천 결과 반환

## 주요 목표
- Candidate Generation: Recall@300 개선
- Ranking: AUC 개선
- Serving: 빠른 inference 지원