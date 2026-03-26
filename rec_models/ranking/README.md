# Ranking

후보 상품들을 정교하게 재정렬하는 랭킹 모델을 구현한다.

## 목표
Candidate 단계에서 추출된 상품들 중 클릭 또는 구매 가능성이 높은 순으로 정렬한다.

## 주요 기능
- DeepFM 또는 Wide&Deep 모델 구현
- user/item/context/cross feature 처리
- CTR/CVR prediction
- AUC 평가
- inference score 계산

## 입력 예시
- user feature
- item feature
- context feature
- cross feature

## 출력
- item별 ranking score
- 재정렬된 추천 리스트