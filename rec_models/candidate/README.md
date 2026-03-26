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