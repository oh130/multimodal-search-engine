"""
Two-Tower retrieval 평가 코드 위치.

역할:
- Recall@K 등 retrieval 성능을 측정한다.
- candidate 품질을 ranking 이전 단계에서 독립적으로 검증한다.

작성 지침:
- 평가 데이터셋과 학습 데이터셋의 역할을 섞지 않는다.
- top-k 기준, 정답 정의, 배치 평가 방식이 명확해야 한다.
- 공통 지표 구현은 가능하면 common.metrics를 재사용하는 방향으로 맞춘다.
"""
