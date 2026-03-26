"""
Two-Tower 기반 candidate retrieval 추론 정의 위치.

역할:
- 학습된 user tower로 user embedding을 생성하고 top-k candidate를 찾는다.
- 서비스 계층이나 offline 평가에서 retrieval 진입점으로 사용된다.

작성 지침:
- 추론 입력 형식은 serving 계층에서 그대로 호출할 수 있게 단순하게 유지한다.
- item embedding 저장 포맷과 검색 인덱스 사용 방식이 호환되도록 맞춘다.
- 반환값에는 최소한 candidate item id와 score 기준이 드러나야 한다.
"""
