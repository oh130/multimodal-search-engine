"""
서비스용 candidate 추론 계층 위치.

역할:
- 배포된 candidate 모델과 item index를 로드하고 top-k 후보를 반환한다.
- recommend 파이프라인에서 retrieval 전용 서비스 인터페이스를 제공한다.

작성 지침:
- 모델 로딩과 실제 추론 호출 책임을 분리해 테스트 가능하게 설계한다.
- 저장된 embedding 또는 검색 인덱스와의 호환성을 우선 보장한다.
- 반환 형식은 ranking 단계에서 바로 사용할 수 있게 단순해야 한다.
"""
