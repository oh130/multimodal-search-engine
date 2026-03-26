"""
추천 전체 파이프라인 진입점 위치.

역할:
- candidate retrieval, ranking inference, session feature 결합을 거쳐 최종 추천 리스트를 반환한다.
- 서비스 환경에서 호출하는 최상위 recommend 함수 책임을 가진다.

작성 지침:
- 학습 코드에 직접 의존하지 않고 저장된 모델 또는 서비스 객체만 사용한다.
- 단계별 입출력을 명확히 분리해 디버깅 가능한 구조로 유지한다.
- timeout, fallback, 빈 결과 처리 같은 서비스 관점 예외를 고려한다.
"""
