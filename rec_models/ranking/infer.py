"""
랭킹 점수 추론 로직 위치.

역할:
- candidate item 리스트를 입력받아 각 item의 ranking score를 계산한다.
- serving 계층에서 재정렬 직전에 호출되는 추론 인터페이스가 된다.

작성 지침:
- 추론 입력에 필요한 feature가 무엇인지 계약을 명확히 유지한다.
- 반환값은 item id와 score를 함께 다루기 쉬운 형태여야 한다.
- batch scoring과 단건 scoring 모두 확장 가능한 구조를 염두에 둔다.
"""
