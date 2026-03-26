"""
세션 클릭 시퀀스 인코더 정의 위치.

역할:
- GRU 또는 Transformer 기반으로 최근 클릭 흐름을 encoding해 session embedding을 만든다.
- 사용자의 단기 관심사를 벡터 형태로 요약해 downstream 모듈에 전달한다.

작성 지침:
- 입력 sequence 길이, padding, masking 기준을 명확히 정한다.
- 장기 선호 feature와 혼동되지 않도록 session 표현의 책임을 분리한다.
- candidate와 ranking 어느 쪽에서 사용할지 고려해 출력 형식을 단순하게 유지한다.
"""
