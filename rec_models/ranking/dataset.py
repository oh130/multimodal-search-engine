"""
랭킹 모델 학습용 dataset 정의 위치.

역할:
- candidate 결과와 user/item/context feature를 묶어 학습 샘플을 만든다.
- 클릭 또는 구매 label 기준으로 ranking 입력 형식을 구성한다.

작성 지침:
- candidate 단계 출력과 ranking 입력 사이의 연결 규칙을 명확히 유지한다.
- label 생성 기준과 negative 샘플 포함 방식을 문서화 가능한 구조로 설계한다.
- feature join 책임과 모델 입력 변환 책임을 섞지 않도록 구분한다.
"""
