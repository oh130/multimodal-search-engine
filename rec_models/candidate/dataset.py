"""
Two-Tower 학습용 dataset 정의 위치.

역할:
- user feature, positive item, negative item 샘플을 구성하는 기준을 담당한다.
- retrieval 학습에 필요한 배치 입력 형식을 일관되게 만든다.

작성 지침:
- positive/negative 샘플 생성 규칙을 명확히 분리한다.
- feature schema와 맞지 않는 전처리는 이 파일 안에서 임의로 확장하지 않는다.
- 학습/검증 데이터 분리 기준과 샘플링 전략이 드러나도록 설계한다.
"""
