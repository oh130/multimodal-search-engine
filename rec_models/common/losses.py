"""
추천 모델 학습용 loss 함수 정의 위치.

역할:
- BCE loss, pairwise loss, contrastive-style loss 등 학습 목적 함수를 관리한다.
- candidate와 ranking 학습 코드가 공통 규칙 아래에서 loss를 선택할 수 있게 한다.

작성 지침:
- loss 선택 기준은 모델 목적과 label 구조에 맞게 분리한다.
- sampling 전략이나 margin 같은 부가 파라미터는 함수 인터페이스에서 드러나야 한다.
- 실험용 임시 loss를 추가하더라도 이름과 목적이 분명해야 한다.
"""
