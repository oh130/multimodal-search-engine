"""
Two-Tower 모델 정의 위치.

역할:
- user tower와 item tower를 각각 구성하고 embedding을 생성한다.
- 사용자 표현과 상품 표현을 같은 retrieval 공간에 정렬하는 구조를 담당한다.

작성 지침:
- tower 입력 feature 범위와 embedding 출력 차원을 명확히 유지한다.
- user tower와 item tower는 독립적으로 교체 가능하게 설계한다.
- similarity 계산 방식은 infer 단계와 동일한 가정을 쓰도록 맞춘다.
"""
