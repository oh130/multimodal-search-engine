"""
랭킹 모델 정의 위치.

역할:
- DeepFM 또는 Wide&Deep 기반으로 feature interaction과 dense representation을 처리한다.
- user/item/context/cross feature를 조합해 ranking score를 계산하는 중심 모듈이다.

작성 지침:
- wide 파트와 deep 파트의 feature 책임을 분리해서 설계한다.
- session feature가 들어오더라도 기존 입력 계약을 깨지 않도록 확장한다.
- score 출력이 CTR, CVR, 또는 통합 score 중 무엇인지 명확히 정한다.
"""
