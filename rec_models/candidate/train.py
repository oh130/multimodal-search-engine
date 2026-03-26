"""
Two-Tower 모델 학습 스크립트 위치.

역할:
- dataset 로드, loss 계산, optimizer step, checkpoint 저장 흐름을 담당한다.
- retrieval 모델 학습 실험의 메인 진입점이 된다.

작성 지침:
- 데이터 로딩, 모델 초기화, 학습 루프, 평가, 저장 단계를 분리해서 구성한다.
- 로그에는 최소한 loss와 retrieval 관련 validation 지표가 남아야 한다.
- 학습 코드가 서비스 추론 코드에 직접 의존하지 않도록 분리한다.
"""
