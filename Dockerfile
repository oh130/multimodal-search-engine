# Python 베이스 이미지
FROM python:3.10-slim

# 환경 변수 설정 (파이썬 최적화)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 작업 디렉터리
WORKDIR /app

# 시스템 패키지 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements 먼저 복사 (캐시 활용)
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 나머지 코드 복사
COPY . .

# 기본 실행 (검색 엔진 실행)
CMD ["python", "search_engine.py"]