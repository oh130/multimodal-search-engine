# pront

`pront`는 멀티모달 검색 엔진 프로젝트의 프론트엔드 UI입니다.  
React, TypeScript, Vite 기반으로 구성되어 있으며, 현재는 `api_gateway`를 통해 백엔드와 연결되도록 맞춰져 있습니다.

## 사용 기술

- React 18
- TypeScript
- Vite

## 실행 방법

의존성을 설치한 뒤 개발 서버를 실행합니다.

```bash
npm install
npm run dev
```

기본 개발 서버 주소:

- 프론트엔드: `http://localhost:5173`

## 백엔드 연결 방식

프론트엔드는 API Gateway와 연결되도록 설정되어 있습니다.

- API Gateway: `http://localhost:8000`

개발 환경에서는 [vite.config.ts](C:\Users\손석범\Documents\GitHub\multimodal-search-engine\pront\vite.config.ts) 에서 `/api` 요청을 `http://localhost:8000` 으로 프록시합니다.

즉, 프론트에서는 아래 경로를 그대로 호출하면 됩니다.

- `POST /api/search`
- `GET /api/recommend`
- `POST /api/events`

## 환경 변수

Vite 프록시 대신 직접 백엔드 주소를 지정하고 싶다면 아래 환경 변수를 사용할 수 있습니다.

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## 주요 파일

- [src/App.tsx](C:\Users\손석범\Documents\GitHub\multimodal-search-engine\pront\src\App.tsx): 메인 UI
- [src/api.ts](C:\Users\손석범\Documents\GitHub\multimodal-search-engine\pront\src\api.ts): API 호출 및 응답 정규화
- [vite.config.ts](C:\Users\손석범\Documents\GitHub\multimodal-search-engine\pront\vite.config.ts): Vite 설정 및 개발 프록시

## 주요 기능

- 텍스트 / 이미지 / 멀티모달 검색 UI
- 추천 결과 패널
- 추천 아이템 클릭 시 백엔드 이벤트 전송
- 백엔드 응답이 없을 때를 위한 프론트 fallback 표시

## 참고 사항

- 프론트엔드를 사용하기 전에 API Gateway가 먼저 실행 중이어야 합니다.
- 이미지 검색 시 업로드한 이미지는 브라우저에서 base64로 변환된 뒤 백엔드로 전달됩니다.
- 현재 일부 UI 한글 문구는 인코딩 문제로 깨져 있을 수 있으며, 별도로 정리할 필요가 있습니다.
