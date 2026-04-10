"""
Search Engine API.

Endpoints:
  POST /search   Text, image, or hybrid search
  GET  /health
"""

from __future__ import annotations

import base64
import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from search_engine import MultimodalSearchEngine


LOGGER = logging.getLogger(__name__)

DEFAULT_MODE = os.getenv("SEARCH_ENGINE_MODE") or os.getenv("MODE", "test")
DEFAULT_DATA_ROOT = os.getenv("DATA_ROOT", "/app/data/raw")


@dataclass
class SearchEngineState:
    engine: MultimodalSearchEngine | None = None
    initializing: bool = False
    init_error: str | None = None


state = SearchEngineState()


class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)


def _decode_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:  # pragma: no cover - FastAPI request path
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {exc}") from exc


def _initialize_search_engine() -> None:
    data_root = Path(DEFAULT_DATA_ROOT)
    LOGGER.info("Initializing search engine mode=%s data_root=%s", DEFAULT_MODE, data_root)
    try:
        engine = MultimodalSearchEngine(
            mode=DEFAULT_MODE,
            data_root=str(data_root),
        )
    except Exception as exc:
        LOGGER.exception("Search engine initialization failed")
        state.init_error = str(exc)
        state.engine = None
    else:
        state.engine = engine
        state.init_error = None
        LOGGER.info("Search engine ready items=%s dimension=%s", len(engine.items), engine.dimension)
    finally:
        state.initializing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app

    state.initializing = True
    state.engine = None
    state.init_error = None
    threading.Thread(target=_initialize_search_engine, name="search-engine-init", daemon=True).start()
    yield


app = FastAPI(title="Search Engine", lifespan=lifespan)


@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    query = req.query.strip()
    image = _decode_image(req.image_base64) if req.image_base64 else None

    if not query and image is None:
        raise HTTPException(status_code=400, detail="query 또는 image_base64 중 하나는 필요합니다.")

    if state.init_error is not None:
        raise HTTPException(status_code=500, detail=f"검색 엔진 초기화 실패: {state.init_error}")

    if state.engine is None:
        raise HTTPException(status_code=503, detail="검색 엔진 초기화 중입니다. 잠시 후 다시 시도하세요.")

    try:
        return state.engine.search(
            query=query or None,
            image=image,
            top_k=req.top_k,
        )
    except Exception as exc:  # pragma: no cover - FastAPI request path
        LOGGER.exception("Search request failed")
        raise HTTPException(status_code=500, detail=f"검색 처리 실패: {exc}") from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    if state.init_error is not None:
        return {
            "status": "error",
            "mode": DEFAULT_MODE,
            "data_root": DEFAULT_DATA_ROOT,
            "error": state.init_error,
        }

    if state.engine is None:
        return {
            "status": "initializing",
            "mode": DEFAULT_MODE,
            "data_root": DEFAULT_DATA_ROOT,
        }

    return {
        "status": "ok",
        "mode": state.engine.mode,
        "data_root": str(state.engine.data_root),
        "index_size": len(state.engine.items),
        "dimension": state.engine.dimension,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)
