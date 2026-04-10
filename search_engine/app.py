Search Engine API — port 8002

엔드포인트:
  POST /search   텍스트/이미지/hybrid 검색
  GET  /health
"""

from __future__ import annotations

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

from search_engine import MultimodalSearchEngine

LOGGER = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512
DATA_PATH = Path("/app/data/processed/articles_feature.csv")

# 전역 객체
clip_model: CLIPModel
clip_processor: CLIPProcessor
search_engine: MultimodalSearchEngine
product_metadata: dict[str, dict[str, Any]] = {}


def _load_clip() -> tuple[CLIPModel, CLIPProcessor]:
    LOGGER.info("CLIP 모델 로딩 중: %s", CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model.eval()
    LOGGER.info("CLIP 모델 로딩 완료")
    return model, processor


def _build_index(engine: MultimodalSearchEngine) -> dict[str, dict[str, Any]]:
    """articles_feature.csv로 FAISS 인덱스 빌드. 데이터 없으면 빈 인덱스로 시작."""
    if not DATA_PATH.exists():
        LOGGER.warning("상품 데이터 없음: %s — 빈 인덱스로 시작", DATA_PATH)
        return {}

    df = pd.read_csv(DATA_PATH, dtype=str).fillna("")
    if df.empty:
        LOGGER.warning("상품 데이터가 비어있음 — 빈 인덱스로 시작")
        return {}

    LOGGER.info("상품 %d건 임베딩 시작", len(df))
    embeddings = []
    metadata = {}

    for _, row in df.iterrows():
        article_id = str(row.get("article_id", "")).strip()
        if not article_id:
            continue

        text = " ".join(filter(None, [
            row.get("prod_name", ""),
            row.get("product_type_name", ""),
            row.get("colour_group_name", ""),
            row.get("department_name", ""),
        ]))

        inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
            emb = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            emb = torch.nn.functional.normalize(emb, dim=-1)

        embeddings.append(emb.squeeze().numpy())
        metadata[len(embeddings) - 1] = {
            "article_id": article_id,
            "product_id": article_id,
            "name": row.get("prod_name", ""),
            "price": 0.0,
            "category": row.get("category", ""),
            "main_category": row.get("main_category", ""),
            "color": row.get("colour_group_name", ""),
        }

    if not embeddings:
        return {}

    engine.build_index(
        embeddings=np.array(embeddings, dtype=np.float32),
        item_ids=list(range(len(embeddings))),
        metadatas=list(metadata.values()),
    )
    LOGGER.info("FAISS 인덱스 빌드 완료: %d건", len(embeddings))
    return {str(v["product_id"]): v for v in metadata.values()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clip_model, clip_processor, search_engine, product_metadata
    clip_model, clip_processor = _load_clip()
    search_engine = MultimodalSearchEngine("test")
    product_metadata = _build_index(search_engine)
    yield


app = FastAPI(title="Search Engine", lifespan=lifespan)


# ── 요청/응답 스키마 ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = ""
    image_base64: str | None = None
    top_k: int = 10


# ── 엔드포인트 ────────────────────────────────────────────────

@app.post("/search")
async def search(req: SearchRequest) -> dict[str, Any]:
    start = time.perf_counter()
    has_text = bool(req.query.strip())
    has_image = bool(req.image_base64)

    if not has_text and not has_image:
        raise HTTPException(status_code=400, detail="query 또는 image_base64 중 하나는 필요합니다.")

    text_emb = None
    image_emb = None

    if has_text:
        inputs = clip_processor(text=[req.query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
            emb = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
            text_emb = torch.nn.functional.normalize(emb, dim=-1).squeeze().numpy()

    if has_image:
        try:
            image_bytes = base64.b64decode(req.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = clip_model.get_image_features(**inputs)
                emb = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                image_emb = torch.nn.functional.normalize(emb, dim=-1).squeeze().numpy()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    if has_text and has_image:
        search_type = "hybrid"
    elif has_image:
        search_type = "image"
    else:
        search_type = "text"

    if not search_engine._is_built:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "search_type": search_type,
            "results": [],
            "latency_ms": round(latency_ms, 2),
            "total_count": 0,
        }

    raw_results = search_engine.search(
        query_type=search_type,
        embedding=text_emb if search_type == "text" else (image_emb if search_type == "image" else None),
        text_embedding=text_emb if search_type == "hybrid" else None,
        image_embedding=image_emb if search_type == "hybrid" else None,
        top_k=req.top_k,
    )

    results = []
    for r in raw_results:
        meta = r.metadata or {}
        results.append({
            "product_id": str(meta.get("product_id", r.item_id)),
            "name": meta.get("name", ""),
            "score": round(r.score, 4),
            "price": float(meta.get("price", 0.0)),
        })

    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "search_type": search_type,
        "results": results,
        "latency_ms": round(latency_ms, 2),
        "total_count": len(results),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_size": len(search_engine) if search_engine._is_built else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=False)
