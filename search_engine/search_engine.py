from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

LOGGER = logging.getLogger(__name__)

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
DEFAULT_TOP_K = 10
DEFAULT_PORT = 8002


class _NumpyInnerProductIndex:
    # FAISS를 쓸 수 없는 환경에서만 사용하는 최소 기능 대체 인덱스다.
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of shape (n, {self.dimension})")
        self.vectors = np.vstack([self.vectors, vectors]) if len(self.vectors) else vectors.copy()

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        query_vec = np.asarray(query_vec, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if self.vectors.size == 0:
            empty_scores = np.empty((query_vec.shape[0], 0), dtype=np.float32)
            empty_indices = np.empty((query_vec.shape[0], 0), dtype=np.int64)
            return empty_scores, empty_indices

        scores = query_vec @ self.vectors.T
        k = min(int(top_k), self.vectors.shape[0])
        indices = np.argsort(-scores, axis=1)[:, :k]
        rows = np.arange(scores.shape[0])[:, None]
        top_scores = scores[rows, indices]
        return top_scores.astype(np.float32), indices.astype(np.int64)


@dataclass
class SearchItem:
    product_id: str
    name: str
    price: float
    description: str
    image: Optional[Image.Image]
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    item_id: str
    score: float
    metadata: Dict[str, Any]


def encode_image_file(path: Path | str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


class OpenAIClipEmbedder:
    # 텍스트/이미지를 OpenAI CLIP 공통 임베딩 공간으로 변환한다.
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
        fail_on_load_error: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fail_on_load_error = fail_on_load_error
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.dim: Optional[int] = None
        self._load_error: Optional[Exception] = None
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self.model is not None and self.processor is not None and self.dim is not None:
            return
        try:
            LOGGER.info("Loading CLIP model: %s", self.model_name)
            # 오프라인 캐시를 먼저 확인하고, 없을 때만 원격 다운로드를 시도한다.
            load_attempts = (
                {"local_files_only": True},
                {"local_files_only": False},
            )
            last_error: Optional[Exception] = None
            for kwargs in load_attempts:
                try:
                    self.processor = CLIPProcessor.from_pretrained(self.model_name, **kwargs)
                    self.model = CLIPModel.from_pretrained(self.model_name, **kwargs)
                    break
                except Exception as exc:
                    last_error = exc
                    self.processor = None
                    self.model = None
                    if kwargs.get("local_files_only"):
                        LOGGER.info("CLIP model not found in local cache, retrying with remote download enabled")
                    else:
                        raise
            if self.model is None or self.processor is None:
                raise last_error or RuntimeError(f"Unable to load CLIP model '{self.model_name}'")
            self.model.to(self.device)
            self.model.eval()
            self.dim = int(getattr(self.model.config, "projection_dim", 512))
            LOGGER.info("CLIP model ready on %s with dim=%d", self.device, self.dim)
        except Exception as exc:  # pragma: no cover
            self._load_error = exc
            message = (
                f"Failed to load OpenAI CLIP model '{self.model_name}'. "
                "This project requires a trained CLIP model and does not support an untrained fallback. "
                f"Original error: {exc}"
            )
            LOGGER.exception(message)
            if self.fail_on_load_error:
                raise RuntimeError(message) from exc

    def _require_components(self) -> Tuple[CLIPModel, CLIPProcessor, int]:
        self._ensure_loaded()
        if self.model is None or self.processor is None or self.dim is None:
            message = (
                f"OpenAI CLIP model '{self.model_name}' is unavailable. "
                "A trained CLIP checkpoint is required for this search engine."
            )
            if self._load_error is not None:
                message = f"{message} Last error: {self._load_error}"
            raise RuntimeError(message)
        return self.model, self.processor, self.dim

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm

    def embed_text(self, text: str) -> np.ndarray:
        model, processor, dim = self._require_components()
        value = (text or "").strip()
        if not value:
            return np.zeros(dim, dtype=np.float32)
        # text tower의 pooled output을 projection layer로 512차원 CLIP 공간에 투영한다.
        inputs = processor(text=[value], return_tensors="pt", padding=True, truncation=True)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
            pooled = text_outputs[1]
            features = model.text_projection(pooled)
            features = torch.nn.functional.normalize(features, dim=-1)
        return self._normalize(features[0].detach().cpu().numpy())

    def embed_image(self, image: Image.Image) -> np.ndarray:
        model, processor, _ = self._require_components()
        rgb_image = image.convert("RGB")
        # vision tower의 pooled output을 projection layer로 512차원 CLIP 공간에 투영한다.
        inputs = processor(images=rgb_image, return_tensors="pt")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        with torch.no_grad():
            vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_outputs[1]
            features = model.visual_projection(pooled)
            features = torch.nn.functional.normalize(features, dim=-1)
        return self._normalize(features[0].detach().cpu().numpy())

    def combine_embeddings(self, vectors: Sequence[np.ndarray]) -> np.ndarray:
        # hybrid 검색은 텍스트/이미지 벡터를 평균낸 뒤 다시 정규화한다.
        _, _, dim = self._require_components()
        usable = [self._normalize(vec) for vec in vectors if vec is not None and np.any(vec)]
        if not usable:
            return np.zeros(dim, dtype=np.float32)
        combined = np.mean(np.stack(usable).astype(np.float32), axis=0)
        return self._normalize(combined)

    def embed_item(self, text: str, image: Optional[Image.Image] = None) -> np.ndarray:
        vectors: List[np.ndarray] = []
        if text and text.strip():
            vectors.append(self.embed_text(text))
        if image is not None:
            vectors.append(self.embed_image(image))
        return self.combine_embeddings(vectors)

    def embed_query(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> Tuple[np.ndarray, str]:
        has_text = bool(text and text.strip())
        has_image = image is not None
        if has_text and has_image:
            return self.combine_embeddings([self.embed_text(text or ""), self.embed_image(image)]), "hybrid"
        if has_image:
            return self.embed_image(image), "image"
        return self.embed_text(text or ""), "text"

    def project_external_embedding(self, embedding: np.ndarray, modality: str = "text") -> np.ndarray:
        # app.py 등 외부 코드가 만든 임베딩도 CLIP 검색 공간 차원에 맞춰 흡수한다.
        model, _, dim = self._require_components()
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.size == 0:
            return np.zeros(dim, dtype=np.float32)

        if vector.ndim == 1 and vector.shape[0] == dim:
            return self._normalize(vector)

        if vector.ndim >= 2 and vector.shape[-1] == dim:
            flattened = vector.reshape(-1, dim)
            return self._normalize(flattened.mean(axis=0))

        if modality == "image":
            hidden_dim = int(getattr(model.config.vision_config, "hidden_size", dim))
            projection = model.visual_projection
        else:
            hidden_dim = int(getattr(model.config.text_config, "hidden_size", dim))
            projection = model.text_projection

        if vector.ndim == 1 and vector.shape[0] == hidden_dim:
            tensor = torch.from_numpy(vector).to(self.device).unsqueeze(0)
            with torch.no_grad():
                projected = projection(tensor)
                projected = torch.nn.functional.normalize(projected, dim=-1)
            return self._normalize(projected[0].detach().cpu().numpy())

        if vector.ndim >= 2 and vector.shape[-1] == hidden_dim:
            flattened = vector.reshape(-1, hidden_dim)
            pooled = flattened.mean(axis=0, dtype=np.float32)
            tensor = torch.from_numpy(pooled).to(self.device).unsqueeze(0)
            with torch.no_grad():
                projected = projection(tensor)
                projected = torch.nn.functional.normalize(projected, dim=-1)
            return self._normalize(projected[0].detach().cpu().numpy())

        flattened = vector.reshape(-1).astype(np.float32)
        if flattened.shape[0] > dim:
            flattened = flattened[:dim]
        elif flattened.shape[0] < dim:
            flattened = np.pad(flattened, (0, dim - flattened.shape[0]))
        return self._normalize(flattened)


class MultimodalSearchEngine:
    """OpenAI CLIP + FAISS(HNSW) based multimodal search engine."""

    def __init__(
        self,
        mode: str = "test",
        data_root: Optional[str] = None,
        top_k_default: int = DEFAULT_TOP_K,
        clip_model_name: str = CLIP_MODEL_NAME,
    ) -> None:
        self.mode = (mode or "test").lower().strip()
        self.data_root = self._resolve_data_root(data_root)
        self.top_k_default = int(top_k_default)
        self.embedder = OpenAIClipEmbedder(model_name=clip_model_name)
        self.items: List[SearchItem] = []
        self.item_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self.index: Any = None
        self.dimension = int(self.embedder.dim or 512)
        self._is_built = False

        if self.mode == "production":
            self.items = self._load_production_items()
        else:
            self.items = self._build_dummy_items()

        self._build_index()

    @staticmethod
    def _resolve_data_root(data_root: Optional[str]) -> Path:
        if data_root:
            return Path(data_root)

        file_dir = Path(__file__).resolve().parent
        project_root = file_dir.parent
        # docker-compose와 로컬 실행을 모두 지원하기 위해 후보 경로를 순서대로 확인한다.
        candidates = [
            os.getenv("DATA_ROOT"),
            file_dir / "data",
            project_root / "data",
            Path("/app/data"),
            Path("/app/data/processed"),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if (path / "articles.csv").exists():
                return path

        for candidate in candidates:
            if candidate:
                return Path(candidate)
        return project_root / "data"

    def _build_dummy_items(self) -> List[SearchItem]:
        # test 모드에서는 더미 이미지와 설명을 직접 만들어 즉시 검색 가능 상태로 만든다.
        palette = [
            (231, 76, 60),
            (52, 152, 219),
            (46, 204, 113),
            (155, 89, 182),
            (241, 196, 15),
            (230, 126, 34),
            (236, 240, 241),
            (52, 73, 94),
        ]
        samples = [
            ("100001", "Women Casual White Shirt", 29.9, "women apparel shirt white cotton casual"),
            ("100002", "Men Denim Jacket", 79.0, "men outerwear denim jacket blue casual"),
            ("100003", "Slim Fit Black Jeans", 49.5, "men bottoms black jeans slim fit"),
            ("100004", "Floral Summer Dress", 59.9, "women dress floral summer lightweight"),
            ("100005", "Kids Sports Sneakers", 39.9, "kids shoes sporty comfortable white"),
            ("100006", "Warm Wool Coat", 129.0, "women outerwear coat wool winter beige"),
            ("100007", "Canvas Shoulder Bag", 34.9, "accessories bag canvas casual neutral"),
            ("100008", "Striped Knit Sweater", 44.9, "women knit sweater striped warm casual"),
            ("100009", "Formal Navy Trousers", 54.9, "men trousers formal navy office"),
            ("100010", "Printed Long Sleeve Tee", 24.9, "men t-shirt printed long sleeve casual"),
            ("100011", "Pleated Midi Skirt", 39.9, "women skirt pleated midi elegant"),
            ("100012", "Running Shorts", 19.9, "men shorts sport running breathable"),
        ]

        items: List[SearchItem] = []
        for idx, (product_id, name, price, desc) in enumerate(samples):
            color = palette[idx % len(palette)]
            img = Image.new("RGB", (128, 128), color)
            draw = ImageDraw.Draw(img)
            draw.rectangle((16, 16, 112, 112), outline=(255, 255, 255), width=4)
            draw.text((14, 52), name[:12], fill=(255, 255, 255))
            items.append(
                SearchItem(
                    product_id=product_id,
                    name=name,
                    price=float(price),
                    description=desc,
                    image=img,
                    metadata={
                        "mode": "test",
                        "category": name.split()[0].lower(),
                        "name": name,
                        "description": desc,
                        "price": float(price),
                    },
                )
            )
        LOGGER.info("Prepared %d dummy items for test mode", len(items))
        return items

    def _load_production_items(self) -> List[SearchItem]:
        # production 모드에서는 H&M articles.csv를 읽어 상품 메타데이터를 구성한다.
        articles_path = self.data_root / "articles.csv"
        if not articles_path.exists():
            raise FileNotFoundError(f"articles.csv not found: {articles_path}")

        articles = pd.read_csv(articles_path).fillna("")
        price_map = self._load_article_price_map()

        items: List[SearchItem] = []
        for _, row in articles.iterrows():
            article_id = self._article_id(row)
            if not article_id:
                continue
            name = self._build_article_name(row)
            description = self._build_article_description(row)
            image = self._locate_article_image(row)
            price = float(price_map.get(article_id, 0.0))
            metadata = {
                "mode": "production",
                "article_id": article_id,
                "product_code": str(row.get("product_code", "")),
                "product_name": row.get("prod_name", row.get("product_name", name)),
                "product_type_name": row.get("product_type_name", ""),
                "graphical_appearance_name": row.get("graphical_appearance_name", ""),
                "colour_group_name": row.get("colour_group_name", ""),
                "perceived_colour_value_name": row.get("perceived_colour_value_name", ""),
                "index_name": row.get("index_name", ""),
                "department_name": row.get("department_name", ""),
                "section_name": row.get("section_name", ""),
                "garment_group_name": row.get("garment_group_name", ""),
                "detail_desc": row.get("detail_desc", ""),
                "image_name": row.get("image_name", ""),
                "price": price,
            }
            items.append(
                SearchItem(
                    product_id=article_id,
                    name=name,
                    price=price,
                    description=description,
                    image=image,
                    metadata=metadata,
                )
            )
        LOGGER.info("Prepared %d production items from %s", len(items), articles_path)
        return items

    def _load_article_price_map(self) -> Dict[str, float]:
        candidates = [
            self.data_root / "transactions_train.csv",
            self.data_root / "processed" / "transactions_train.csv",
            self.data_root / "train_data.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, usecols=["article_id", "price"]).dropna()
                df["article_id"] = df["article_id"].astype(str)
                grouped = df.groupby("article_id")["price"].mean()
                return {str(key): float(value) for key, value in grouped.items()}
            except Exception as exc:
                LOGGER.warning("Failed to load prices from %s: %s", path, exc)
        return {}

    @staticmethod
    def _article_id(row: pd.Series) -> str:
        for key in ("article_id", "product_code", "item_id"):
            value = str(row.get(key, "")).strip()
            if value:
                return value
        return ""

    @staticmethod
    def _build_article_name(row: pd.Series) -> str:
        candidates = [
            str(row.get("prod_name", "")).strip(),
            str(row.get("product_name", "")).strip(),
            str(row.get("product_type_name", "")).strip(),
            str(row.get("detail_desc", "")).strip(),
        ]
        for value in candidates:
            if value:
                return value[:120]
        return "item"

    @staticmethod
    def _build_article_description(row: pd.Series) -> str:
        # CLIP 텍스트 검색 품질을 위해 색상/카테고리/설명 필드를 하나의 문장으로 합친다.
        fields = [
            str(row.get("prod_name", "")).strip(),
            str(row.get("product_type_name", "")).strip(),
            str(row.get("graphical_appearance_name", "")).strip(),
            str(row.get("colour_group_name", "")).strip(),
            str(row.get("perceived_colour_value_name", "")).strip(),
            str(row.get("index_name", "")).strip(),
            str(row.get("department_name", "")).strip(),
            str(row.get("section_name", "")).strip(),
            str(row.get("garment_group_name", "")).strip(),
            str(row.get("detail_desc", "")).strip(),
        ]
        return " | ".join(field for field in fields if field)

    def _locate_article_image(self, row: pd.Series) -> Optional[Image.Image]:
        # 이미지가 있으면 텍스트와 함께 상품 임베딩에 반영하고, 없으면 텍스트만 사용한다.
        image_name = str(row.get("image_name", "")).strip()
        article_id = self._article_id(row)
        candidates: List[Path] = []

        if image_name:
            candidates.extend(
                [
                    self.data_root / "images" / image_name,
                    self.data_root / image_name,
                ]
            )

        if article_id:
            candidates.extend(
                [
                    self.data_root / "images" / f"{article_id}.jpg",
                    self.data_root / "images" / f"{article_id}.png",
                    self.data_root / "images" / article_id,
                    self.data_root / f"{article_id}.jpg",
                ]
            )
            try:
                padded = f"{int(float(article_id)):010d}.jpg"
                candidates.append(self.data_root / "images" / padded)
            except Exception:
                pass

        for path in candidates:
            if path.exists():
                image = encode_image_file(path)
                if image is not None:
                    return image
        return None

    def _build_index(self) -> None:
        # 엔진 내부 데이터셋으로부터 상품별 CLIP 임베딩을 생성해 기본 인덱스를 만든다.
        vectors: List[np.ndarray] = []
        for item in self.items:
            vectors.append(self.embedder.embed_item(text=item.description or item.name, image=item.image))

        if not vectors:
            raise ValueError("No items available to index")

        self._embeddings = np.vstack(vectors).astype(np.float32)
        self._normalize_matrix_inplace(self._embeddings)
        self.dimension = int(self._embeddings.shape[1])
        self.item_ids = [str(item.product_id) for item in self.items]

        if faiss is not None:
            # Inner Product + L2 normalize 조합이라 cosine similarity 검색처럼 동작한다.
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
            self.index.add(self._embeddings)
            LOGGER.info("Built FAISS HNSW index with %d items", len(self.items))
        else:
            self.index = _NumpyInnerProductIndex(self.dimension)
            self.index.add(self._embeddings)
            LOGGER.warning("FAISS unavailable, using NumPy fallback index with %d items", len(self.items))
        self._is_built = True

    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: Optional[Sequence[Any]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        # app.py가 외부에서 만든 임베딩을 넘기는 레거시 경로와의 호환용 빌더다.
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim < 2:
            raise ValueError("embeddings must be at least a 2D array")
        if vectors.shape[0] == 0:
            raise ValueError("embeddings must not be empty")

        metadata_list = list(metadatas) if metadatas is not None else [{} for _ in range(vectors.shape[0])]
        ids = list(item_ids) if item_ids is not None else list(range(vectors.shape[0]))
        if len(ids) != vectors.shape[0] or len(metadata_list) != vectors.shape[0]:
            raise ValueError("embeddings, item_ids, and metadatas must have the same length")

        normalized_rows: List[np.ndarray] = []
        for row, metadata in zip(vectors, metadata_list):
            modality = "image" if str((metadata or {}).get("search_type", "")).lower() == "image" else "text"
            normalized_rows.append(self.embedder.project_external_embedding(row, modality=modality))

        self._embeddings = np.vstack(normalized_rows).astype(np.float32)
        self._normalize_matrix_inplace(self._embeddings)
        self.dimension = int(self._embeddings.shape[1])

        self.items = []
        self.item_ids = []
        for idx, (item_id, metadata) in enumerate(zip(ids, metadata_list)):
            payload = dict(metadata or {})
            product_id = str(payload.get("product_id", item_id))
            name = str(payload.get("name", product_id))
            price = float(payload.get("price", 0.0))
            description = str(payload.get("description", payload.get("prod_name", name)))
            self.items.append(
                SearchItem(
                    product_id=product_id,
                    name=name,
                    price=price,
                    description=description,
                    image=None,
                    metadata=payload,
                )
            )
            self.item_ids.append(str(item_id))

        if faiss is not None:
            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64
            self.index.add(self._embeddings)
        else:
            self.index = _NumpyInnerProductIndex(self.dimension)
            self.index.add(self._embeddings)
        self._is_built = True
        LOGGER.info("Built compatibility index with %d items", len(self.items))

    @staticmethod
    def _normalize_matrix_inplace(matrix: np.ndarray) -> None:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms

    def _search_from_vector(self, query_vec: np.ndarray, top_k: int) -> List[SearchResult]:
        if self.index is None:
            raise RuntimeError("Search index is not initialized")

        # 인덱스와 쿼리를 모두 정규화해 inner product 검색이 안정적으로 되도록 한다.
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        if faiss is not None:
            faiss.normalize_L2(query_vec)
        else:
            self._normalize_matrix_inplace(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results: List[SearchResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            item_id = self.item_ids[idx] if idx < len(self.item_ids) else str(item.product_id)
            results.append(SearchResult(item_id=str(item_id), score=float(score), metadata=dict(item.metadata)))
        return results

    def _prepare_query_vector(self, query_vec: np.ndarray, modality: str) -> np.ndarray:
        # 외부 임베딩의 shape이 달라도 현재 인덱스 차원에 맞는 단일 쿼리 벡터로 변환한다.
        vector = np.asarray(query_vec, dtype=np.float32)
        if vector.size == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        if vector.ndim == 1 and vector.shape[0] == self.dimension:
            return self.embedder._normalize(vector)
        if vector.ndim >= 2 and vector.shape[-1] == self.dimension:
            flattened = vector.reshape(-1, self.dimension)
            return self.embedder._normalize(flattened.mean(axis=0))
        return self.embedder.project_external_embedding(vector, modality=modality)

    def search(
        self,
        query: Optional[str] = None,
        image: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
        query_type: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
        text_embedding: Optional[np.ndarray] = None,
        image_embedding: Optional[np.ndarray] = None,
    ) -> Any:
        # 1) 외부 임베딩 호환 모드: app.py가 직접 만든 벡터를 받아 검색
        if query_type is not None or embedding is not None or text_embedding is not None or image_embedding is not None:
            top_k = max(1, int(top_k or self.top_k_default))
            if query_type == "hybrid":
                query_vec = self.embedder.combine_embeddings(
                    [
                        self._prepare_query_vector(text_embedding, "text") if text_embedding is not None else None,
                        self._prepare_query_vector(image_embedding, "image") if image_embedding is not None else None,
                    ]
                )
            elif query_type == "image":
                source = image_embedding if image_embedding is not None else embedding
                query_vec = self._prepare_query_vector(source, "image")
            else:
                source = text_embedding if text_embedding is not None else embedding
                query_vec = self._prepare_query_vector(source, "text")

            if query_vec.size == 0 or not np.any(query_vec):
                return []
            return self._search_from_vector(query_vec, top_k)

        # 2) self-contained 모드: query/image를 받아 이 파일 내부에서 CLIP 임베딩까지 수행
        if self.index is None:
            raise RuntimeError("Search index is not initialized")

        top_k = max(1, int(top_k or self.top_k_default))
        query_vec, search_type = self.embedder.embed_query(text=query, image=image)
        if not np.any(query_vec):
            return {
                "search_type": search_type,
                "results": [],
                "latency_ms": 0.0,
                "total_count": 0,
            }

        started = time.perf_counter()
        vector_results = self._search_from_vector(query_vec, top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0

        results: List[Dict[str, Any]] = []
        for hit in vector_results:
            meta = hit.metadata or {}
            results.append(
                {
                    "product_id": str(meta.get("product_id", hit.item_id)),
                    "name": str(meta.get("name", "")),
                    "score": float(hit.score),
                    "price": float(meta.get("price", 0.0)),
                }
            )

        return {
            "search_type": search_type,
            "results": results,
            "latency_ms": round(latency_ms, 3),
            "total_count": len(results),
        }

    def __len__(self) -> int:
        return len(self.items)

    def save_index(self, index_path: str, metadata_path: str) -> None:
        if self.index is None:
            raise RuntimeError("Index not initialized")
        if faiss is not None:
            faiss.write_index(self.index, index_path)
        else:
            np.savez_compressed(index_path + ".npz", vectors=getattr(self.index, "vectors", None))

        payload = {
            "mode": self.mode,
            "dimension": self.dimension,
            "items": [
                {
                    "product_id": item.product_id,
                    "name": item.name,
                    "price": item.price,
                    "description": item.description,
                    "metadata": item.metadata,
                }
                for item in self.items
            ],
        }
        Path(metadata_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_from_artifacts(
        cls,
        index_path: str,
        metadata_path: str,
        mode: str = "production",
        clip_model_name: str = CLIP_MODEL_NAME,
    ) -> "MultimodalSearchEngine":
        obj = cls.__new__(cls)
        obj.mode = mode
        obj.data_root = cls._resolve_data_root(None)
        obj.top_k_default = DEFAULT_TOP_K
        obj.embedder = OpenAIClipEmbedder(model_name=clip_model_name)
        obj.dimension = int(obj.embedder.dim or 512)
        if faiss is not None:
            obj.index = faiss.read_index(index_path)
        else:
            data = np.load(index_path + ".npz")
            obj.index = _NumpyInnerProductIndex(int(data["vectors"].shape[1]))
            obj.index.add(data["vectors"])

        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        obj.items = [
            SearchItem(
                product_id=str(item.get("product_id", "")),
                name=str(item.get("name", "")),
                price=float(item.get("price", 0.0)),
                description=str(item.get("description", "")),
                image=None,
                metadata=dict(item.get("metadata", {})),
            )
            for item in meta.get("items", [])
        ]
        obj._embeddings = None
        obj._is_built = True
        obj.item_ids = [str(item.product_id) for item in obj.items]
        return obj

def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _sample_queries(mode: str) -> Iterable[str]:
    if mode == "production":
        return ("dress", "black jeans", "blue jacket")
    return ("white shirt", "running shorts", "canvas bag")


if __name__ == "__main__":
    _configure_logging()
    selected_mode = os.getenv("MODE", "test")
    LOGGER.info("Starting standalone search engine in %s mode", selected_mode)
    engine = MultimodalSearchEngine(mode=selected_mode)
    print(f"[search_engine] mode={selected_mode}")
    print(f"[search_engine] data_root={engine.data_root}")
    print(f"[search_engine] items={len(engine.items)}")
    print(f"[search_engine] dimension={engine.dimension}")
    for sample_query in _sample_queries(selected_mode):
        result = engine.search(query=sample_query, top_k=3)
        print(f"[search_engine] query={sample_query!r} -> {json.dumps(result, ensure_ascii=False)}")
