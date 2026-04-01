from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import json
import os

import faiss
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], List[float]]


@dataclass
class SearchResult:
    item_id: int
    score: float
    metadata: Optional[Dict[str, Any]] = None


class MultimodalHNSWSearchEngine:
    """FAISS HNSW based multimodal search engine.

    Assumptions
    -----------
    - Embeddings are already computed before calling this class.
    - Text/image embeddings live in the same vector space (e.g. CLIP space).
    - All vectors are normalized to unit length before indexing/searching.
      This lets us use HNSW (L2) and interpret the ranking as cosine similarity.

    Notes
    -----
    - HNSW is a strong choice for low-latency retrieval when the full index fits in RAM.
    - To keep latency low, keep the index in memory, use float32, and tune efSearch.
    - 200ms is a system target, not a hard guarantee; it depends on hardware,
      vector count, top_k, and efSearch.
    """

    def __init__(
        self,
        dim: int,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        normalize_embeddings: bool = True,
        num_threads: Optional[int] = None,
    ) -> None:
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.normalize_embeddings = normalize_embeddings

        if num_threads is not None:
            faiss.omp_set_num_threads(num_threads)

        base_index = faiss.IndexHNSWFlat(dim, m)
        base_index.hnsw.efConstruction = ef_construction
        base_index.hnsw.efSearch = ef_search

        # Wrap with ID map so we can use stable item IDs.
        self.index = faiss.IndexIDMap2(base_index)
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self._is_built = False

    # --------------------------
    # Internal helpers
    # --------------------------
    def _to_2d_float32(self, vectors: ArrayLike) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D array, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"Expected vector dim {self.dim}, got {arr.shape[1]}")
        return np.ascontiguousarray(arr)

    def _normalize_if_needed(self, vectors: np.ndarray) -> np.ndarray:
        if self.normalize_embeddings:
            faiss.normalize_L2(vectors)
        return vectors

    def _set_search_params(self, ef_search: Optional[int] = None) -> None:
        if ef_search is None:
            ef_search = self.ef_search
        # The actual underlying HNSW index lives inside IndexIDMap2.
        base = faiss.downcast_index(self.index.index)
        base.hnsw.efSearch = int(ef_search)

    @staticmethod
    def _squared_l2_to_cosine_sim(distances: np.ndarray) -> np.ndarray:
        # For unit-normalized vectors:
        # ||a - b||^2 = 2 - 2cos(a,b)  => cos = 1 - d/2
        return 1.0 - 0.5 * distances

    # --------------------------
    # Index build / update
    # --------------------------
    def add_items(
        self,
        embeddings: ArrayLike,
        item_ids: Optional[Sequence[int]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        vectors = self._to_2d_float32(embeddings)
        vectors = self._normalize_if_needed(vectors)

        n = vectors.shape[0]
        if item_ids is None:
            start_id = 0 if not self._is_built else self._next_id()
            item_ids = list(range(start_id, start_id + n))
        else:
            if len(item_ids) != n:
                raise ValueError("Length of item_ids must match number of embeddings")
            item_ids = list(map(int, item_ids))

        ids = np.asarray(item_ids, dtype=np.int64)
        self.index.add_with_ids(vectors, ids)

        if metadatas is not None:
            if len(metadatas) != n:
                raise ValueError("Length of metadatas must match number of embeddings")
            for item_id, meta in zip(ids.tolist(), metadatas):
                self.metadata[item_id] = dict(meta)

        self._is_built = True

    def build_index(
        self,
        embeddings: ArrayLike,
        item_ids: Optional[Sequence[int]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        # Reinitialize to ensure a clean build.
        self.clear()
        self.add_items(embeddings, item_ids=item_ids, metadatas=metadatas)

    def clear(self) -> None:
        base_index = faiss.IndexHNSWFlat(self.dim, self.m)
        base_index.hnsw.efConstruction = self.ef_construction
        base_index.hnsw.efSearch = self.ef_search
        self.index = faiss.IndexIDMap2(base_index)
        self.metadata = {}
        self._is_built = False

    def _next_id(self) -> int:
        # Simple monotonic id assignment when caller does not provide IDs.
        if len(self.metadata) == 0:
            return 0
        return max(self.metadata.keys()) + 1

    # --------------------------
    # Search API
    # --------------------------
    def search_by_embedding(
        self,
        query_embedding: ArrayLike,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        if not self._is_built:
            raise RuntimeError("Index is empty. Call build_index() or add_items() first.")

        q = self._to_2d_float32(query_embedding)
        q = self._normalize_if_needed(q)
        self._set_search_params(ef_search)

        distances, ids = self.index.search(q, top_k)
        distances = distances[0]
        ids = ids[0]

        scores = self._squared_l2_to_cosine_sim(distances)
        results: List[SearchResult] = []
        for item_id, score in zip(ids.tolist(), scores.tolist()):
            if item_id == -1:
                continue
            results.append(
                SearchResult(
                    item_id=int(item_id),
                    score=float(score),
                    metadata=self.metadata.get(int(item_id)),
                )
            )
        return results

    def search_text(
        self,
        text_embedding: ArrayLike,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """Text query embedding -> retrieve similar items."""
        return self.search_by_embedding(text_embedding, top_k=top_k, ef_search=ef_search)

    def search_image(
        self,
        image_embedding: ArrayLike,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """Image query embedding -> retrieve similar items."""
        return self.search_by_embedding(image_embedding, top_k=top_k, ef_search=ef_search)

    def search_hybrid(
        self,
        text_embedding: Optional[ArrayLike] = None,
        image_embedding: Optional[ArrayLike] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
        top_k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[SearchResult]:
        """Hybrid query = weighted fusion of text + image query embeddings.

        This is the fastest hybrid strategy because it performs only one ANN search.
        """
        if text_embedding is None and image_embedding is None:
            raise ValueError("At least one of text_embedding or image_embedding must be provided")

        if text_embedding is not None and image_embedding is not None:
            t = self._to_2d_float32(text_embedding)
            i = self._to_2d_float32(image_embedding)
            if t.shape[0] != 1 or i.shape[0] != 1:
                raise ValueError("Hybrid search expects a single text vector and a single image vector")
            q = text_weight * t + image_weight * i
        elif text_embedding is not None:
            q = self._to_2d_float32(text_embedding)
        else:
            q = self._to_2d_float32(image_embedding)

        return self.search_by_embedding(q, top_k=top_k, ef_search=ef_search)

    def search(
        self,
        query_type: str,
        embedding: Optional[ArrayLike] = None,
        text_embedding: Optional[ArrayLike] = None,
        image_embedding: Optional[ArrayLike] = None,
        top_k: int = 10,
        ef_search: Optional[int] = None,
        text_weight: float = 0.5,
        image_weight: float = 0.5,
    ) -> List[SearchResult]:
        """Generic API for text / image / hybrid queries.

        query_type:
            - 'text'
            - 'image'
            - 'hybrid'
        """
        query_type = query_type.lower().strip()
        if query_type == "text":
            if embedding is None:
                raise ValueError("For text search, provide embedding=...")
            return self.search_text(embedding, top_k=top_k, ef_search=ef_search)
        if query_type == "image":
            if embedding is None:
                raise ValueError("For image search, provide embedding=...")
            return self.search_image(embedding, top_k=top_k, ef_search=ef_search)
        if query_type == "hybrid":
            return self.search_hybrid(
                text_embedding=text_embedding,
                image_embedding=image_embedding,
                text_weight=text_weight,
                image_weight=image_weight,
                top_k=top_k,
                ef_search=ef_search,
            )
        raise ValueError("query_type must be one of: 'text', 'image', 'hybrid'")

    # --------------------------
    # Persistence
    # --------------------------
    def save(self, index_path: str, metadata_path: str) -> None:
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        self.index = faiss.read_index(index_path)
        self._is_built = True
        if metadata_path is not None and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # JSON object keys are strings; convert back to int.
            self.metadata = {int(k): v for k, v in raw.items()}
        else:
            self.metadata = {}

    # --------------------------
    # Utilities
    # --------------------------
    def __len__(self) -> int:
        return self.index.ntotal

    def info(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "normalize_embeddings": self.normalize_embeddings,
            "ntotal": len(self),
        }


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example: precomputed CLIP embeddings
    dim = 512
    engine = MultimodalHNSWSearchEngine(
        dim=dim,
        m=32,
        ef_construction=200,
        ef_search=64,
        normalize_embeddings=True,
        num_threads=4,
    )

    # Dummy database embeddings (already computed elsewhere)
    db_embeddings = np.random.randn(10000, dim).astype(np.float32)
    db_embeddings /= np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-12

    db_ids = list(range(1000))
    db_meta = [{"modality": "image", "path": f"img_{i}.jpg"} for i in range(1000)]
    engine.build_index(db_embeddings, item_ids=db_ids, metadatas=db_meta)

    # Text query embedding (already computed elsewhere)
    text_q = np.random.randn(dim).astype(np.float32)
    text_q /= np.linalg.norm(text_q) + 1e-12

    # Image query embedding (already computed elsewhere)
    image_q = np.random.randn(dim).astype(np.float32)
    image_q /= np.linalg.norm(image_q) + 1e-12

    text_results = engine.search_text(text_q, top_k=10)
    image_results = engine.search_image(image_q, top_k=10)
    hybrid_results = engine.search_hybrid(text_embedding=text_q, image_embedding=image_q, top_k=10)

    print("TEXT RESULTS")
    for r in text_results:
        print(r)

    print("\nIMAGE RESULTS")
    for r in image_results:
        print(r)

    print("\nHYBRID RESULTS")
    for r in hybrid_results:
        print(r)
