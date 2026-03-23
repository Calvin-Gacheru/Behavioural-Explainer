"""
vector_store.py — Qdrant vector store lifecycle manager.

Responsibilities:
  1. Bootstrap a Qdrant collection with the correct schema and cosine distance.
  2. Build a LlamaIndex VectorStoreIndex backed by that collection.
  3. Upsert new nodes (with idempotency — same chunk ID won't be duplicated).
  4. Expose a retriever for downstream query engines.

Design principle: this class owns the storage layer. It knows about embeddings
(because it needs to build the index), but nothing about prompts or the LLM.
"""

from __future__ import annotations

from typing import Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

# BGE-small produces 384-dim vectors; nomic-embed-text produces 768-dim.
# We auto-probe the embedding model at collection creation time.
_EMBED_DIM_MAP: dict[str, int] = {
    "bge-small": 384,
    "bge-small:en": 384,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def _resolve_embed_dim(model_name: str) -> int:
    """
    Return the known embedding dimension for a model name.
    Falls back to 768 (safe default for most Ollama embedding models).
    """
    for key, dim in _EMBED_DIM_MAP.items():
        if model_name.lower().startswith(key):
            return dim
    logger.warning(
        f"Unknown embed model '{model_name}' — defaulting to dim=768. "
        "Set the correct dimension in _EMBED_DIM_MAP if retrieval quality suffers."
    )
    return 768


class VectorStoreManager:
    """
    Manages a Qdrant collection and a LlamaIndex VectorStoreIndex on top of it.

    Usage
    -----
    >>> manager = VectorStoreManager()
    >>> manager.bootstrap()                   # create collection if not exists
    >>> manager.ingest(nodes)                 # upsert TextNode list
    >>> retriever = manager.get_retriever()   # use in query engine
    """

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection: str = settings.qdrant_collection,
        embed_model_name: str = settings.ollama_embed_model,
        ollama_base_url: str = settings.ollama_base_url,
        top_k: int = settings.top_k_retrieval,
    ) -> None:
        self.collection = collection
        self.top_k = top_k
        self.embed_dim = _resolve_embed_dim(embed_model_name)

        # ── Qdrant client ─────────────────────────────────────────────────────
        self._client = QdrantClient(host=host, port=port, timeout=30)

        # ── Embedding model (Ollama) ──────────────────────────────────────────
        self._embed_model: BaseEmbedding = OllamaEmbedding(
            model_name=embed_model_name,
            base_url=ollama_base_url,
        )

        # ── LlamaIndex vector store wrapper ───────────────────────────────────
        self._vector_store = QdrantVectorStore(
            client=self._client,
            collection_name=self.collection,
        )

        # Populated after bootstrap()
        self._index: Optional[VectorStoreIndex] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """
        Create the Qdrant collection if it doesn't exist yet.
        Sets up cosine similarity and payload indexes for metadata filtering.
        Safe to call on every startup — idempotent.
        """
        existing = {c.name for c in self._client.get_collections().collections}

        if self.collection not in existing:
            logger.info(
                f"Creating Qdrant collection '{self.collection}' "
                f"(dim={self.embed_dim}, metric=Cosine)"
            )
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embed_dim,
                    distance=qdrant_models.Distance.COSINE,
                    # HNSW index tuning: higher m = better recall, more RAM
                    hnsw_config=qdrant_models.HnswConfigDiff(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10_000,
                    ),
                ),
                optimizers_config=qdrant_models.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )

            # Create payload indexes so we can filter by metadata efficiently
            self._create_payload_indexes()
            logger.success(f"Collection '{self.collection}' created.")
        else:
            logger.info(f"Collection '{self.collection}' already exists — skipping creation.")

        # Build (or reconnect to) the LlamaIndex index
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def ingest(self, nodes: list[TextNode]) -> None:
        """
        Embed and upsert a list of TextNodes into Qdrant.
        LlamaIndex handles batching and deduplication by node ID.
        """
        if not nodes:
            logger.warning("ingest() called with empty node list — nothing to do.")
            return

        if self._index is None:
            raise RuntimeError("Call bootstrap() before ingest().")

        logger.info(f"Embedding and upserting {len(nodes)} nodes → Qdrant...")

        # This triggers embedding via OllamaEmbedding and upserts to Qdrant
        for i in range(0, len(nodes), 64):      # batch in groups of 64
            batch = nodes[i : i + 64]
            self._index.insert_nodes(batch)
            logger.debug(f"  Upserted batch {i // 64 + 1} ({len(batch)} nodes)")

        collection_info = self._client.get_collection(self.collection)
        total_vectors = collection_info.vectors_count
        logger.success(
            f"Ingestion complete. Collection '{self.collection}' now has "
            f"{total_vectors} vector(s)."
        )

    def get_retriever(self, top_k: Optional[int] = None):
        """Return a configured VectorIndexRetriever for use in a query engine."""
        if self._index is None:
            raise RuntimeError("Call bootstrap() before get_retriever().")

        return self._index.as_retriever(
            similarity_top_k=top_k or self.top_k,
            # Optional: add metadata filters here if you want theory-scoped retrieval
            # filters=MetadataFilters(filters=[...])
        )

    def get_index(self) -> VectorStoreIndex:
        """Expose the underlying index for advanced use."""
        if self._index is None:
            raise RuntimeError("Call bootstrap() before get_index().")
        return self._index

    def collection_stats(self) -> dict:
        """Return collection stats for health checks and monitoring."""
        info = self._client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _create_payload_indexes(self) -> None:
        """
        Create Qdrant payload indexes on the metadata fields we care about.
        This makes metadata-filtered retrieval (e.g., by theory_category) fast.
        """
        fields = [
            ("source_document", qdrant_models.PayloadSchemaType.KEYWORD),
            ("theory_category", qdrant_models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in fields:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug(f"  Payload index created: {field_name}")
            except UnexpectedResponse as exc:
                # Index already exists — safe to ignore
                logger.debug(f"  Payload index '{field_name}' already exists: {exc}")
