"""
document_processor.py — Ingestion pipeline for psychology PDFs.

Responsibilities:
  1. Recursively discover all PDFs under a directory.
  2. Extract text with a fast PyMuPDF-backed reader.
  3. Enrich node metadata (source file, theory category guessed from filename).
  4. Chunk documents with SentenceSplitter (recursive character strategy).
  5. Return a flat list of TextNode objects ready for embedding.

Design principle: this class knows nothing about vector stores or LLMs.
It only transforms files → structured nodes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings


# ── Theory category heuristic ─────────────────────────────────────────────────
# Maps filename keywords → a human-readable psychological theory category.
# Extend this dict as you add more PDFs to your corpus.
THEORY_KEYWORD_MAP: dict[str, str] = {
    "attachment": "Attachment Theory",
    "bowlby": "Attachment Theory",
    "ainsworth": "Attachment Theory",
    "cognitive_dissonance": "Cognitive Dissonance",
    "festinger": "Cognitive Dissonance",
    "schema": "Schema Theory",
    "beck": "Cognitive Behavioral Theory",
    "cbt": "Cognitive Behavioral Theory",
    "operant": "Operant Conditioning",
    "skinner": "Operant Conditioning",
    "maslow": "Humanistic Psychology",
    "hierarchy": "Humanistic Psychology",
    "psychodynamic": "Psychodynamic Theory",
    "freud": "Psychodynamic Theory",
    "jung": "Analytical Psychology",
    "trauma": "Trauma & PTSD Frameworks",
    "ptsd": "Trauma & PTSD Frameworks",
    "social_learning": "Social Learning Theory",
    "bandura": "Social Learning Theory",
    "self_determination": "Self-Determination Theory",
    "deci": "Self-Determination Theory",
    "transactional": "Transactional Analysis",
    "berne": "Transactional Analysis",
    "emotion_regulation": "Emotion Regulation Theory",
    "gottman": "Relational Dynamics",
    "narrative": "Narrative Therapy",
    "systems": "Family Systems Theory",
    "bowen": "Family Systems Theory",
    "dialectical": "Dialectical Behavior Therapy",
    "dbt": "Dialectical Behavior Therapy",
}


def _infer_theory_category(filename: str) -> str:
    """
    Guess the psychological theory category from a filename.
    Falls back to 'General Psychology' when no keyword matches.
    """
    stem = filename.lower().replace("-", "_").replace(" ", "_")
    for keyword, category in THEORY_KEYWORD_MAP.items():
        if keyword in stem:
            return category
    return "General Psychology"


def _sanitize_text(text: str) -> str:
    """
    Light cleanup: collapse excessive whitespace, strip stray page numbers,
    remove lines that are purely numeric (page footers).
    """
    # Drop lines that are just page numbers (e.g. "  42  " or "— 42 —")
    text = re.sub(r"^\s*[—\-]?\s*\d+\s*[—\-]?\s*$", "", text, flags=re.MULTILINE)
    # Normalize whitespace runs to a single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class DocumentProcessor:
    """
    Ingests a directory of psychology PDFs and returns chunked TextNodes
    enriched with metadata.

    Usage
    -----
    >>> processor = DocumentProcessor(pdf_dir=Path("./data/pdfs"))
    >>> nodes = processor.run()
    >>> print(f"Produced {len(nodes)} nodes")
    """

    def __init__(
        self,
        pdf_dir: Optional[Path] = None,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.pdf_dir = pdf_dir or settings.pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # SentenceSplitter is LlamaIndex's recursive character splitter.
        # It tries to split on sentence boundaries first, then falls back to
        # character-level splits — preserving semantic coherence.
        self._splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"(?<=[.?!])\s+",  # sentence boundary
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> list[TextNode]:
        """
        Full ingestion pipeline.
        Returns a flat list of TextNode objects with enriched metadata.
        """
        pdf_files = self._discover_pdfs()
        if not pdf_files:
            logger.warning(f"No PDFs found in {self.pdf_dir}. Add files and re-ingest.")
            return []

        logger.info(f"Discovered {len(pdf_files)} PDF(s) in {self.pdf_dir}")

        raw_documents = self._load_documents(pdf_files)
        logger.info(f"Loaded {len(raw_documents)} raw document(s)")

        nodes = self._chunk_and_enrich(raw_documents)
        logger.success(f"Produced {len(nodes)} TextNode(s) ready for embedding")
        return nodes

    # ── Private helpers ───────────────────────────────────────────────────────

    def _discover_pdfs(self) -> list[Path]:
        """Recursively find all .pdf files under self.pdf_dir."""
        if not self.pdf_dir.exists():
            raise FileNotFoundError(
                f"PDF directory not found: {self.pdf_dir}\n"
                f"Create it with: mkdir -p {self.pdf_dir}"
            )
        return sorted(self.pdf_dir.rglob("*.pdf"))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _load_documents(self, pdf_files: list[Path]):
        """
        Load PDFs using LlamaIndex's SimpleDirectoryReader.
        PyMuPDF backend is used when available for superior text extraction.
        Retries up to 3 times on transient I/O errors.
        """
        reader = SimpleDirectoryReader(
            input_files=[str(p) for p in pdf_files],
            filename_as_id=True,       # uses filename as the document ID
            raise_on_error=False,      # log failures, don't crash the whole run
        )
        return reader.load_data()

    def _chunk_and_enrich(self, documents) -> list[TextNode]:
        """
        Split documents into chunks and attach psychological metadata.
        Each node gets:
          - source_document: original PDF filename
          - theory_category: inferred psychological framework
          - chunk_index: position within the source document
        """
        all_nodes: list[TextNode] = []

        for doc in documents:
            # Sanitize extracted text before chunking
            doc.text = _sanitize_text(doc.text)

            if not doc.text.strip():
                logger.warning(f"Empty text after sanitization for: {doc.doc_id}")
                continue

            # Infer the theory category from the source filename
            source_name = Path(doc.doc_id).name if doc.doc_id else "unknown.pdf"
            theory_category = _infer_theory_category(source_name)

            # Chunk the document
            chunks: list[TextNode] = self._splitter.get_nodes_from_documents([doc])

            # Enrich each chunk with metadata
            for idx, node in enumerate(chunks):
                node.metadata.update(
                    {
                        "source_document": source_name,
                        "theory_category": theory_category,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                    }
                )
                # LlamaIndex will include metadata in retrieval context
                node.excluded_embed_metadata_keys = []
                node.excluded_llm_metadata_keys = ["chunk_index", "total_chunks"]

            all_nodes.extend(chunks)
            logger.debug(
                f"  {source_name} → {len(chunks)} chunks [{theory_category}]"
            )

        return all_nodes
