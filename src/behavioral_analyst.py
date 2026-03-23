"""
behavioral_analyst.py — The core analysis engine.

This is where the prompt engineering lives. The BehavioralAnalyst does NOT
give advice. It is a Structural Analyst: it identifies the mechanical dynamics
of a human situation using retrieved psychological frameworks and explains
*why* the pattern exists — not what to do about it.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │  User Input                                                  │
  │    ↓                                                         │
  │  [Embed query] → [Qdrant ANN search] → [Top-K nodes]        │
  │    ↓                                                         │
  │  [Build context from nodes + metadata]                       │
  │    ↓                                                         │
  │  [Custom Structural Analyst Prompt] → [Llama 3.1 via Ollama] │
  │    ↓                                                         │
  │  Structured Behavioral Analysis Output                       │
  └──────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.ollama import Ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.vector_store import VectorStoreManager


# ── Structural Analyst System Prompt ──────────────────────────────────────────
#
# Design philosophy:
#   - No advice. No "you should". No prescriptions.
#   - The analyst explains mechanics, not morality.
#   - It names specific theories from the retrieved corpus.
#   - It treats human behavior as a system to be decoded, not fixed.
#   - It separates the person's narrative from the structural pattern.
#
STRUCTURAL_ANALYST_TEMPLATE = PromptTemplate(
    """\
You are a Structural Analyst specializing in behavioral mechanics. \
Your role is NOT to advise, console, or suggest solutions. \
Your role is to dissect the architecture of the situation — to name \
what is actually happening at a structural level using psychological frameworks.

You speak with precision. You do not moralize. \
You treat human behavior as a system with inputs, outputs, and feedback loops.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRIEVED PSYCHOLOGICAL FRAMEWORKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{context_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITUATION PRESENTED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{query_str}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRUCTURAL ANALYSIS — follow this exact format:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**I. CORE CONFLICT IDENTIFICATION**
State the central structural tension in one or two sentences. \
Strip away the narrative. What is the actual dynamic at play?

**II. OPERATIVE PSYCHOLOGICAL FRAMEWORKS**
Name 2–3 specific theories from the retrieved corpus that are mechanically \
active in this situation. For each theory:
  - State the theory's name and its core claim.
  - Explain precisely how its mechanism maps to the presented situation.
  - Quote or closely reference the retrieved source material where relevant.

**III. BEHAVIORAL MECHANICS**
Synthesize the frameworks into a unified explanation of *why* this pattern \
exists and what is sustaining it. Explain the feedback loop. \
Identify which variables are locked and which are in motion.

**IV. STRUCTURAL SIGNATURE**
Name the pattern in clinical or theoretical terms (e.g., "avoidant \
attachment cycle," "cognitive dissonance reduction via rationalization," \
"negative reinforcement loop"). One to three sentences. No advice.

Do not offer solutions. Do not say "you should." \
Do not validate or invalidate the situation. \
Remain structurally precise and theoretically grounded throughout.
"""
)


@dataclass
class AnalysisResult:
    """Container for a completed behavioral analysis."""

    query: str
    response: str
    source_theories: list[str] = field(default_factory=list)
    source_documents: list[str] = field(default_factory=list)
    raw_nodes: list = field(default_factory=list)

    def __str__(self) -> str:
        return self.response

    def sources_summary(self) -> str:
        """Human-readable source attribution."""
        if not self.source_theories:
            return "No sources retrieved."
        lines = []
        for doc, theory in zip(self.source_documents, self.source_theories):
            lines.append(f"  • [{theory}] {doc}")
        return "\n".join(lines)


class BehavioralAnalyst:
    """
    The main interface for behavioral analysis.

    Composes the retriever (from VectorStoreManager) with a custom prompt
    and the Ollama LLM into a single query engine.

    Usage
    -----
    >>> analyst = BehavioralAnalyst(store_manager)
    >>> result = analyst.analyze("My partner goes silent whenever I bring up finances.")
    >>> print(result)
    >>> print(result.sources_summary())
    """

    def __init__(
        self,
        store_manager: VectorStoreManager,
        llm_model: str = settings.ollama_llm_model,
        ollama_base_url: str = settings.ollama_base_url,
        top_k: Optional[int] = None,
    ) -> None:
        self._store_manager = store_manager

        # ── LLM (Llama 3.1 via Ollama) ────────────────────────────────────────
        self._llm = Ollama(
            model=llm_model,
            base_url=ollama_base_url,
            request_timeout=300.0,       # Llama 3.1 can be slow; generous timeout
            temperature=0.3,             # Lower temp = more precise, less hallucinated
            context_window=8192,
            additional_kwargs={
                "num_predict": 2048,     # max output tokens
                "repeat_penalty": 1.1,  # reduce repetitive phrasing
            },
        )

        # ── Query Engine ──────────────────────────────────────────────────────
        retriever = store_manager.get_retriever(top_k=top_k)

        response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            text_qa_template=STRUCTURAL_ANALYST_TEMPLATE,
            response_mode="compact",    # compact: fewer LLM calls, tighter synthesis
            streaming=False,
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )

        logger.info(
            f"BehavioralAnalyst initialized — LLM: {llm_model}, "
            f"top_k={top_k or settings.top_k_retrieval}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=3, max=15),
        reraise=True,
    )
    def analyze(self, situation: str) -> AnalysisResult:
        """
        Run a full structural analysis on a described situation.

        Parameters
        ----------
        situation : str
            A description of a human interaction, conflict, or behavioral pattern.

        Returns
        -------
        AnalysisResult
            Contains the analysis text, retrieved theories, and source documents.
        """
        if not situation.strip():
            raise ValueError("Empty situation string passed to analyze().")

        logger.info(f"Analyzing: {situation[:80]}{'...' if len(situation) > 80 else ''}")

        response = self._query_engine.query(situation)

        # Extract metadata from retrieved source nodes
        source_theories: list[str] = []
        source_documents: list[str] = []
        raw_nodes = []

        if hasattr(response, "source_nodes"):
            for node_with_score in response.source_nodes:
                node = node_with_score.node
                meta = node.metadata or {}
                theory = meta.get("theory_category", "Unknown")
                doc = meta.get("source_document", "Unknown")
                if theory not in source_theories:
                    source_theories.append(theory)
                if doc not in source_documents:
                    source_documents.append(doc)
                raw_nodes.append(node_with_score)

        return AnalysisResult(
            query=situation,
            response=str(response),
            source_theories=source_theories,
            source_documents=source_documents,
            raw_nodes=raw_nodes,
        )

    def analyze_stream(self, situation: str):
        """
        Streaming variant of analyze(). Yields text chunks as they arrive from Ollama.
        Useful for CLI or web streaming interfaces.
        Note: source node metadata is not available in streaming mode.
        """
        if not situation.strip():
            raise ValueError("Empty situation string passed to analyze_stream().")

        # Swap to a streaming synthesizer temporarily
        from llama_index.core.response_synthesizers import get_response_synthesizer as grs

        stream_synthesizer = grs(
            llm=self._llm,
            text_qa_template=STRUCTURAL_ANALYST_TEMPLATE,
            response_mode="compact",
            streaming=True,
        )
        stream_engine = RetrieverQueryEngine(
            retriever=self._store_manager.get_retriever(),
            response_synthesizer=stream_synthesizer,
        )

        streaming_response = stream_engine.query(situation)
        for chunk in streaming_response.response_gen:
            yield chunk
