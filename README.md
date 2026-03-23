# Behavioral Explainer — RAG System for Psychological Framework Analysis

A production-grade Python RAG system that analyzes human interactions by
retrieving and synthesizing real psychological frameworks. No advice. No
platitudes. Pure structural analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         BEHAVIORAL EXPLAINER                            │
├─────────────────┬───────────────────────┬───────────────────────────────┤
│ DocumentProcessor│   VectorStoreManager  │      BehavioralAnalyst        │
│                 │                       │                               │
│ ┌─────────────┐ │ ┌───────────────────┐ │ ┌───────────────────────────┐ │
│ │ PDF Scanner │ │ │  Qdrant Client    │ │ │  Ollama (Llama 3.1)       │ │
│ │ (rglob)     │ │ │  (cosine, HNSW)   │ │ │  temp=0.3, ctx=8192       │ │
│ └──────┬──────┘ │ └────────┬──────────┘ │ └───────────┬───────────────┘ │
│        ↓        │          ↓            │             ↓                 │
│ ┌─────────────┐ │ ┌───────────────────┐ │ ┌───────────────────────────┐ │
│ │ PyMuPDF     │ │ │  Payload Indexes  │ │ │  Structural Analyst       │ │
│ │ Text Extract│ │ │  theory_category  │ │ │  Prompt Template          │ │
│ └──────┬──────┘ │ │  source_document  │ │ └───────────┬───────────────┘ │
│        ↓        │ └────────┬──────────┘ │             ↓                 │
│ ┌─────────────┐ │          ↓            │ ┌───────────────────────────┐ │
│ │ SentenceSplit│ │ ┌───────────────────┐ │ │  RetrieverQueryEngine     │ │
│ │ 1024t / 50t │ │ │  OllamaEmbedding  │ │ │  (compact synthesis)      │ │
│ └──────┬──────┘ │ │  (bge-small/nomic)│ │ └───────────────────────────┘ │
│        ↓        │ └───────────────────┘ │                               │
│ ┌─────────────┐ │                       │                               │
│ │ Metadata    │ │                       │                               │
│ │ Enrichment  │ │                       │                               │
│ └─────────────┘ │                       │                               │
└─────────────────┴───────────────────────┴───────────────────────────────┘
         ↓                    ↓                           ↓
    TextNode[]          Qdrant Collection          AnalysisResult
    (chunked PDFs)      (vectors + metadata)       (structured output)
```

---

## Project Structure

```
behavioral_explainer/
├── main.py                    # CLI entry point (Typer)
├── requirements.txt           # Pinned dependencies
├── .env                       # Runtime configuration
├── src/
│   ├── __init__.py
│   ├── config.py              # Pydantic settings (reads .env)
│   ├── document_processor.py  # PDF ingestion + chunking
│   ├── vector_store.py        # Qdrant lifecycle manager
│   └── behavioral_analyst.py  # Query engine + custom prompt
├── knowledge_base/
│   └── pdfs/                  # ← Drop your psychology PDFs here
├── scripts/
│   └── setup.sh               # One-shot Fedora setup
└── data_storage/            # Auto-created: Qdrant persistence QDRANT DATABASE LIVES HERE
```

---

## Setup (Fedora 43)

### Step 1 — One-shot setup script

```bash
# Clone / create the project directory
cd behavioral_explainer

# Make script executable and run
chmod +x scripts/setup.sh
bash scripts/setup.sh
```

This script installs: Docker, Qdrant (Docker), Ollama, Llama 3.1, nomic-embed-text, Python 3.11 venv.

### Step 2 — Manual setup (if you prefer control)

```bash
# ── System packages ───────────────────────────────────────────────────────────
sudo dnf install -y python3.11 python3.11-devel python3-pip docker curl

# ── Docker + Qdrant ───────────────────────────────────────────────────────────
sudo systemctl enable --now docker
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# ── Ollama + models ───────────────────────────────────────────────────────────
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl enable --now ollama
ollama pull llama3.1
ollama pull nomic-embed-text        # 768-dim, fast
# OR: ollama pull bge-small:en       # 384-dim, even faster

# ── Python environment ────────────────────────────────────────────────────────
python3.11 -m venv .venv
source .venv/bin/activate
pip install pydantic-settings
pip install -r requirements.txt
```

---

## Usage

### Ingest your PDFs

```bash
source .venv/bin/activate

# Basic ingestion
python main.py ingest

# Custom PDF directory
python main.py ingest --pdf-dir /path/to/your/pdfs

# Reset collection and re-ingest from scratch
python main.py ingest --reset
```

**PDF naming convention** (drives automatic theory tagging):
```
attachment_theory_bowlby.pdf       → Attachment Theory
cognitive_dissonance_festinger.pdf → Cognitive Dissonance
operant_conditioning_skinner.pdf   → Operant Conditioning
family_systems_bowen.pdf           → Family Systems Theory
```

### Run a structural analysis

```bash
# Single analysis (batch mode)
python main.py analyze "My partner goes completely silent whenever I bring up
our finances, but then erupts in anger two days later. I don't understand why
addressing the issue directly seems to make everything worse."

# With more retrieved chunks
python main.py analyze "..." --top-k 7

# Without source attribution
python main.py analyze "..." --no-sources
```

### Streaming output (real-time token generation)

```bash
python main.py stream "My coworker constantly agrees with me in meetings but
then works against everything we agreed on privately."
```

### Interactive REPL

```bash
python main.py repl
# ▶ Situation: describe anything, press Enter
# Type 'exit' to quit
```

### System status

```bash
python main.py status
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_LLM_MODEL` | `llama3.1` | LLM model name |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `QDRANT_COLLECTION` | `behavioral_frameworks` | Collection name |
| `CHUNK_SIZE` | `1024` | Tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `5` | Retrieved chunks per query |

---

## Troubleshooting

### Qdrant not connecting
```bash
docker ps                          # Is qdrant running?
docker logs qdrant                 # Check logs
docker start qdrant                # Start if stopped
curl http://localhost:6333/healthz # Health check
```

### Ollama not responding
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
ollama list                        # Verify models are downloaded
```

### Wrong embedding dimensions
If you switch embedding models after initial ingestion, reset the collection:
```bash
python main.py ingest --reset
```

### Slow analysis
- Llama 3.1 requires ~8GB RAM minimum, 16GB+ recommended
- Use `llama3.1:8b` (default) not `llama3.1:70b` on consumer hardware
- Reduce `TOP_K_RETRIEVAL` to 3 for faster responses

---

## Extending the System

### Add new theory categories
Edit `THEORY_KEYWORD_MAP` in `src/document_processor.py`:
```python
"polyvagal": "Polyvagal Theory",
"porges": "Polyvagal Theory",
```

### Filter retrieval by theory
In `src/vector_store.py`, `get_retriever()`, add:
```python
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
filters = MetadataFilters(filters=[
    MetadataFilter(key="theory_category", value="Attachment Theory")
])
return self._index.as_retriever(similarity_top_k=top_k, filters=filters)
```

### Swap the LLM
In `.env`, set `OLLAMA_LLM_MODEL=mistral` or any Ollama-compatible model.

---

## Example Output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**I. CORE CONFLICT IDENTIFICATION**
The situation presents a classic avoidant-anxious attachment pairing in
which one partner's attempts at proximity-seeking (discussing the future)
trigger the other partner's proximity-avoidance response (emotional shutdown),
creating a self-reinforcing cycle.

**II. OPERATIVE PSYCHOLOGICAL FRAMEWORKS**

1. Attachment Theory (Bowlby, 1969)
   The emotional shutdown behavior maps precisely to the deactivating strategy
   described in avoidant attachment: when attachment needs feel threatening,
   the system suppresses emotional engagement to regulate proximity...

2. Emotional Regulation Theory (Gross, 1998)
   The partner's response constitutes expressive suppression — a downstream
   regulation strategy in which emotional response tendencies are inhibited
   after they have already been activated...

**III. BEHAVIORAL MECHANICS**
The feedback loop operates as follows: proximity-seeking (question about the
future) → threat activation in the avoidant system → deactivating strategy
(shutdown) → anxiety escalation in the anxious partner → increased
proximity-seeking attempts → further threat activation...

**IV. STRUCTURAL SIGNATURE**
Anxious-avoidant attachment cycle with asymmetric emotional regulation
strategies; expressive suppression on one axis, hyperactivation on the other.
```
