#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# setup.sh — Complete setup for Behavioral Explainer on Fedora 43
#
# Run once: bash scripts/setup.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; exit 1; }

info "═══════════════════════════════════════════════════════"
info "  Behavioral Explainer — Fedora 43 Setup"
info "═══════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system dependencies..."
sudo dnf install -y \
  python3.11 \
  python3.11-devel \
  python3-pip \
  python3-virtualenv \
  curl \
  docker \
  docker-compose \
  git \
  gcc \
  g++ \
  make
success "System packages installed."

# ── 2. Start Docker ───────────────────────────────────────────────────────────
info "Enabling and starting Docker..."
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
success "Docker running. (Log out and back in if you get permission errors.)"

# ── 3. Qdrant via Docker ──────────────────────────────────────────────────────
info "Pulling and starting Qdrant..."
docker pull qdrant/qdrant:latest

# Remove stale container if exists
docker rm -f qdrant 2>/dev/null || true

docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest

# Wait for Qdrant to be ready
info "Waiting for Qdrant to be ready..."
for i in {1..15}; do
  if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
    success "Qdrant is up at http://localhost:6333"
    break
  fi
  sleep 2
  if [ "$i" -eq 15 ]; then
    error "Qdrant failed to start. Check: docker logs qdrant"
  fi
done

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
info "Installing Ollama..."
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.ai/install.sh | sh
else
  warn "Ollama already installed — skipping."
fi

info "Starting Ollama service..."
sudo systemctl enable --now ollama || ollama serve &>/dev/null &
sleep 3

# ── 5. Pull Ollama models ─────────────────────────────────────────────────────
info "Pulling Llama 3.1 (this may take a while on first run)..."
ollama pull llama3.1

info "Pulling nomic-embed-text embedding model..."
ollama pull nomic-embed-text
# Alternative high-performance option:
# ollama pull bge-small:en

success "Ollama models ready."

# ── 6. Python virtual environment ────────────────────────────────────────────
info "Creating Python 3.11 virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

info "Installing Python dependencies..."
pip install --upgrade pip wheel setuptools
pip install pydantic-settings   # required for config.py
pip install -r requirements.txt

success "Python environment ready."

# ── 7. Data directory ─────────────────────────────────────────────────────────
mkdir -p data/pdfs
info "Put your psychology PDFs into: $(pwd)/data/pdfs/"
info "Naming tip: 'attachment_theory_bowlby.pdf', 'cognitive_dissonance_festinger.pdf', etc."
info "The filename drives automatic theory category tagging."

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
info "═══════════════════════════════════════════════════════"
success "Setup complete! Next steps:"
echo ""
echo "  1.  Add PDFs to:    ./data/pdfs/"
echo "  2.  Ingest:         python main.py ingest"
echo "  3.  Analyze:        python main.py analyze \"Your situation here\""
echo "  4.  Interactive:    python main.py repl"
echo "  5.  Check status:   python main.py status"
echo ""
info "═══════════════════════════════════════════════════════"
