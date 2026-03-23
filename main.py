"""
main.py — CLI entry point for the Behavioral Explainer system.

Commands
--------
  ingest    Ingest PDF files into Qdrant
  analyze   Run a structural behavioral analysis (batch mode)
  stream    Run a structural behavioral analysis (streaming output)
  status    Show collection stats and system health
  repl      Interactive REPL for continuous analysis sessions

Run with: python main.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger 
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.table import Table

# ── Logger setup ──────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "behavioral_explainer.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)

# ── Lazy imports (avoids slow startup on --help) ──────────────────────────────
def _get_components():
    from src.document_processor import DocumentProcessor
    from src.vector_store import VectorStoreManager
    from src.behavioral_analyst import BehavioralAnalyst
    return DocumentProcessor, VectorStoreManager, BehavioralAnalyst


console = Console()
app = typer.Typer(
    name="behavioral-explainer",
    help="Structural behavioral analysis using RAG + psychological frameworks.",
    add_completion=False,
    rich_markup_mode="rich",
)


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    pdf_dir: Path = typer.Option(
        None,
        "--pdf-dir", "-d",
        help="Directory containing psychology PDFs. Defaults to .env PDF_DIR.",
    ),
    reset: bool = typer.Option(
        False,
        "--reset", "-r",
        help="Drop and recreate the Qdrant collection before ingesting.",
    ),
):
    """
    [bold green]Ingest[/bold green] psychology PDFs into the Qdrant vector store.

    This command processes all PDFs in the target directory, splits them into
    chunks, embeds them via Ollama, and upserts them into Qdrant.
    """
    DocumentProcessor, VectorStoreManager, _ = _get_components()

    rprint(Panel.fit(
        "[bold]Behavioral Explainer — Ingestion Pipeline[/bold]",
        border_style="cyan"
    ))

    # ── Reset collection if requested ─────────────────────────────────────────
    store = VectorStoreManager()

    if reset:
        from qdrant_client import QdrantClient
        from src.config import settings
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        existing = {c.name for c in client.get_collections().collections}
        if settings.qdrant_collection in existing:
            client.delete_collection(settings.qdrant_collection)
            logger.info(f"Collection '{settings.qdrant_collection}' dropped.")

    store.bootstrap()

    # ── Process documents ─────────────────────────────────────────────────────
    processor = DocumentProcessor(pdf_dir=pdf_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing PDFs...", total=None)
        nodes = processor.run()
        progress.update(task, description=f"Produced {len(nodes)} nodes")

    if not nodes:
        rprint("[yellow]No nodes produced. Check your PDF directory.[/yellow]")
        raise typer.Exit(1)

    # ── Ingest into Qdrant ────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"Embedding & upserting {len(nodes)} nodes...", total=None)
        store.ingest(nodes)

    stats = store.collection_stats()
    rprint(f"\n[bold green]✓ Ingestion complete![/bold green]")
    rprint(f"  Collection : [cyan]{stats['collection']}[/cyan]")
    rprint(f"  Vectors    : [cyan]{stats['vectors_count']}[/cyan]")


@app.command()
def analyze(
    situation: str = typer.Argument(
        ...,
        help="Describe the human interaction or behavioral pattern to analyze.",
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of framework chunks to retrieve."),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show retrieved sources."),
):
    """
    [bold green]Analyze[/bold green] a situation using retrieved psychological frameworks.

    Example:
      python main.py analyze "My partner shuts down emotionally whenever I try to discuss our future."
    """
    _, VectorStoreManager, BehavioralAnalyst = _get_components()

    rprint(Panel.fit(
        "[bold]Behavioral Explainer — Structural Analysis[/bold]",
        border_style="cyan"
    ))

    store = VectorStoreManager()
    store.bootstrap()

    analyst = BehavioralAnalyst(store_manager=store, top_k=top_k)

    rprint(f"\n[bold yellow]Situation:[/bold yellow] {situation}\n")
    console.print(Rule(style="dim"))

    with Progress(
        SpinnerColumn(),
        TextColumn("Running structural analysis via Llama 3.1..."),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("", total=None)
        result = analyst.analyze(situation)

    console.print(Rule(style="dim"))
    console.print(Markdown(result.response))

    if show_sources and result.source_theories:
        console.print(Rule(style="dim"))
        rprint("\n[bold dim]Retrieved Frameworks:[/bold dim]")
        rprint(result.sources_summary())


@app.command()
def stream(
    situation: str = typer.Argument(
        ...,
        help="Describe the human interaction or behavioral pattern to analyze.",
    ),
):
    """
    [bold green]Stream[/bold green] a structural analysis with real-time token output.

    Identical to analyze but prints tokens as they arrive from Ollama.
    """
    _, VectorStoreManager, BehavioralAnalyst = _get_components()

    store = VectorStoreManager()
    store.bootstrap()
    analyst = BehavioralAnalyst(store_manager=store)

    rprint(f"\n[bold yellow]Situation:[/bold yellow] {situation}\n")
    console.print(Rule(style="dim"))

    for chunk in analyst.analyze_stream(situation):
        console.print(chunk, end="", markup=False)

    console.print("\n")
    console.print(Rule(style="dim"))


@app.command()
def status():
    """
    [bold green]Status[/bold green] — Show collection stats and connectivity check.
    """
    _, VectorStoreManager, _ = _get_components()

    rprint(Panel.fit("[bold]System Status[/bold]", border_style="cyan"))

    try:
        store = VectorStoreManager()
        store.bootstrap()
        stats = store.collection_stats()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="dim")
        table.add_column("Value", style="bold cyan")
        table.add_row("Collection", stats["collection"])
        table.add_row("Total Vectors", str(stats["vectors_count"]))
        table.add_row("Indexed Vectors", str(stats["indexed_vectors_count"]))
        table.add_row("Status", str(stats["status"]))
        console.print(table)

    except Exception as exc:
        rprint(f"[red]✗ Qdrant connection failed:[/red] {exc}")
        rprint("[yellow]Is Qdrant running? Try: docker start qdrant[/yellow]")
        raise typer.Exit(1)


@app.command()
def repl():
    """
    [bold green]REPL[/bold green] — Interactive session for continuous analysis.

    Type your situation at the prompt. Type 'exit' or Ctrl-C to quit.
    """
    _, VectorStoreManager, BehavioralAnalyst = _get_components()

    rprint(Panel(
        "[bold]Behavioral Explainer — Interactive REPL[/bold]\n"
        "[dim]Describe any human interaction or pattern. Type [bold]exit[/bold] to quit.[/dim]",
        border_style="cyan",
    ))

    store = VectorStoreManager()
    store.bootstrap()
    analyst = BehavioralAnalyst(store_manager=store)

    while True:
        try:
            situation = console.input("\n[bold cyan]▶ Situation:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            rprint("\n[dim]Session ended.[/dim]")
            break

        if situation.lower() in {"exit", "quit", "q"}:
            rprint("[dim]Session ended.[/dim]")
            break

        if not situation:
            continue

        console.print(Rule(style="dim"))

        try:
            result = analyst.analyze(situation)
            console.print(Markdown(result.response))
            if result.source_theories:
                rprint(f"\n[dim]Sources: {' | '.join(result.source_theories)}[/dim]")
        except Exception as exc:
            rprint(f"[red]Analysis failed:[/red] {exc}")

        console.print(Rule(style="dim"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
