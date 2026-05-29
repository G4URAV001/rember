"""
Rember CLI — command-line interface built with Typer + Rich.

Commands:
  rember init               Initialize config and data directory
  rember ingest <source>    Ingest a file or text
  rember query <question>   Ask a question
  rember list               List ingested documents
  rember stats              Show storage stats
  rember delete <doc_id>    Delete a document
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="rember",
    help="[bold cyan]Rember[/] — A RAG pipeline that remembers everything you feed it.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()
err_console = Console(stderr=True, style="bold red")

logging.basicConfig(level=logging.WARNING)  # suppress info logs in CLI by default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pipeline():
    """Lazily build the pipeline (triggers config + API client init)."""
    try:
        from rember.config import load_settings
        from rember.pipeline import Pipeline

        settings = load_settings()
        return Pipeline.from_settings(settings)
    except ValueError as e:
        err_console.print(f"[bold red]Configuration error:[/] {e}")
        err_console.print(
            "\n[yellow]Tip:[/] Run [bold]rember init[/] and make sure your "
            "[bold].env[/] file contains [bold]GOOGLE_API_KEY[/]."
        )
        raise typer.Exit(1)


def _get_query_engine(pipeline):
    """Build a Retriever + Answerer from an existing Pipeline."""
    from rember.query.answerer import Answerer
    from rember.query.retriever import Retriever

    retriever = Retriever(
        vector_store=pipeline.vector_store,
        metadata_store=pipeline.metadata_store,
        embedding_provider=pipeline.embedding_provider,
        config=pipeline._settings.query,
    )
    answerer = Answerer(llm_registry=pipeline.llm_registry)
    return retriever, answerer


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def init(
    config_dir: Annotated[
        str,
        typer.Option("--dir", help="Directory to initialise (default: current dir)"),
    ] = ".",
) -> None:
    """
    Initialise Rember: create config template and data directory.
    """
    dest = Path(config_dir).resolve()

    console.print(Panel.fit(
        "[bold cyan]Initialising Rember[/]",
        border_style="cyan",
    ))

    # Copy config.default.yaml → config.yaml if not already present
    default_config = Path(__file__).parent.parent.parent / "config.default.yaml"
    user_config = dest / "config.yaml"

    if user_config.exists():
        console.print(f"[yellow]⚠[/]  config.yaml already exists — skipping.")
    else:
        if default_config.exists():
            shutil.copy(default_config, user_config)
            console.print(f"[green]✓[/]  Created [bold]config.yaml[/]")
        else:
            console.print("[yellow]⚠[/]  config.default.yaml not found — skipping config copy.")

    # Copy .env.example → .env if not already present
    env_example = dest / ".env.example"
    env_file = dest / ".env"
    if env_file.exists():
        console.print(f"[yellow]⚠[/]  .env already exists — skipping.")
    elif env_example.exists():
        shutil.copy(env_example, env_file)
        console.print(f"[green]✓[/]  Created [bold].env[/] from template")
    else:
        console.print("[yellow]⚠[/]  .env.example not found — create .env manually.")

    # Create data directory
    from rember.config import load_settings
    settings = load_settings()
    data_dir = settings.storage.resolved_data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]✓[/]  Data directory: [bold]{data_dir}[/]")

    console.print("\n[bold green]Done![/] Next steps:")
    console.print("  1. Add your [bold]GOOGLE_API_KEY[/] to [bold].env[/]")
    console.print("  2. Run [bold cyan]rember ingest --text \"Hello, world!\"[/]")
    console.print("  3. Run [bold cyan]rember query \"What did I store?\"[/]")


@app.command()
def ingest(
    source: Annotated[
        str | None,
        typer.Argument(help="File path to ingest (e.g. notes.txt)"),
    ] = None,
    text: Annotated[
        str | None,
        typer.Option("--text", "-t", help="Raw text to ingest directly"),
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Metadata tags as key=value (repeatable)"),
    ] = None,
) -> None:
    """
    Ingest a file or raw text into the knowledge base.

    Examples:
      rember ingest notes.txt
      rember ingest --text "Python was created in 1991"
      rember ingest report.md --tag project=work --tag author=alice
    """
    if not source and not text:
        err_console.print("Provide a file path or use [bold]--text[/] to ingest raw text.")
        raise typer.Exit(1)

    if source and text:
        err_console.print("Provide either a file path OR [bold]--text[/], not both.")
        raise typer.Exit(1)

    # Parse tags
    metadata: dict = {}
    for t in (tag or []):
        if "=" in t:
            k, _, v = t.partition("=")
            metadata[k.strip()] = v.strip()
        else:
            err_console.print(f"Invalid tag format: '{t}'. Use [bold]key=value[/].")
            raise typer.Exit(1)

    pipeline = _get_pipeline()

    with console.status("[bold cyan]Ingesting…[/]", spinner="dots"):
        try:
            if text:
                doc = pipeline.ingest_text(text, metadata=metadata)
            else:
                doc = pipeline.ingest_file(source, metadata=metadata)  # type: ignore[arg-type]
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            err_console.print(f"[bold red]Ingestion failed:[/] {e}")
            raise typer.Exit(1)

    # Show different info depending on source type
    source_type = doc.source_type
    if source_type in ("image", "video"):
        size_bytes = doc.metadata.get("file_size_bytes", 0)
        content_info = f"File size: {_fmt_bytes(size_bytes)}"
        if source_type == "image":
            w = doc.metadata.get("image_width", "?")
            h = doc.metadata.get("image_height", "?")
            fmt = doc.metadata.get("image_format", "")
            content_info += f" ({w}\u00d7{h} {fmt})"
        elif source_type == "video":
            dur = doc.metadata.get("video_duration", 0)
            content_info += f" ({dur:.1f}s video)"
    else:
        content_info = f"Content length: {len(doc.raw_content):,} chars"

    console.print(Panel.fit(
        f"[bold green]\u2713 Ingested successfully[/]\n\n"
        f"[dim]Document ID:[/] {doc.id}\n"
        f"[dim]Source type:[/] {doc.source_type}\n"
        f"[dim]{content_info}[/]\n"
        f"[dim]Tags:[/] {metadata or '(none)'}",
        title="[bold]Rember[/]",
        border_style="green",
    ))


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Your question")],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Number of context chunks to retrieve"),
    ] = 5,
    min_score: Annotated[
        float,
        typer.Option("--min-score", help="Minimum similarity score (0.0–1.0)"),
    ] = 0.3,
    show_sources: Annotated[
        bool,
        typer.Option("--sources/--no-sources", help="Show source chunks"),
    ] = True,
) -> None:
    """
    Ask a question and get an answer from your knowledge base.

    Example:
      rember query "What did I learn about Python?"
    """
    pipeline = _get_pipeline()
    retriever, answerer = _get_query_engine(pipeline)

    with console.status("[bold cyan]Thinking…[/]", spinner="dots"):
        results = retriever.retrieve(question, top_k=top_k, min_score=min_score)
        answer = answerer.answer(question, results)

    # Answer panel
    console.print()
    console.print(Panel(
        Markdown(answer.answer),
        title=f"[bold cyan]Answer[/]  [dim]— {len(results)} sources[/]",
        border_style="cyan",
        padding=(1, 2),
    ))

    # Sources table
    if show_sources and results:
        console.print()
        table = Table(
            title="Sources",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Source", style="yellow")
        table.add_column("Preview", style="white")

        for i, r in enumerate(results, start=1):
            source = r.metadata.get("source_path") or r.metadata.get(
                "filename", r.document_id[:12]
            )
            preview = r.content[:100].replace("\n", " ") + ("…" if len(r.content) > 100 else "")
            table.add_row(
                str(i),
                f"{r.score:.2f}",
                str(source),
                preview,
            )

        console.print(table)

    if not results:
        console.print(
            "[yellow]No relevant information found.[/] "
            "Try ingesting more content with [bold]rember ingest[/]."
        )


@app.command(name="list")
def list_documents(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Maximum number of documents to show"),
    ] = 20,
) -> None:
    """List all ingested documents."""
    pipeline = _get_pipeline()
    docs = pipeline.metadata_store.list_documents()

    if not docs:
        console.print(
            "[yellow]No documents ingested yet.[/] "
            "Use [bold cyan]rember ingest[/] to add content."
        )
        return

    table = Table(
        title=f"Knowledge Base — {len(docs)} document(s)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("ID", style="dim", width=12)
    table.add_column("Type", style="yellow", width=8)
    table.add_column("Source / Preview", style="white")
    table.add_column("Tags", style="green")
    table.add_column("Ingested", style="dim")

    for doc in docs[:limit]:
        doc_id_short = doc.id[:8] + "…"
        source = doc.source_path or doc.raw_content[:50].replace("\n", " ") + "…"
        tags = ", ".join(f"{k}={v}" for k, v in doc.metadata.items() if k not in ("filename", "file_extension"))
        ingested = doc.created_at.strftime("%Y-%m-%d %H:%M")
        table.add_row(doc_id_short, doc.source_type, source, tags or "—", ingested)

    console.print(table)

    if len(docs) > limit:
        console.print(f"[dim]Showing {limit} of {len(docs)} documents. Use --limit to see more.[/]")


@app.command()
def stats() -> None:
    """Show storage statistics."""
    pipeline = _get_pipeline()
    s = pipeline.get_stats()

    console.print()
    console.print(Panel(
        f"[bold]Documents:[/]  {s['document_count']:,}\n"
        f"[bold]Chunks:[/]     {s['chunk_count']:,}\n"
        f"[bold]Vectors:[/]    {s['vector_count']:,}  "
            f"[dim](dim={s['embedding_dimension']})[/]\n\n"
        f"[bold]DB size:[/]    {_fmt_bytes(s['db_size_bytes'])}\n"
        f"[bold]DB path:[/]    {s['db_path']}\n"
        f"[bold]Index:[/]      {s['index_path']}",
        title="[bold cyan]Rember Stats[/]",
        border_style="cyan",
        padding=(1, 2),
    ))


@app.command()
def delete(
    doc_id: Annotated[str, typer.Argument(help="Document ID (or prefix) to delete")],
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """
    Delete a document and its chunks from the knowledge base.

    Note: Vectors are NOT removed from the FAISS index immediately
    (FAISS flat indexes don't support removal). They become orphaned and
    will be skipped in results because their SQLite records are gone.
    A full rebuild would be needed to reclaim vector space.
    """
    pipeline = _get_pipeline()
    docs = pipeline.metadata_store.list_documents()

    # Find matching doc(s) by full ID or prefix
    matches = [d for d in docs if d.id.startswith(doc_id)]

    if not matches:
        err_console.print(f"No document found with ID starting with '{doc_id}'.")
        raise typer.Exit(1)

    if len(matches) > 1:
        err_console.print(
            f"Ambiguous prefix '{doc_id}' matches {len(matches)} documents. "
            "Provide more characters."
        )
        raise typer.Exit(1)

    doc = matches[0]

    if not force:
        source = doc.source_path or doc.raw_content[:60] + "…"
        confirm = typer.confirm(
            f"Delete document {doc.id[:12]}… (source: {source})?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Cancelled.[/]")
            raise typer.Exit(0)

    deleted = pipeline.metadata_store.delete_document(doc.id)
    if deleted:
        console.print(f"[green]✓[/]  Deleted document [bold]{doc.id[:12]}…[/]")
        console.print(
            "[dim]Note: FAISS vectors are orphaned (not removed). "
            "Orphaned vectors are harmless — they score but have no matching metadata.[/]"
        )
    else:
        err_console.print("Deletion failed — document not found.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
