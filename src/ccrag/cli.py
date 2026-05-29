from __future__ import annotations

from pathlib import Path

import click

from .embedder import DEFAULT_MODEL


@click.group()
def main():
    """ccrag — RAG pipeline for Claude Code."""


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Sentence-transformers model for embeddings.")
@click.option("--force", is_flag=True, help="Drop and rebuild the index from scratch.")
def index(path: str, model: str, force: bool):
    """Index a codebase at PATH (default: current directory)."""
    from .embedder import Embedder, models_cache_dir
    from .indexer import index_repo

    root = Path(path).resolve()

    if force:
        import shutil
        # Drop only the vector index — keep the cached model so --force does not
        # trigger a re-download.
        index_dir = root / ".ccrag" / "index"
        if index_dir.exists():
            shutil.rmtree(index_dir)
            click.echo(f"Removed existing index at {index_dir}")

    embedder = Embedder(model, cache_folder=models_cache_dir(root))
    index_repo(root, embedder, force=force, verbose=True)


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Sentence-transformers model for embeddings.")
def serve(path: str, model: str):
    """Start the MCP server for the indexed codebase at PATH."""
    from .server import run_server

    root = Path(path).resolve()
    ccrag_dir = root / ".ccrag"
    if not ccrag_dir.exists():
        raise click.ClickException(
            f"No index found at {ccrag_dir}. Run `ccrag index {path}` first."
        )

    run_server(root, model)


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--watch", is_flag=True, help="Watch for file changes and re-index automatically.")
def watch(path: str, model: str, watch: bool):
    """Watch PATH and incrementally re-index changed files."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        raise click.ClickException("Install watchdog: pip install watchdog")

    import time

    from .chunker import IGNORE_DIRS, TEXT_EXTENSIONS
    from .embedder import Embedder, models_cache_dir
    from .indexer import index_file

    root = Path(path).resolve()
    embedder = Embedder(model, cache_folder=models_cache_dir(root))

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            self._handle(event.src_path)

        def on_created(self, event):
            self._handle(event.src_path)

        def _handle(self, src: str):
            p = Path(src)
            if p.is_dir():
                return
            if any(part in IGNORE_DIRS for part in p.parts):
                return
            if p.suffix.lower() not in TEXT_EXTENSIONS:
                return
            click.echo(f"Re-indexing {p.relative_to(root)}")
            try:
                index_file(root, p, embedder)
            except Exception as e:
                click.echo(f"  error: {e}", err=True)

    observer = Observer()
    observer.schedule(Handler(), str(root), recursive=True)
    observer.start()
    click.echo(f"Watching {root} for changes (Ctrl+C to stop)…")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


@main.command()
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
def status(path: str):
    """Show index statistics for the codebase at PATH."""
    from .chunker import ast_available
    from .store import stats

    root = Path(path).resolve()
    s = stats(root)
    if not s.get("indexed"):
        click.echo("Not indexed. Run `ccrag index` first.")
    else:
        click.echo(f"Index: {s['files']} files, {s['chunks']} chunks")

    if ast_available():
        click.echo("Chunking: AST (tree-sitter, function/class boundaries)")
    else:
        click.echo(
            "Chunking: line-window fallback — tree-sitter not installed. "
            "Install with: pip install 'ccrag[ast]'"
        )


@main.command(name="mcp-config")
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
def mcp_config(path: str):
    """Print the MCP server config snippet to add to .claude/settings.json."""
    import json
    import shutil

    root = Path(path).resolve()
    ccrag_bin = shutil.which("ccrag") or "ccrag"

    config = {
        "mcpServers": {
            "ccrag": {
                "command": ccrag_bin,
                "args": ["serve", str(root)],
            }
        }
    }
    click.echo(json.dumps(config, indent=2))
