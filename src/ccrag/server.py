from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .embedder import Embedder, models_cache_dir
from .store import TABLE_NAME, open_db, search, stats

mcp = FastMCP("ccrag")

_root: Path = Path.cwd()
_embedder: Embedder = Embedder()


def _get_table():
    db = open_db(_root)
    if TABLE_NAME not in db.table_names():
        return None
    return db.open_table(TABLE_NAME)


@mcp.tool()
def search_codebase(query: str, n_results: int = 8) -> str:
    """Search the indexed codebase for code relevant to a natural language query.

    Returns the top matching code chunks with file paths and line numbers.
    Use this whenever you need to understand how something is implemented,
    find a function, or get context about the codebase.
    """
    table = _get_table()
    if table is None:
        return "Codebase index not found. Run `ccrag index` in the project root first."

    query_vec = _embedder.embed_one(query)
    results = search(table, query_vec, n=n_results)

    if not results:
        return "No matching code found."

    parts = []
    for r in results:
        header = f"### {r['file_path']} (lines {r['start_line']}–{r['end_line']})"
        parts.append(f"{header}\n```{r['language']}\n{r['content']}\n```")

    return "\n\n".join(parts)


@mcp.tool()
def codebase_stats() -> str:
    """Return statistics about the indexed codebase: number of files and chunks."""
    s = stats(_root)
    if not s.get("indexed"):
        return "Index not found. Run `ccrag index` first."
    return f"Index: {s['files']} files, {s['chunks']} chunks."


def run_server(root: Path, model: str):
    global _root, _embedder
    _root = root.resolve()
    _embedder = Embedder(model, cache_folder=models_cache_dir(_root))
    mcp.run(transport="stdio")
