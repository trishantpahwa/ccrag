from __future__ import annotations

import gc
import hashlib
from pathlib import Path

from .chunker import Chunk, ast_available, chunk_file, iter_files
from .embedder import Embedder
from .store import delete_file_chunks, get_or_create_table, open_db, upsert_chunks

BATCH_SIZE = 16


def _chunk_id(file_path: str, start: int, end: int) -> str:
    key = f"{file_path}:{start}:{end}"
    return hashlib.sha1(key.encode()).hexdigest()


def _ensure_gitignore(root: Path):
    gitignore = root / ".gitignore"
    entry = ".ccrag/"
    if gitignore.exists():
        content = gitignore.read_text()
        if entry not in content.splitlines():
            gitignore.write_text(content.rstrip("\n") + f"\n{entry}\n")
    else:
        gitignore.write_text(f"{entry}\n")


def index_repo(root: Path, embedder: Embedder, force: bool = False, verbose: bool = True):
    root = root.resolve()
    _ensure_gitignore(root)

    gitignore = root / ".gitignore"
    files = list(iter_files(root, gitignore))

    if verbose:
        print(f"Found {len(files)} files to index under {root}")
        if not ast_available():
            print(
                "Warning: tree-sitter not installed — using line-window chunking "
                "for all files (function/class-boundary chunking disabled). "
                "Install it with: pip install 'ccrag[ast]'"
            )

    total_chunks = 0
    pending: list[Chunk] = []
    table_ref: list = []  # lazy-init container

    def _get_table():
        if not table_ref:
            db = open_db(root)
            table_ref.append(get_or_create_table(db, embedder.dim))
        return table_ref[0]

    def _flush(batch: list[Chunk]):
        import torch
        texts = [c.content for c in batch]
        with torch.no_grad():
            vectors = embedder.embed(texts)
        records = [
            {
                "id": _chunk_id(c.file_path, c.start_line, c.end_line),
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "language": c.language,
                "content": c.content,
                "vector": v.tolist(),
            }
            for c, v in zip(batch, vectors)
        ]
        upsert_chunks(_get_table(), records)
        gc.collect()

    for f in files:
        chunks = chunk_file(f)
        rel = str(f.relative_to(root))
        for c in chunks:
            c.file_path = rel
        pending.extend(chunks)
        while len(pending) >= BATCH_SIZE:
            batch, pending = pending[:BATCH_SIZE], pending[BATCH_SIZE:]
            _flush(batch)
            total_chunks += len(batch)
            if verbose:
                print(f"  {total_chunks} chunks indexed", end="\r")

    if pending:
        _flush(pending)
        total_chunks += len(pending)

    if verbose:
        print(f"\nDone. {total_chunks} chunks stored in {root / '.ccrag'}")
        print(f"Added .ccrag/ to {root / '.gitignore'}")


def index_file(root: Path, file_path: Path, embedder: Embedder):
    """Re-index a single file (used by the watcher)."""
    root = root.resolve()
    db = open_db(root)
    table = get_or_create_table(db, embedder.dim)

    rel = str(file_path.relative_to(root))
    delete_file_chunks(table, rel)

    chunks = chunk_file(file_path)
    if not chunks:
        return

    texts = [c.content for c in chunks]
    vectors = embedder.embed(texts)
    records = [
        {
            "id": _chunk_id(rel, c.start_line, c.end_line),
            "file_path": rel,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "language": c.language,
            "content": c.content,
            "vector": v.tolist(),
        }
        for c, v in zip(chunks, vectors)
    ]
    upsert_chunks(table, records)
