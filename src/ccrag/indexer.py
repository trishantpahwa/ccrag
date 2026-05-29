from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

from .chunker import Chunk, ast_available, chunk_file, iter_files
from .embedder import Embedder
from .store import (
    CCRAG_DIR,
    TABLE_NAME,
    delete_file_chunks,
    get_or_create_table,
    open_db,
    upsert_chunks,
)

BATCH_SIZE = 16
MANIFEST_VERSION = 1


def _chunk_id(file_path: str, start: int, end: int) -> str:
    key = f"{file_path}:{start}:{end}"
    return hashlib.sha1(key.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    """Content hash of a file, used to detect whether it changed since last index."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def _manifest_path(root: Path) -> Path:
    return root / CCRAG_DIR / "manifest.json"


def _load_manifest(root: Path) -> dict:
    p = _manifest_path(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _save_manifest(root: Path, embedder: Embedder, files: dict[str, str]):
    p = _manifest_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "version": MANIFEST_VERSION,
        "model": getattr(embedder, "model_name", None),
        "files": files,
    }, indent=2))


def _ensure_gitignore(root: Path):
    gitignore = root / ".gitignore"
    entry = ".ccrag/"
    if gitignore.exists():
        content = gitignore.read_text()
        if entry not in content.splitlines():
            gitignore.write_text(content.rstrip("\n") + f"\n{entry}\n")
    else:
        gitignore.write_text(f"{entry}\n")


def _make_records(chunks: list[Chunk], vectors) -> list[dict]:
    return [
        {
            "id": _chunk_id(c.file_path, c.start_line, c.end_line),
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "language": c.language,
            "content": c.content,
            "vector": v.tolist(),
        }
        for c, v in zip(chunks, vectors)
    ]


def index_repo(root: Path, embedder: Embedder, force: bool = False, verbose: bool = True):
    """Index a repo, re-embedding only files whose content changed since last run.

    A manifest of per-file content hashes lives in ``.ccrag/manifest.json``.
    Unchanged files are skipped entirely (the embedding step is the expensive
    part), changed files are re-chunked and re-embedded, and files removed from
    disk have their chunks pruned. Switching the embedding model triggers a full
    rebuild, since old vectors are incompatible.
    """
    root = root.resolve()
    _ensure_gitignore(root)

    gitignore = root / ".gitignore"
    files = list(iter_files(root, gitignore))

    db = open_db(root)
    table_exists = TABLE_NAME in db.table_names()
    manifest = _load_manifest(root)
    model_changed = bool(manifest) and manifest.get("model") != getattr(embedder, "model_name", None)

    if model_changed and table_exists:
        # Embeddings from the previous model (and possibly a different dimension)
        # are no longer comparable — rebuild from scratch.
        db.drop_table(TABLE_NAME)
        table_exists = False

    previous = {} if (not table_exists or model_changed) else manifest.get("files", {})

    current: dict[str, str] = {}
    rel_to_path: dict[str, Path] = {}
    for f in files:
        rel = str(f.relative_to(root))
        current[rel] = _file_hash(f)
        rel_to_path[rel] = f

    changed = [rel for rel in current if previous.get(rel) != current[rel]]
    deleted = [rel for rel in previous if rel not in current]

    if verbose:
        unchanged = len(current) - len(changed)
        print(f"Found {len(files)} files ({len(changed)} new/changed, "
              f"{unchanged} unchanged, {len(deleted)} removed)")
        if changed and not ast_available():
            print(
                "Warning: tree-sitter not installed — using line-window chunking "
                "for all files (function/class-boundary chunking disabled). "
                "Install it with: pip install 'ccrag[ast]'"
            )

    if not changed and not deleted:
        if verbose:
            print("Index already up to date.")
        _save_manifest(root, embedder, current)
        return

    table = get_or_create_table(db, embedder.dim)

    # Drop chunks for removed files, and for changed files (re-inserted below).
    for rel in deleted:
        delete_file_chunks(table, rel)
    for rel in changed:
        delete_file_chunks(table, rel)

    total_chunks = 0
    pending: list[Chunk] = []

    def _flush(batch: list[Chunk]):
        nonlocal total_chunks
        import torch
        texts = [c.content for c in batch]
        with torch.no_grad():
            vectors = embedder.embed(texts)
        upsert_chunks(table, _make_records(batch, vectors))
        gc.collect()
        total_chunks += len(batch)
        if verbose:
            print(f"  {total_chunks} chunks embedded", end="\r")

    for rel in changed:
        chunks = chunk_file(rel_to_path[rel])
        for c in chunks:
            c.file_path = rel
        pending.extend(chunks)
        while len(pending) >= BATCH_SIZE:
            batch, pending = pending[:BATCH_SIZE], pending[BATCH_SIZE:]
            _flush(batch)

    if pending:
        _flush(pending)

    _save_manifest(root, embedder, current)

    if verbose:
        print(f"\nDone. {total_chunks} chunks from {len(changed)} file(s) embedded, "
              f"{len(deleted)} file(s) removed. Index at {root / CCRAG_DIR}")


def index_file(root: Path, file_path: Path, embedder: Embedder):
    """Re-index a single file (used by the watcher)."""
    root = root.resolve()
    db = open_db(root)
    table = get_or_create_table(db, embedder.dim)

    rel = str(file_path.relative_to(root))
    delete_file_chunks(table, rel)

    chunks = chunk_file(file_path)
    if chunks:
        for c in chunks:
            c.file_path = rel
        vectors = embedder.embed([c.content for c in chunks])
        upsert_chunks(table, _make_records(chunks, vectors))

    # Keep the manifest in sync so a later full `index` skips this file.
    manifest = _load_manifest(root)
    files = manifest.get("files", {})
    files[rel] = _file_hash(file_path)
    _save_manifest(root, embedder, files)
