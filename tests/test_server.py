import pytest

import ccrag.server as srv
from ccrag.store import get_or_create_table, open_db, upsert_chunks


@pytest.fixture
def indexed_root(tmp_path, fake_embedder, make_record):
    db = open_db(tmp_path)
    tbl = get_or_create_table(db, fake_embedder.dim)
    upsert_chunks(
        tbl,
        [
            make_record("1", "def parse_tree(): ...", file_path="src/chunker.py",
                        start=10, end=20, lang="python"),
            make_record("2", "def embed_text(): ...", file_path="src/embedder.py",
                        start=5, end=12, lang="python"),
        ],
    )
    return tmp_path


def _wire(monkeypatch, root, embedder):
    monkeypatch.setattr(srv, "_root", root)
    monkeypatch.setattr(srv, "_embedder", embedder)


def test_search_codebase_returns_formatted_hits(indexed_root, fake_embedder, monkeypatch):
    _wire(monkeypatch, indexed_root, fake_embedder)
    out = srv.search_codebase("def embed_text(): ...", n_results=5)
    assert "src/embedder.py" in out
    assert "lines" in out
    assert "```python" in out


def test_codebase_stats_reports_counts(indexed_root, fake_embedder, monkeypatch):
    _wire(monkeypatch, indexed_root, fake_embedder)
    out = srv.codebase_stats()
    assert "2 chunks" in out
    assert "files" in out


def test_search_codebase_without_index(tmp_path, fake_embedder, monkeypatch):
    _wire(monkeypatch, tmp_path, fake_embedder)
    out = srv.search_codebase("anything")
    assert "index not found" in out.lower()
