import pytest

from ccrag.indexer import _chunk_id, _ensure_gitignore, index_file, index_repo
from ccrag.store import TABLE_NAME, open_db, stats


def _lines(n: int, word: str = "line") -> str:
    return "\n".join(f"{word} {i} with sufficient content here" for i in range(n))


def _table(tmp_path):
    return open_db(tmp_path).open_table(TABLE_NAME)


def test_chunk_id_deterministic_and_unique():
    a = _chunk_id("f.py", 1, 10)
    b = _chunk_id("f.py", 1, 10)
    c = _chunk_id("f.py", 1, 11)
    assert a == b
    assert a != c


def test_ensure_gitignore_adds_entry_once(tmp_path):
    _ensure_gitignore(tmp_path)
    _ensure_gitignore(tmp_path)
    assert (tmp_path / ".gitignore").read_text().count(".ccrag/") == 1


def test_index_repo_indexes_files_and_updates_gitignore(tmp_path, fake_embedder):
    (tmp_path / "a.txt").write_text(_lines(80))
    (tmp_path / "b.txt").write_text(_lines(50, "row"))
    index_repo(tmp_path, fake_embedder, verbose=False)

    s = stats(tmp_path)
    assert s["indexed"] is True
    assert s["files"] == 2
    assert s["chunks"] > 0
    assert ".ccrag/" in (tmp_path / ".gitignore").read_text()


def test_index_file_replaces_stale_chunks(tmp_path, fake_embedder):
    f = tmp_path / "a.txt"
    f.write_text(_lines(80))
    index_file(tmp_path, f, fake_embedder)
    assert _table(tmp_path).count_rows() >= 2

    # Shrink the file: the watcher path deletes existing chunks before inserting.
    f.write_text(_lines(20))
    index_file(tmp_path, f, fake_embedder)

    rows = _table(tmp_path).search().limit(100).select(["end_line"]).to_list()
    assert rows
    assert all(r["end_line"] <= 20 for r in rows)


@pytest.mark.xfail(
    reason="index_repo upserts by chunk id and never prunes; stale chunks from "
    "changed files linger. Only the watcher (index_file) deletes before insert.",
    strict=False,
)
def test_index_repo_prunes_stale_chunks(tmp_path, fake_embedder):
    f = tmp_path / "a.txt"
    f.write_text(_lines(80))
    index_repo(tmp_path, fake_embedder, verbose=False)

    f.write_text(_lines(20))
    index_repo(tmp_path, fake_embedder, verbose=False)

    rows = _table(tmp_path).search().limit(100).select(["start_line"]).to_list()
    assert all(r["start_line"] <= 20 for r in rows)
