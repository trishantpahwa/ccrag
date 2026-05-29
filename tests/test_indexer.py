from ccrag.indexer import _chunk_id, _ensure_gitignore, index_file, index_repo
from ccrag.store import TABLE_NAME, open_db, search, stats


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


def test_index_repo_prunes_stale_chunks(tmp_path, fake_embedder):
    f = tmp_path / "a.txt"
    f.write_text(_lines(80))
    index_repo(tmp_path, fake_embedder, verbose=False)

    f.write_text(_lines(20))
    index_repo(tmp_path, fake_embedder, verbose=False)

    rows = _table(tmp_path).search().limit(100).select(["start_line"]).to_list()
    assert all(r["start_line"] <= 20 for r in rows)


def test_index_repo_skips_unchanged_files(tmp_path, fake_embedder):
    (tmp_path / "a.txt").write_text(_lines(80))
    (tmp_path / "b.txt").write_text(_lines(50, "row"))
    index_repo(tmp_path, fake_embedder, verbose=False)
    assert fake_embedder.embedded_texts  # everything embedded on first pass
    count = len(fake_embedder.embedded_texts)

    # Re-index with no changes: nothing should be re-embedded.
    index_repo(tmp_path, fake_embedder, verbose=False)
    assert len(fake_embedder.embedded_texts) == count


def test_index_repo_reindexes_only_changed_file(tmp_path, fake_embedder):
    (tmp_path / "a.txt").write_text(_lines(80))
    (tmp_path / "b.txt").write_text(_lines(80, "row"))
    index_repo(tmp_path, fake_embedder, verbose=False)
    baseline = len(fake_embedder.embedded_texts)

    (tmp_path / "a.txt").write_text(_lines(80, "changed"))
    index_repo(tmp_path, fake_embedder, verbose=False)

    # Only a.txt's two windowed chunks are re-embedded; b.txt is skipped.
    assert len(fake_embedder.embedded_texts) - baseline == 2
    assert stats(tmp_path)["files"] == 2


def test_index_repo_removes_deleted_files(tmp_path, fake_embedder):
    (tmp_path / "a.txt").write_text(_lines(80))
    (tmp_path / "b.txt").write_text(_lines(50, "row"))
    index_repo(tmp_path, fake_embedder, verbose=False)
    assert stats(tmp_path)["files"] == 2

    (tmp_path / "b.txt").unlink()
    index_repo(tmp_path, fake_embedder, verbose=False)

    s = stats(tmp_path)
    assert s["files"] == 1
    rows = _table(tmp_path).search().limit(100).select(["file_path"]).to_list()
    assert all(r["file_path"] == "a.txt" for r in rows)


def test_index_repo_rebuilds_on_model_change(tmp_path, fake_embedder_cls):
    (tmp_path / "a.txt").write_text(_lines(80))

    old = fake_embedder_cls(model_name="model-A", dim=8)
    index_repo(tmp_path, old, verbose=False)

    # Different model (and dimension): the index must rebuild, not error on the
    # stale 8-dim table, and re-embed everything under the new model.
    new = fake_embedder_cls(model_name="model-B", dim=16)
    index_repo(tmp_path, new, verbose=False)

    assert new.embedded_texts
    assert stats(tmp_path)["files"] == 1
    # New 16-dim vectors are queryable, proving the table was recreated.
    assert search(_table(tmp_path), new.embed_one("anything"), n=1)


def test_index_file_updates_manifest_so_full_index_skips(tmp_path, fake_embedder):
    f = tmp_path / "a.txt"
    f.write_text(_lines(80))
    index_file(tmp_path, f, fake_embedder)
    count = len(fake_embedder.embedded_texts)

    # A subsequent full index should treat the watcher-indexed file as unchanged.
    index_repo(tmp_path, fake_embedder, verbose=False)
    assert len(fake_embedder.embedded_texts) == count
