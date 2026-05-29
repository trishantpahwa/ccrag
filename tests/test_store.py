from ccrag.store import (
    TABLE_NAME,
    delete_file_chunks,
    get_or_create_table,
    open_db,
    search,
    stats,
    upsert_chunks,
)


def _table(tmp_path, dim):
    db = open_db(tmp_path)
    return get_or_create_table(db, dim)


def test_get_or_create_table_is_idempotent(tmp_path, fake_embedder):
    db = open_db(tmp_path)
    t1 = get_or_create_table(db, fake_embedder.dim)
    assert TABLE_NAME in db.table_names()
    t2 = get_or_create_table(db, fake_embedder.dim)
    assert t1.name == t2.name


def test_upsert_inserts_rows(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(tbl, [make_record("1", "alpha"), make_record("2", "beta")])
    assert tbl.count_rows() == 2


def test_upsert_dedupes_and_updates_by_id(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(tbl, [make_record("x", "old content")])
    upsert_chunks(tbl, [make_record("x", "new content")])
    assert tbl.count_rows() == 1
    rows = tbl.search().limit(10).select(["id", "content"]).to_list()
    assert rows[0]["content"] == "new content"


def test_delete_file_chunks(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(
        tbl,
        [
            make_record("1", "a", file_path="a.py"),
            make_record("2", "b", file_path="a.py"),
            make_record("3", "c", file_path="b.py"),
        ],
    )
    delete_file_chunks(tbl, "a.py")
    rows = tbl.search().limit(10).select(["file_path"]).to_list()
    assert tbl.count_rows() == 1
    assert rows[0]["file_path"] == "b.py"


def test_search_ranks_exact_match_first(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(
        tbl,
        [
            make_record("1", "alpha apple fruit"),
            make_record("2", "beta banana yellow"),
            make_record("3", "gamma grape vine"),
        ],
    )
    qv = fake_embedder.embed_one("beta banana yellow")
    res = search(tbl, qv, n=5)
    assert res[0]["content"] == "beta banana yellow"
    assert set(res[0]) >= {"file_path", "start_line", "end_line", "language", "content"}


def test_search_respects_limit(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(tbl, [make_record(str(i), f"text {i}") for i in range(10)])
    assert len(search(tbl, fake_embedder.embed_one("text 3"), n=4)) == 4


def test_stats_indexed(tmp_path, fake_embedder, make_record):
    tbl = _table(tmp_path, fake_embedder.dim)
    upsert_chunks(
        tbl,
        [
            make_record("1", "a", file_path="a.py"),
            make_record("2", "b", file_path="a.py"),
            make_record("3", "c", file_path="b.py"),
        ],
    )
    s = stats(tmp_path)
    assert s["indexed"] is True
    assert s["chunks"] == 3
    assert s["files"] == 2


def test_stats_not_indexed(tmp_path):
    assert stats(tmp_path) == {"indexed": False}
