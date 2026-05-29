from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

if TYPE_CHECKING:
    import lancedb

CCRAG_DIR = ".ccrag"
TABLE_NAME = "chunks"

SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("language", pa.string()),
    pa.field("content", pa.string()),
    pa.field("vector", pa.list_(pa.float32())),
])


def _db_path(root: Path) -> Path:
    return root / CCRAG_DIR / "index"


def open_db(root: Path) -> "lancedb.DBConnection":
    import lancedb
    return lancedb.connect(str(_db_path(root)))


def get_or_create_table(db: "lancedb.DBConnection", dim: int):
    import lancedb
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    vector_schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("language", pa.string()),
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])
    return db.create_table(TABLE_NAME, schema=vector_schema)


def upsert_chunks(table, records: list[dict]):
    if not records:
        return
    # LanceDB merge_insert for upsert by id
    import pandas as pd
    df = pd.DataFrame(records)
    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(df)


def delete_file_chunks(table, file_path: str):
    table.delete(f"file_path = '{file_path}'")


def search(table, query_vec: np.ndarray, n: int = 8) -> list[dict]:
    results = (
        table.search(query_vec.tolist())
        .limit(n)
        .select(["file_path", "start_line", "end_line", "language", "content"])
        .to_list()
    )
    return results


def stats(root: Path) -> dict:
    try:
        db = open_db(root)
        if TABLE_NAME not in db.table_names():
            return {"indexed": False}
        tbl = db.open_table(TABLE_NAME)
        count = tbl.count_rows()
        files = set(r["file_path"] for r in tbl.search().limit(100000).select(["file_path"]).to_list())
        return {"indexed": True, "chunks": count, "files": len(files)}
    except Exception as e:
        return {"indexed": False, "error": str(e)}
