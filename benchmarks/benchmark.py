#!/usr/bin/env python
"""Benchmark ccrag indexing throughput and retrieval quality.

Runs against a target repo (default: ccrag itself) using the REAL embedding
model, so the first run downloads model weights. Indexing happens in a
throwaway temp copy of the repo's indexable files, so your real `.ccrag/`
index is never touched.

Reports:
  * indexing time and throughput (chunks/s, files/s)
  * search latency (p50 / p95 / mean)
  * retrieval quality over a golden query set (recall@k, MRR)

Usage:
    python benchmarks/benchmark.py [PATH] [--top-k 8] [--model NAME]
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ccrag.chunker import ast_available, iter_files  # noqa: E402
from ccrag.embedder import Embedder  # noqa: E402
from ccrag.indexer import index_repo  # noqa: E402
from ccrag.store import TABLE_NAME, open_db, search  # noqa: E402

# (query, substring the top-ranked file path is expected to contain).
# Tuned for ccrag's own source tree.
GOLDEN: list[tuple[str, str]] = [
    ("how does chunking split code at function boundaries", "chunker.py"),
    ("generate embeddings for text with sentence transformers", "embedder.py"),
    ("start the MCP server over stdio", "server.py"),
    ("upsert chunks into the vector store with merge insert", "store.py"),
    ("vector similarity search returning the top k results", "store.py"),
    ("watch the filesystem and re-index changed files", "cli.py"),
    ("command line interface entry point and subcommands", "cli.py"),
    ("tree-sitter node types per programming language", "chunker.py"),
    ("compute index statistics: number of files and chunks", "store.py"),
    ("print the MCP config snippet for claude settings", "cli.py"),
]


def _copy_indexable(src_root: Path, dst_root: Path) -> int:
    gi = src_root / ".gitignore"
    n = 0
    for f in iter_files(src_root, gi if gi.exists() else None):
        dst = dst_root / f.relative_to(src_root)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst)
        n += 1
    return n


def _percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    idx = max(0, min(len(s) - 1, round(pct / 100 * len(s)) - 1))
    return s[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?",
                    default=str(Path(__file__).resolve().parent.parent),
                    help="repo to benchmark (default: ccrag itself)")
    ap.add_argument("--top-k", type=int, default=8, help="results per query")
    ap.add_argument("--model", default=None, help="sentence-transformers model name")
    args = ap.parse_args()

    target = Path(args.path).resolve()
    print(f"Target repo : {target}")
    print(f"AST chunking: {'on' if ast_available() else 'OFF (line-window fallback)'}")

    tmp = Path(tempfile.mkdtemp(prefix="ccrag-bench-"))
    try:
        copied = _copy_indexable(target, tmp)
        print(f"Indexable files: {copied}\n")

        embedder = Embedder(args.model) if args.model else Embedder()
        t0 = time.perf_counter()
        embedder.embed_one("warm up")  # exclude model download/load from timings
        print(f"Model load + warmup: {time.perf_counter() - t0:.2f}s\n")

        # --- Indexing ---
        t0 = time.perf_counter()
        index_repo(tmp, embedder, verbose=False)
        index_s = time.perf_counter() - t0

        tbl = open_db(tmp).open_table(TABLE_NAME)
        chunks = tbl.count_rows()
        print("== Indexing ==")
        print(f"  files       : {copied}")
        print(f"  chunks      : {chunks}")
        print(f"  time        : {index_s:.2f}s")
        print(f"  throughput  : {chunks / index_s:.1f} chunks/s, "
              f"{copied / index_s:.1f} files/s\n")

        # --- Search latency + quality ---
        k = args.top_k
        latencies_ms: list[float] = []
        reciprocal_ranks: list[float] = []
        hits = 0
        rows = []
        for query, expect in GOLDEN:
            t0 = time.perf_counter()
            res = search(tbl, embedder.embed_one(query), n=k)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            paths = [r["file_path"] for r in res]
            rank = next((i + 1 for i, p in enumerate(paths) if expect in p), None)
            hits += int(rank is not None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            rows.append((query, expect, paths[0] if paths else "-", rank))

        print("== Retrieval (golden set) ==")
        for query, expect, top1, rank in rows:
            mark = f"rank {rank}" if rank else "MISS"
            print(f"  [{mark:>7}] want {expect:<11} got {top1:<26} :: {query}")

        n = len(GOLDEN)
        print("\n== Metrics ==")
        print(f"  queries      : {n}")
        print(f"  recall@{k:<6}: {hits / n:.2f}")
        print(f"  MRR          : {statistics.mean(reciprocal_ranks):.3f}")
        print(f"  latency p50  : {statistics.median(latencies_ms):.1f} ms")
        print(f"  latency p95  : {_percentile(latencies_ms, 95):.1f} ms")
        print(f"  latency mean : {statistics.mean(latencies_ms):.1f} ms")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
