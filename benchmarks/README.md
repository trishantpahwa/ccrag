# Benchmarks

`benchmark.py` measures ccrag's indexing throughput and retrieval quality
end-to-end, using the **real** embedding model.

```bash
pip install -e '.[ast]'
python benchmarks/benchmark.py            # benchmark ccrag on itself
python benchmarks/benchmark.py /some/repo  # benchmark another repo
python benchmarks/benchmark.py --top-k 5
```

It copies the target repo's indexable files into a temp directory and indexes
there, so your real `.ccrag/` index is never modified. The first run downloads
the `all-MiniLM-L6-v2` weights.

Reported metrics:

| Metric | Meaning |
|---|---|
| chunks/s, files/s | indexing throughput (model load excluded) |
| recall@k | fraction of golden queries whose expected file appears in the top-k |
| MRR | mean reciprocal rank of the first correct file |
| latency p50/p95/mean | per-query embed + vector-search time |

The golden query set in `GOLDEN` is tuned for ccrag's own source tree; edit it
when benchmarking a different repo.
