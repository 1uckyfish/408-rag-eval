"""网格实验：size x overlap，输出一张 CSV。

用法：
    python3 src/run_grid.py --corpus data/corpus --queries data/queries.jsonl \
        --out results/grid_bm25.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval import load_corpus, load_queries, evaluate   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--sizes", default="128,256,512,1024")
    ap.add_argument("--overlaps", default="0,64")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/grid_bm25.csv")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    sizes = [int(x) for x in args.sizes.split(",")]
    overlaps = [int(x) for x in args.overlaps.split(",")]

    rows = []
    for size in sizes:
        for ov in overlaps:
            if ov >= size:
                continue
            _, s = evaluate(docs, queries, size, ov, args.k)
            rows.append({
                "retriever": "bm25", "size": size, "overlap": ov, "k": args.k,
                "chunks": s["chunks"], "recall": round(s["recall@k"], 4),
                "mrr": round(s["mrr@k"], 4), "ndcg": round(s["ndcg@k"], 4),
                "p95_ms": round(s["p95_ms"], 2),
            })
            print(rows[-1])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n已写出 {args.out}（{len(rows)} 组配置）")


if __name__ == "__main__":
    main()
