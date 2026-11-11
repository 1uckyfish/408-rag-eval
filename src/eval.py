"""单配置评测：切分 -> 建索引 -> 检索 -> 打分。

用法：
    python3 src/eval.py --corpus data/corpus --queries data/queries.jsonl \
        --size 300 --overlap 60 --k 5

输出：每题一行 + 汇总；--out 可写 CSV。
"""

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import paragraphs, chunk          # noqa: E402
from bm25 import BM25, tokenize                # noqa: E402
from metrics import recall_at_k, mrr_at_k, ndcg_at_k  # noqa: E402


def load_corpus(dir_path):
    docs = {}
    for fn in sorted(os.listdir(dir_path)):
        if fn.lower().endswith((".md", ".txt")):
            name = os.path.splitext(fn)[0]
            with open(os.path.join(dir_path, fn), encoding="utf-8") as f:
                docs[name] = f.read()
    return docs


def load_queries(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_index(docs, size, overlap):
    """返回 (chunks, bm25)，chunks 里带 doc/start/end/covered。"""
    chunks, texts = [], []
    for doc, text in docs.items():
        paras = paragraphs(text)
        mids = [(s + e) / 2 for (s, e) in paras]
        for (s, e) in chunk(text, size, overlap):
            covered = {(doc, i) for i, m in enumerate(mids) if s <= m < e}
            chunks.append({"doc": doc, "start": s, "end": e,
                           "text": text[s:e], "covered": covered})
            texts.append(tokenize(text[s:e]))
    return chunks, BM25(texts)


def evaluate(docs, queries, size, overlap, k, warmup=3, repeat=5):
    chunks, index = build_index(docs, size, overlap)
    rows = []
    lat = []
    for q in queries:
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        t0 = time.perf_counter()
        hits = index.search(q["query"], top_k=k)
        dt = (time.perf_counter() - t0) * 1000
        for _ in range(warmup):
            index.search(q["query"], top_k=k)
        lat.append(dt)
        ranked = [chunks[i]["covered"] for i, _ in hits]
        rows.append({
            "qid": q["qid"],
            "type": q.get("type", ""),
            "recall": recall_at_k(ranked, gold, k),
            "mrr": mrr_at_k(ranked, gold, k),
            "ndcg": ndcg_at_k(ranked, gold, k),
            "latency_ms": dt,
        })
    n = max(1, len(rows))
    summary = {
        "config": f"bm25 size={size} overlap={overlap} k={k}",
        "chunks": len(chunks),
        "recall@k": sum(r["recall"] for r in rows) / n,
        "mrr@k": sum(r["mrr"] for r in rows) / n,
        "ndcg@k": sum(r["ndcg"] for r in rows) / n,
        "p95_ms": sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
    }
    return rows, summary


def main():
    try:                      # Windows 控制台默认 cp936，中文会变问号
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    rows, s = evaluate(docs, queries, args.size, args.overlap, args.k)

    print(f"# 语料 {len(docs)} 篇 / {sum(len(t) for t in docs.values())} 字"
          f" | query {len(queries)} 条 | chunk {s['chunks']} 个")
    print(f"# {s['config']}")
    print(f"{'qid':<6}{'type':<6}{'recall':>8}{'mrr':>8}{'ndcg':>8}{'ms':>9}")
    for r in rows:
        print(f"{r['qid']:<6}{r['type']:<6}{r['recall']:>8.3f}"
              f"{r['mrr']:>8.3f}{r['ndcg']:>8.3f}{r['latency_ms']:>9.2f}")
    print("-" * 45)
    print(f"{'MEAN':<12}{s['recall@k']:>8.3f}{s['mrr@k']:>8.3f}"
          f"{s['ndcg@k']:>8.3f}{s['p95_ms']:>9.2f}   (P95)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n已写出 {args.out}")


if __name__ == "__main__":
    main()
