"""Q3：rerank 单点实验 —— BM25 召回 top-N，cross-encoder 重排，看值不值。

问题定义
--------
先 BM25 取 top-20，再用 cross-encoder 给每条 (query, chunk 文本) 打相关性分，
重排后取前 5。基线是同一批候选、不重排的 BM25（512/0、recall@5 0.8403）。

为什么要同时报两组数
--------------------
rerank 动的是排序，不是召回，所以 recall@5 可能几乎不动，价值主要在 recall@1 和 MRR；
代价则全在延迟——20 条候选各跑一次前向，这才是它贵的根源。
"同一批候选"很重要：如果直接跟全量 BM25 比，涨的可能是召回而不是排序。

用法
----
    .\\.venv\\Scripts\\python.exe src\\rerank_eval.py
    .\\.venv\\Scripts\\python.exe src\\rerank_eval.py --topn 10
    .\\.venv\\Scripts\\python.exe src\\rerank_eval.py --model BAAI/bge-reranker-base

注意：必须在 .venv 里跑（要 torch + sentence-transformers）；首次运行会下载约 1GB 模型。
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import build_index, load_corpus, load_queries  # noqa: E402
from metrics import mrr_at_k, ndcg_at_k, recall_at_k   # noqa: E402


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * 0.95))]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--topn", type=int, default=20, help="先召回多少条交给 cross-encoder")
    ap.add_argument("--model", default="BAAI/bge-reranker-base")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    chunks, index = build_index(docs, args.size, args.overlap)
    print(f"语料 {len(docs)} 篇，{args.size}/{args.overlap} → {len(chunks)} chunks")

    from sentence_transformers import CrossEncoder
    print(f"加载 reranker: {args.model}（首次运行会下载约 1GB）", flush=True)
    model = CrossEncoder(args.model, max_length=512)

    rows, base_ms, rr_ms = [], [], []
    for q in queries:
        gold = {(g["doc"], int(g["para"])) for g in q["gold"]}
        if not gold:
            continue
        text = q["query"]

        t0 = time.perf_counter()
        hits = index.search(text, top_k=args.topn)            # [(chunk_idx, score)]
        base_ms.append((time.perf_counter() - t0) * 1000)

        # 同一段文本可能被多个 chunk 覆盖（滑窗重叠），去重后再送 cross-encoder
        cands, seen = [], set()
        for ci, _ in hits:
            body = chunks[ci]["text"].strip()
            if body and body not in seen:
                seen.add(body)
                cands.append(ci)
        if not cands:
            continue

        base_sets = [chunks[ci]["covered"] & gold for ci in cands]

        t0 = time.perf_counter()
        scores = model.predict([(text, chunks[ci]["text"]) for ci in cands])
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
        rr_ms.append((time.perf_counter() - t0) * 1000)

        rr_sets = [chunks[cands[i]]["covered"] & gold for i in order]
        rows.append({
            "qid": q["qid"], "type": q.get("type", ""),
            "base_r1": recall_at_k(base_sets, gold, 1),
            "base_r5": recall_at_k(base_sets, gold, args.k),
            "base_mrr": mrr_at_k(base_sets, gold, args.k),
            "base_ndcg": ndcg_at_k(base_sets, gold, args.k),
            "rr_r1": recall_at_k(rr_sets, gold, 1),
            "rr_r5": recall_at_k(rr_sets, gold, args.k),
            "rr_mrr": mrr_at_k(rr_sets, gold, args.k),
            "rr_ndcg": ndcg_at_k(rr_sets, gold, args.k),
            "rr_ms": rr_ms[-1],
        })

    n = len(rows)

    def avg(key):
        return sum(r[key] for r in rows) / n

    print(f"\n{args.topn} 召回 → {args.k}（{n} 条 query，同一批候选内比较）")
    print(f"{'':12}{'recall@1':>10}{'recall@5':>10}{'MRR':>8}{'nDCG@5':>9}")
    print(f"{'BM25 基线':10}{avg('base_r1'):>10.4f}{avg('base_r5'):>10.4f}"
          f"{avg('base_mrr'):>8.4f}{avg('base_ndcg'):>9.4f}")
    print(f"{'+rerank':10}{avg('rr_r1'):>10.4f}{avg('rr_r5'):>10.4f}"
          f"{avg('rr_mrr'):>8.4f}{avg('rr_ndcg'):>9.4f}")
    print(f"{'差值':10}{avg('rr_r1') - avg('base_r1'):>+10.4f}"
          f"{avg('rr_r5') - avg('base_r5'):>+10.4f}"
          f"{avg('rr_mrr') - avg('base_mrr'):>+8.4f}"
          f"{avg('rr_ndcg') - avg('base_ndcg'):>+9.4f}")
    print(f"\n延迟 P95：BM25 召回 {p95(base_ms):.2f} ms  |  "
          f"召回+重排 {p95([a + b for a, b in zip(base_ms, rr_ms)]):.2f} ms")

    up = sum(1 for r in rows if r["rr_r5"] > r["base_r5"])
    down = sum(1 for r in rows if r["rr_r5"] < r["base_r5"])
    up1 = sum(1 for r in rows if r["rr_r1"] > r["base_r1"])
    print(f"逐题 recall@5：涨 {up} / 跌 {down} / 平 {n - up - down}；"
          f"recall@1 涨 {up1} 条")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"已写出 {args.out}")


if __name__ == "__main__":
    main()
