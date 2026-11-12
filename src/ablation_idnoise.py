"""顺手做的消融实验：把进程编号从 query 里剥掉，看指标变化。

动机见 docs/06 案例 2：q008 的 "P1、P2、P3" 把一条同步题拽进了银行家算法章节，
因为这些编号在死锁/银行家算例里同样高频，却不是内容词。

做法（只在 query 侧剥，语料不动）：
    把 P1/P2/p1… 和 进程1/进程2 统一替换成空串，再跑一遍 BM25 512/0、k=5。

用法：
    py -3 src/ablation_idnoise.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import build_index, load_queries, load_corpus      # noqa: E402
from metrics import mrr_at_k, ndcg_at_k, recall_at_k          # noqa: E402

PAT = re.compile(r"\b[Pp]\d+\b|进程\s*\d+")
SIZE, OVERLAP, K = 512, 0, 5


def strip_ids(text):
    return re.sub(r"\s+", " ", PAT.sub(" ", text)).strip()


def run(docs, queries, transform=None):
    chunks, index = build_index(docs, SIZE, OVERLAP)
    r1 = r5 = mrr = ndcg = 0.0
    for q in queries:
        text = transform(q["query"]) if transform else q["query"]
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        hits = index.search(text, top_k=K)
        ranked = [chunks[i]["covered"] for i, _ in hits]
        r1 += recall_at_k(ranked, gold, 1)
        r5 += recall_at_k(ranked, gold, K)
        mrr += mrr_at_k(ranked, gold, K)
        ndcg += ndcg_at_k(ranked, gold, K)
    n = max(1, len(queries))
    return {"r1": r1 / n, "r5": r5 / n, "mrr": mrr / n, "ndcg": ndcg / n}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    docs = load_corpus("data/corpus")
    queries = load_queries("data/queries.jsonl")

    changed = [q["qid"] for q in queries if strip_ids(q["query"]) != q["query"]]
    base = run(docs, queries)
    noid = run(docs, queries, strip_ids)

    print(f"配置：BM25 {SIZE}/{OVERLAP}，k={K}，{len(queries)} 条 query")
    print(f"被改写的 query：{len(changed)} 条 -> {', '.join(changed)}")
    print()
    print(f"{'':<10}{'r@1':>8}{'r@5':>8}{'MRR':>8}{'nDCG':>8}")
    print(f"{'baseline':<10}{base['r1']:>8.4f}{base['r5']:>8.4f}"
          f"{base['mrr']:>8.4f}{base['ndcg']:>8.4f}")
    print(f"{'去编号':<10}{noid['r1']:>8.4f}{noid['r5']:>8.4f}"
          f"{noid['mrr']:>8.4f}{noid['ndcg']:>8.4f}")
    print(f"{'Δ':<10}{noid['r1']-base['r1']:>+8.4f}{noid['r5']-base['r5']:>+8.4f}"
          f"{noid['mrr']-base['mrr']:>+8.4f}{noid['ndcg']-base['ndcg']:>+8.4f}")


if __name__ == "__main__":
    main()
