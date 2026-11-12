"""按 A/B/C 分桶挖失败案例。

对默认配置（BM25 512/0）逐题跑一遍，把三类题捞出来：
    A. 完全没检索到：recall@5 = 0（gold 段一个都没进 top-5）
    B. 找到了但排不上第一：recall@5 = 1 且 recall@1 = 0
    C. 只找到一半：0 < recall@5 < 1（多段 gold 只命中一部分）

对每条还顺手算两个诊断信号，省得人肉翻：
    term_hit_gold  = query 的字符 bigram 有多少落在 gold 段里
    term_hit_top1  = 同样口径落在 top-1 命中段里
两者差得越大，越像"术语不同义"（语义对、字面对不上）。

用法：
    py -3 src\find_failures.py                 # 全部三类，写 CSV + 打印前 10
    py -3 src\find_failures.py --all           # 打印全部失败题
    py -3 src\find_failures.py --qid q047      # 只看某一条的 top-5 明细
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import build_index, load_corpus, load_queries, paragraphs   # noqa: E402
from metrics import recall_at_k                                        # noqa: E402
from bm25 import tokenize                                              # noqa: E402

OUT_CSV = "results/failures_512_0.csv"


def bigrams(tokens):
    return {t for t in tokens if len(t) == 2}


def rank_of_gold(ranked, gold):
    """第一条命中 gold 的结果排第几（1 开始）；一条都没有返回 0。"""
    for i, s in enumerate(ranked, start=1):
        if s & gold:
            return i
    return 0


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
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--qid", default="")
    ap.add_argument("--out", default=OUT_CSV)
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    chunks, index = build_index(docs, args.size, args.overlap)

    # 段落原文，用来算"术语命中"并打印 gold 全文
    paras = {doc: paragraphs(text) for doc, text in docs.items()}

    rows = []
    for q in queries:
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        hits = index.search(q["query"], top_k=5)
        ranked = [chunks[i]["covered"] for i, _ in hits]
        r1 = recall_at_k(ranked, gold, 1)
        r5 = recall_at_k(ranked, gold, 5)

        top1 = chunks[hits[0][0]] if hits else None
        top1_para = ""
        if top1 and top1["covered"]:
            doc, pi = sorted(top1["covered"])[0]
            top1_para = f"{doc}/{pi}"

        gold_text = " ".join(docs[d][s:e]
                             for d, pi in sorted(gold)
                             for (s, e) in [paras[d][pi]])
        q_grams = bigrams(tokenize(q["query"]))
        hit_gold = len(q_grams & bigrams(tokenize(gold_text))) if q_grams else 0
        hit_top1 = len(q_grams & bigrams(tokenize(top1["text"]))) if (q_grams and top1) else 0

        rows.append({
            "qid": q["qid"], "type": q.get("type", ""), "year": q.get("year", ""),
            "query": q["query"], "r1": round(r1, 4), "r5": round(r5, 4),
            "gold": ";".join(f"{d}/{p}" for d, p in sorted(gold)),
            "gold_rank": rank_of_gold(ranked, gold),
            "top1": top1_para, "top1_preview": (top1["text"][:48] if top1 else ""),
            "term_gold": hit_gold, "term_top1": hit_top1,
        })

    def bucket(r):
        if r["r5"] == 0:
            return "A_miss"
        if r["r1"] == 0:
            return "B_rank"
        if r["r5"] < 1:
            return "C_partial"
        return ""

    nA = nB = nC = 0
    for r in rows:
        b = bucket(r)
        r["bucket"] = b
        nA += b == "A_miss"
        nB += b == "B_rank"
        nC += b == "C_partial"

    if args.qid:
        for r in rows:
            if r["qid"] != args.qid:
                continue
            print(f"{r['qid']} [{r['type']}] {r['query']}")
            print(f"  gold={r['gold']}  r@1={r['r1']} r@5={r['r5']}  首命中排名={r['gold_rank']}")
            print(f"  术语命中: gold 段 {r['term_gold']} 个 bigram / top-1 段 {r['term_top1']} 个")
        q = next(x for x in queries if x["qid"] == args.qid)
        hits = index.search(q["query"], top_k=5)
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        print("\n  gold 段原文：")
        for d, pi in sorted(gold):
            s, e = paras[d][pi]
            print(f"  ── [{d}/{pi}] {docs[d][s:e][:400]}…")
        print("\n  top-5 明细：")
        for i, (ci, score) in enumerate(hits, 1):
            c = chunks[ci]
            mark = "★" if (c["covered"] & gold) else " "
            pv = c["text"][:56].replace("\n", " ")
            print(f"  {mark} {i}. {score:6.3f} [{c['doc']} {c['start']}-{c['end']}] {pv}…")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total = len(rows)
    print(f"默认配置 BM25 {args.size}/{args.overlap}，共 {total} 条 query")
    print(f"  A 完全没找到（r@5=0）: {nA} 条")
    print(f"  B 找到但排不到第一（r@5=1, r@1=0）: {nB} 条")
    print(f"  C 只找到一部分（0<r@5<1）: {nC} 条")
    print(f"  全部指标: r@1={sum(r['r1'] for r in rows)/total:.4f} "
          f"r@5={sum(r['r5'] for r in rows)/total:.4f}")
    print(f"  明细已写 {args.out}\n")

    show = rows if args.all else [r for r in rows if r["bucket"]]
    show.sort(key=lambda r: (r["bucket"], r["qid"]))
    limit = len(show) if args.all else args.show
    for r in show[:limit]:
        print(f"[{r['bucket']}] {r['qid']} ({r['type']}, {r['year']}) r@1={r['r1']} r@5={r['r5']}")
        print(f"  Q: {r['query']}")
        print(f"  gold: {r['gold']}   首命中排名: {r['gold_rank'] or '未命中'}")
        print(f"  术语命中 bigram: gold 段 {r['term_gold']} / top-1 段 {r['term_top1']}")
        print(f"  top-1: [{r['top1']}] {r['top1_preview']}…")
        print()


if __name__ == "__main__":
    main()
