"""把项目当"检索系统"用：给一个 408 问题，打印最相关的 5 段原文。

这是评测台的"演示入口"，不是产品：
    评测台（eval.py / run_grid.py）回答的是"哪种配置更好"，
    这个脚本回答的是"这条问题现在能检索到什么"，方便你亲眼看效果、
    也方便现场演示。

用法：
    py -3 src/ask.py "页表项 2B、页大小 2^10B 时页目录最少多少项"
    py -3 src/ask.py "进程和线程的区别" --k 3 --size 512
    py -3 src/ask.py --interactive            # 连续问，输入 q 退出

参数：
    --size / --overlap  切分配置，默认 512/0（当前基线最优的可解释点）
    --k                 返回前几段，默认 5
    --full              打印整段原文，默认只打印前 120 字
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bm25 import tokenize                                    # noqa: E402
from eval import build_index, load_corpus, paragraphs        # noqa: E402


def best_para(docs, paras, covered, query):
    """chunk 可能盖住好几段，挑与问题词重合最多的那一段当作"被引用的段落"。"""
    qs = set(tokenize(query))
    best, best_hit = None, -1
    for doc, idx in covered:
        s, e = paras[doc][idx]
        hit = len(qs & set(tokenize(docs[doc][s:e])))
        if hit > best_hit:
            best, best_hit = (doc, idx), hit
    return best


def fmt_para(docs, paras, doc, idx, width=None):
    s, e = paras[doc][idx]
    text = docs[doc][s:e]
    return text if width is None else text[:width] + ("…" if len(text) > width else "")


def cite(index, chunks, docs, paras, query, k):
    """引用式回答：先给"答案"（原文段落），再给出处和相关度。

    注意：这里没有大模型，所谓"答案"就是检索到的那段原文照抄 ——
    这是有意的：评测对象是检索，生成会引入第二个变量，把问题搅浑。
    """
    hits = index.search(query, top_k=k)
    if not hits:
        print("（没有命中任何段落）")
        return
    ci, score = hits[0]
    c = chunks[ci]
    doc, idx = best_para(docs, paras, c["covered"], query)
    s, e = paras[doc][idx]
    body = docs[doc][s:e].replace("\n", "")
    ps = sorted(p for d, p in c["covered"] if d == c["doc"])

    print(f"\n问题：{query}")
    print("─" * 72)
    print(f"根据 {doc} 第 {idx} 段（共 {len(body)} 字）：\n")
    print("  " + body)
    print()
    print(f"── 出处：{doc} 第 {idx} 段 · 相关度 {score:.2f} · 该 chunk 覆盖第 "
          f"{ps[0]}–{ps[-1]} 段" if ps else f"── 出处：{doc} 第 {idx} 段 · 相关度 {score:.2f}")
    others = []
    for i, sc in hits[1:4]:
        c2 = chunks[i]
        d2, p2 = best_para(docs, paras, c2["covered"], query)
        others.append(f"{d2} 第 {p2} 段（{sc:.1f}）")
    if others:
        print(f"── 其他候选：{'、'.join(others)}")
    print("─" * 72)
    print("说明：答案即检索到的原文段落，未做生成式改写（BM25 无生成能力）。")


def ask(index, chunks, docs, paras, query, k, full):
    hits = index.search(query, top_k=k)
    if not hits:
        print("（没有命中任何段落）")
        return
    print(f"\n查询：{query}")
    print("=" * 72)
    for rank, (ci, score) in enumerate(hits, 1):
        c = chunks[ci]
        # 一个 chunk 可能覆盖多个段落，把命中的段号一并列出，便于对照 gold
        ps = sorted(p for d, p in c["covered"] if d == c["doc"])
        span = f"第 {ps[0]}–{ps[-1]} 段" if ps else "（无完整段落）"
        print(f"[{rank}] {score:7.2f}  {c['doc']}  {span}  "
              f"（字符 {c['start']}–{c['end']}）")
        body = c["text"] if full else c["text"][:120].replace("\n", " ") + "…"
        print(f"    {body}")
        print()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    # Windows 下管道输入按 cp936 解码会把中文问题变成乱码，统一成 utf-8
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    ap = argparse.ArgumentParser(description="在 408 语料上做一次 BM25 检索")
    ap.add_argument("query", nargs="*", help="问题，可带空格（建议整体加引号）")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--full", action="store_true", help="打印整段原文")
    ap.add_argument("--cite", action="store_true",
                    help="引用式回答：给 top-1 段落原文 + 出处 + 相关度")
    ap.add_argument("--interactive", action="store_true", help="连续提问模式")
    args = ap.parse_args()

    docs = load_corpus(args.corpus)
    paras = {d: paragraphs(t) for d, t in docs.items()}
    chunks, index = build_index(docs, args.size, args.overlap)
    print(f"语料：{len(docs)} 篇 / {sum(len(v) for v in paras.values())} 段  "
          f"切分：size={args.size} overlap={args.overlap} → {len(chunks)} chunks")

    if args.interactive:
        while True:
            try:
                q = input("\n问题（回车退出）> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"q", "quit", "exit"}:
                break
            if args.cite:
                cite(index, chunks, docs, paras, q, args.k)
            else:
                ask(index, chunks, docs, paras, q, args.k, args.full)
        return

    query = " ".join(args.query).strip()
    if not query:
        ap.error("请给出问题，或加 --interactive 进入连续提问模式")
    if args.cite:
        cite(index, chunks, docs, paras, query, args.k)
    else:
        ask(index, chunks, docs, paras, query, args.k, args.full)


if __name__ == "__main__":
    main()
