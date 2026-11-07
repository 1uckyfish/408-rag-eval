"""为 gold 标注生成候选段落：对每条 query，用章节级 BM25 直接检索「段落」。

为什么不复用 eval.py 的 chunk 检索：gold 标的是段落，chunk 会跨段，
按 chunk 命中反推段落会引入错位。这里直接以段落为单位建索引，
每题给出 top-k 段落（doc + para + 首句），标 gold 时只需判断「是/不是」。

用法：
    py -3 src/suggest_gold.py                     # 全部 120 条，每题 top-5
    py -3 src/suggest_gold.py --top 6 --out data/gold_review.md
    py -3 src/suggest_gold.py --qid q009 --full   # 只看得某题，并打印候选段全文
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bm25 import BM25, tokenize  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

CORPUS = ROOT / "data" / "corpus"
QUERIES = ROOT / "data" / "queries.jsonl"
DEFAULT_OUT = ROOT / "data" / "gold_review.md"


def load_paragraphs() -> list[dict]:
    """按文件名排序读入语料，para 序号从 0 开始，与 eval.py 的 paragraphs() 保持一致。"""
    items = []
    for path in sorted(CORPUS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        for i, p in enumerate(paras):
            items.append({"doc": path.stem, "para": i, "text": p})
    return items


def load_queries() -> list[dict]:
    rows = []
    for line in QUERIES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def first_sentence(text: str, limit: int = 60) -> str:
    m = re.split(r"[。；！？]", text, maxsplit=1)
    head = m[0] if m else text
    head = head.replace("\n", " ").strip()
    return head if len(head) <= limit else head[:limit] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--qid", default=None, help="只看某一条 query")
    ap.add_argument("--full", action="store_true", help="打印候选段全文")
    args = ap.parse_args()

    paras = load_paragraphs()
    queries = load_queries()
    if args.qid:
        queries = [q for q in queries if q["qid"] == args.qid]
        if not queries:
            print(f"没有 qid={args.qid}", file=sys.stderr)
            return 1

    index = BM25([tokenize(p["text"]) for p in paras])

    lines = ["# gold 候选段落（BM25 段落级检索）", ""]
    lines.append(f"语料 {len({p['doc'] for p in paras})} 篇 / {len(paras)} 段；查询 {len(queries)} 条；每题给 top-{args.top} 候选。")
    lines.append("")
    lines.append("标注方法：在 `data/queries.jsonl` 里把命中段落的 `{\"doc\":\"os-xx\",\"para\":n}` 填进该条 `gold`，")
    lines.append("只保留「给这一段就够答题」的 1–3 段；候选里没有能答的段落，就不要硬标，记下来补语料。")
    lines.append("")

    for q in queries:
        hits = index.search(q["query"], top_k=args.top)
        lines.append(f"## {q['qid']}（{q['type']}，{q['year']}，源 {q.get('source','')}）")
        lines.append("")
        lines.append(f"- query：{q['query']}")
        lines.append(f"- 当前 gold：{q.get('gold') or '空'}")
        lines.append("")
        if not hits:
            lines.append("- 候选：无（BM25 一个词都没命中，说明语料缺这块内容）")
            lines.append("")
            continue
        for rank, (idx, score) in enumerate(hits, start=1):
            p = paras[idx]
            if args.full:
                body = p["text"]
            else:
                body = first_sentence(p["text"])
            lines.append(f"- 候选{rank}：{p['doc']} / para {p['para']}（score {score:.2f}）　{body}")
        lines.append("")

    out = Path(args.out)
    if args.qid:
        # 单条查看模式不落盘，避免把全量候选文件覆盖成一条
        print("\n".join(lines))
        return 0
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {out}（{len(queries)} 条 query，每题 top-{args.top}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
