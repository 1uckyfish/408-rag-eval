"""把 data/raw/_gparts/gold_*.jsonl 标注结果写回 data/queries.jsonl。

标注文件每行 {"qid","gold":[{"doc","para"}],"why"}；本脚本只取 qid 和 gold，
保持 queries.jsonl 原有的字段顺序（qid/query/gold/type/year/source）与顺序不变。

安全检查：
  - 标注的 qid 必须在 queries.jsonl 里存在；反过来缺失的 qid 会列表报出
  - doc 必须是 data/corpus 下真实存在的文件
  - para 必须小于该文件的段落数（按空行切分，与 eval.py 一致）

用法：
    py -3 src/apply_gold.py --dry-run
    py -3 src/apply_gold.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data" / "queries.jsonl"
GOLD_PARTS = ROOT / "data" / "raw" / "_gparts"
CORPUS = ROOT / "data" / "corpus"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def paragraphs(text: str) -> list[str]:
    """与 src/eval.py 保持一致的切段方式：空行分隔，去空白。"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gold_map: dict[str, list[dict]] = {}
    why_map: dict[str, str] = {}
    for path in sorted(GOLD_PARTS.glob("gold_*.jsonl")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = obj["qid"]
            gold = obj.get("gold", [])
            if qid in gold_map:
                # 补标阶段允许同一 qid 出现多次：保留非空的那份（空 gold → 有 gold 是补标）
                if gold_map[qid] and not gold:
                    continue
                if gold and not gold_map[qid]:
                    gold_map[qid] = gold
                    why_map[qid] = obj.get("why", "")
                    continue
                print(f"✗ qid 重复标注：{qid}（{path.name}:{lineno}）", file=sys.stderr)
                return 1
            gold_map[qid] = gold
            why_map[qid] = obj.get("why", "")

    rows = [json.loads(l) for l in QUERIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_paras = {p.stem: len(paragraphs(p.read_text(encoding="utf-8"))) for p in sorted(CORPUS.glob("os-*.md"))}

    errors: list[str] = []
    missing = [r["qid"] for r in rows if r["qid"] not in gold_map]
    unknown = [q for q in gold_map if q not in {r["qid"] for r in rows}]
    if unknown:
        errors.append(f"标注里有 queries.jsonl 不存在的 qid：{unknown[:10]}")
    for qid, gold in gold_map.items():
        for g in gold:
            doc, para = g.get("doc"), g.get("para")
            if doc not in n_paras:
                errors.append(f"{qid}: 语料 {doc} 不存在")
            elif not isinstance(para, int) or not (0 <= para < n_paras[doc]):
                errors.append(f"{qid}: {doc} 的 para {para} 越界（该文件共 {n_paras[doc]} 段）")

    out = []
    for r in rows:
        rec = {
            "qid": r["qid"],
            "query": r["query"],
            "gold": gold_map.get(r["qid"], r["gold"]),
            "type": r["type"],
            "year": r["year"],
            "source": r.get("source", ""),
        }
        out.append(rec)

    n_gold = sum(1 for r in out if r["gold"])
    dist = Counter(len(r["gold"]) for r in out)
    print(f"# {QUERIES}")
    print(f"共 {len(out)} 条；已标 gold {n_gold} 条；缺标注 {len(missing)} 条")
    print("gold 段数分布：" + ", ".join(f"{k} 段 {v} 条" for k, v in sorted(dist.items())))
    if missing:
        print(f"缺标注 qid（前 20）：{missing[:20]}")
    empty = [r["qid"] for r in out if not r["gold"]]
    if empty:
        print(f"gold 为空的 qid：{empty}")

    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors[:40]:
            print(f"  ✗ {e}")
        return 1

    if args.dry_run:
        print("\n--dry-run：未写入")
        return 0

    QUERIES.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8")
    print(f"\n已写回 {QUERIES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
