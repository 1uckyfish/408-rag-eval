"""校验 data/queries.jsonl（检索式 query 集）。

检查项：
  - 每行是一个 JSON 对象，字段 qid/query/gold/type/year 齐全
  - qid 非空且全局唯一
  - query 非空、长度合理，且没有残留考卷腔（"选项"、"____"、"（X 分）"、"A." 等）
  - type 只能是 概念 / 计算 / 综合
  - year 是 2009–2021 的整数
  - gold 是列表，元素形如 {"doc": "...", "para": 0}（本阶段允许空列表）
  - 没有两条 query 完全相同

用法：
    py -3 src/check_queries.py                 # 检查 data/queries.jsonl
    py -3 src/check_queries.py --min 120       # 不足 120 行则退出码 1
    py -3 src/check_queries.py --path data/raw/_qparts/x.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "data" / "queries.jsonl"

# Windows 控制台默认 GBK，打印 ✗/✓ 会抛 UnicodeEncodeError，统一转 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ("qid", "query", "gold", "type", "year")
TYPES = {"概念", "计算", "综合"}

# 题干里残留的考卷腔 / 选项痕迹
LEAK_PATTERNS = [
    (re.compile(r"选项\s*[A-DＡ-Ｄ]"), "选项字样"),
    (re.compile(r"(?<![A-Za-z])[A-D][\.．、]\s"), "A./B. 选项编号"),
    (re.compile(r"_{2,}|＿{2,}"), "下划线填空"),
    (re.compile(r"下列(选项|叙述|关于|关于其)"), "“下列…”前缀"),
    (re.compile(r"（\s*\d+\s*分\s*）"), "分值标记"),
    (re.compile(r"正确的是|错误的是|不正确的是"), "“正确的是”考卷句"),
]


def check_line(obj: dict, lineno: int, errors: list[str]) -> None:
    for field in REQUIRED:
        if field not in obj:
            errors.append(f"第 {lineno} 行缺少字段 {field}")
            return
    qid = obj["qid"]
    query = obj["query"]
    gold = obj["gold"]
    typ = obj["type"]
    year = obj["year"]

    if not isinstance(qid, str) or not qid.strip():
        errors.append(f"第 {lineno} 行 qid 为空或不是字符串")
    if not isinstance(query, str) or not query.strip():
        errors.append(f"第 {lineno} 行 query 为空")
    else:
        n = len(query.strip())
        if n < 8:
            errors.append(f"第 {lineno} 行 query 太短（{n} 字）：{query}")
        if n > 120:
            errors.append(f"第 {lineno} 行 query 太长（{n} 字）")
        for pat, label in LEAK_PATTERNS:
            if pat.search(query):
                errors.append(f"第 {lineno} 行 query 残留{label}：{query}")
    if typ not in TYPES:
        errors.append(f"第 {lineno} 行 type 非法：{typ!r}（只能 概念/计算/综合）")
    if not isinstance(year, int) or not (2009 <= year <= 2021):
        errors.append(f"第 {lineno} 行 year 非法：{year!r}")
    if not isinstance(gold, list):
        errors.append(f"第 {lineno} 行 gold 不是列表")
    else:
        for g in gold:
            if not (isinstance(g, dict) and isinstance(g.get("doc"), str) and isinstance(g.get("para"), int)):
                errors.append(f"第 {lineno} 行 gold 元素格式错：{g!r}")
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--min", type=int, default=0, help="最少行数，不足则退出码 1")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 1

    errors: list[str] = []
    rows: list[dict] = []
    qids: list[str] = []

    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            errors.append(f"第 {lineno} 行是空行（jsonl 不允许空行）")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"第 {lineno} 行 JSON 解析失败：{e}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"第 {lineno} 行不是 JSON 对象")
            continue
        check_line(obj, lineno, errors)
        rows.append(obj)
        if isinstance(obj.get("qid"), str):
            qids.append(obj["qid"])

    dup_qid = [q for q, c in Counter(qids).items() if c > 1]
    if dup_qid:
        errors.append(f"qid 重复：{dup_qid}")
    dup_query = [q for q, c in Counter(r.get("query") for r in rows).items() if c > 1]
    if dup_query:
        errors.append(f"query 文本重复 {len(dup_query)} 条，例如：{dup_query[0][:40]}…")

    n = len(rows)
    print(f"# {path}")
    print(f"共 {n} 条" + (f"（要求 ≥{args.min}）" if args.min else ""))
    if rows:
        print("\n## 年份分布")
        for year, c in sorted(Counter(r.get("year") for r in rows).items()):
            print(f"  {year}: {c}")
        print("\n## type 分布")
        for typ, c in Counter(r.get("type") for r in rows).most_common():
            print(f"  {typ}: {c}")
        lens = [len(r.get("query", "")) for r in rows if isinstance(r.get("query"), str)]
        if lens:
            print(f"\n## query 长度\n  最短 {min(lens)} 字 / 平均 {sum(lens) // len(lens)} 字 / 最长 {max(lens)} 字")
        n_gold = sum(1 for r in rows if r.get("gold"))
        print(f"\n## gold\n  已标注 {n_gold} 条，待标注 {n - n_gold} 条")

    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors[:40]:
            print(f"  ✗ {e}")
        if len(errors) > 40:
            print(f"  … 其余 {len(errors) - 40} 条略")
        return 1
    if args.min and n < args.min:
        print(f"\n✗ 只有 {n} 条，少于要求的 {args.min} 条")
        return 1
    print("\n结构检查通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
