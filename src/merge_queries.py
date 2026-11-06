"""把 data/raw/_qparts/*.jsonl 合并成 data/queries.jsonl，并按年份/题号排序、重编 qid。

分片文件由改写阶段产出，qid 用 "YYYY-NN" 或 "YYYY-NN-b"；合并时统一重编成 q001…qNNN，
原题号保留在 source 字段里，方便回查 exam.md。

用法：
    py -3 src/merge_queries.py            # 合并并重写 data/queries.jsonl
    py -3 src/merge_queries.py --dry-run  # 只报告，不写文件
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "queries.jsonl"
PARTS = ROOT / "data" / "raw" / "_qparts"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

QID_RE = re.compile(r"^(\d{4})-(\d+)(?:-([a-z]))?$")


def sort_key(obj: dict) -> tuple:
    """按 (年份, 题号, 拆问后缀) 排序。"""
    qid = obj.get("qid") or ""
    m = QID_RE.match(qid)
    if m:
        return (int(m.group(1)), int(m.group(2)), m.group(3) or "")
    year = obj.get("year") if isinstance(obj.get("year"), int) else 9999
    return (year, 9999, qid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    part_files = sorted(PARTS.glob("*.jsonl")) if PARTS.is_dir() else []
    if not part_files:
        print(f"没有找到分片文件：{PARTS}/*.jsonl", file=sys.stderr)
        return 1

    rows: list[dict] = []
    seen: dict[str, str] = {}
    for path in part_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"✗ {path.name}:{lineno} JSON 解析失败：{e}", file=sys.stderr)
                return 1
            qid = obj.get("qid", "")
            if qid in seen:
                print(f"✗ qid 撞车 {qid}：{seen[qid]} 与 {path.name}:{lineno}", file=sys.stderr)
                return 1
            seen[qid] = f"{path.name}:{lineno}"
            obj.setdefault("source", qid)
            rows.append(obj)

    rows.sort(key=sort_key)

    out_lines: list[str] = []
    for i, obj in enumerate(rows, start=1):
        record = {
            "qid": f"q{i:03d}",
            "query": obj["query"],
            "gold": obj.get("gold", []),
            "type": obj["type"],
            "year": obj["year"],
            "source": obj.get("source", ""),
        }
        out_lines.append(json.dumps(record, ensure_ascii=False))

    years = sorted({r["year"] for r in rows})
    print(f"合并 {len(part_files)} 个分片 → {len(rows)} 条，年份 {years[0]}–{years[-1]}")
    from collections import Counter

    print("年份分布: " + ", ".join(f"{y}:{c}" for y, c in sorted(Counter(r["year"] for r in rows).items())))
    print("type 分布: " + ", ".join(f"{t}:{c}" for t, c in Counter(r["type"] for r in rows).most_common()))

    if args.dry_run:
        print("--dry-run：未写入")
        return 0

    OUT.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"已写入 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
