"""把 data/raw/_parts/*.md 合并进 data/raw/exam.md，并按年份、题号排序。

exam.md 的每个题目块以 `### <年份>-<题号>` 开头，到下一个 `###` / `##` / 文件尾结束。
合并逻辑：
  1. 读 exam.md 里所有题目块（丢弃 `## 年份` 这类分隔行，重新生成）；
  2. 读 _parts 目录下所有 *.md 的题目块；
  3. 按 (年份, 题号) 去重排序后整体重写 exam.md。

用法:
    py -3 src/merge_parts.py            # 合并并排序
    py -3 src/merge_parts.py --dry-run  # 只报告会合并多少题，不写文件
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAM = ROOT / "data" / "raw" / "exam.md"
PARTS = ROOT / "data" / "raw" / "_parts"

HEAD_RE = re.compile(r"^###\s+(\d{4})-(\d+)\s*$")


def split_blocks(text: str) -> list[tuple[int, int, str]]:
    """返回 [(year, qno, block_text)]，block_text 含标题行，不含末尾空行。"""
    blocks: list[tuple[int, int, list[str]]] = []
    cur: list[str] | None = None
    cur_key: tuple[int, int] | None = None

    for line in text.splitlines():
        m = HEAD_RE.match(line.strip())
        if m:
            if cur is not None and cur_key is not None:
                blocks.append((*cur_key, cur))
            cur = [line.rstrip()]
            cur_key = (int(m.group(1)), int(m.group(2)))
            continue
        if line.startswith("## "):
            # 年份分隔行：结束当前块，本身丢弃
            if cur is not None and cur_key is not None:
                blocks.append((*cur_key, cur))
            cur, cur_key = None, None
            continue
        if cur is not None:
            cur.append(line.rstrip())

    if cur is not None and cur_key is not None:
        blocks.append((*cur_key, cur))

    out: list[tuple[int, int, str]] = []
    for year, qno, lines in blocks:
        while lines and not lines[-1].strip():
            lines.pop()
        if lines:
            out.append((year, qno, "\n".join(lines)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    merged: dict[tuple[int, int], str] = {}
    if EXAM.exists():
        for year, qno, block in split_blocks(EXAM.read_text(encoding="utf-8")):
            merged[(year, qno)] = block

    n_exam = len(merged)
    part_files = sorted(PARTS.glob("*.md")) if PARTS.is_dir() else []
    n_part = 0
    for path in part_files:
        for year, qno, block in split_blocks(path.read_text(encoding="utf-8")):
            key = (year, qno)
            if key in merged and merged[key] != block:
                print(f"  覆盖重复题 {year}-{qno}（来自 {path.name}）", file=sys.stderr)
            merged[key] = block
            n_part += 1

    if not merged:
        print("没有解析到任何题目块", file=sys.stderr)
        return 1

    lines: list[str] = []
    last_year = None
    for (year, qno) in sorted(merged):
        if year != last_year:
            if lines:
                lines.append("")
            lines.append(f"## {year}")
            lines.append("")
            last_year = year
        lines.append(merged[(year, qno)])
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    years = sorted({y for y, _ in merged})
    print(f"exam.md 原有 {n_exam} 题，_parts 提供 {n_part} 题，合并后 {len(merged)} 题，覆盖年份 {years[0]}–{years[-1]}")
    if args.dry_run:
        print("--dry-run：未写入")
        return 0
    EXAM.write_text(text, encoding="utf-8")
    print(f"已写入 {EXAM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
