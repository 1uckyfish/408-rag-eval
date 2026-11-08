"""校验 data/corpus/*.md 语料文件。

检查项：
  - 每篇字数（不含空白）在 800–3000 之间
  - 段落数 ≥ 5，每段 100–400 字（太短的段落做 gold 会太碎，太长会跨 chunk）
  - 段落由空行分隔，段内不换行
  - 没有 Markdown 痕迹（# 标题、列表符号、表格、代码块、加粗）
  - 没有教材/考卷腔的坏句子（"本节"、"如下图所示"、"考纲要求"、"选项"）
  - 同一段落不在两篇里重复出现
  - 文件名形如 os-01.md / os-03-1.md

用法：
    py -3 src/check_corpus.py
    py -3 src/check_corpus.py --min-files 15
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

NAME_RE = re.compile(r"^os-\d{2}(-\d)?\.md$")
MD_PATTERNS = [
    (re.compile(r"^\s*#{1,6}\s", re.M), "Markdown 标题"),
    (re.compile(r"^\s*[-*+]\s", re.M), "列表符号"),
    (re.compile(r"^\s*\|.*\|\s*$", re.M), "表格"),
    (re.compile(r"```"), "代码块"),
    (re.compile(r"\*\*.+?\*\*"), "加粗"),
]
BAD_PHRASES = ["本节", "如下图所示", "如上图", "考纲要求", "选项", "（　）", "____"]
MIN_CHARS, MAX_CHARS = 800, 3000
MIN_PARA, MAX_PARA_LEN = 100, 400
NGRAM = 13
NEAR_DUP = 0.5


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    s = re.sub(r"\s", "", text)
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_duplicate_pairs(paras: list[tuple[str, int, str]], threshold: float) -> list[tuple[float, str, int, str, int]]:
    """字符 n-gram Jaccard 近似重复检测；paras = [(doc, idx, text)]。"""
    grams = [ngrams(t) for _, _, t in paras]
    out = []
    for i in range(len(paras)):
        for j in range(i + 1, len(paras)):
            sim = jaccard(grams[i], grams[j])
            if sim >= threshold:
                out.append((sim, paras[i][0], paras[i][1], paras[j][0], paras[j][1]))
    out.sort(reverse=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-files", type=int, default=1)
    ap.add_argument("--near-dup", type=float, default=NEAR_DUP,
                    help="近似重复阈值（字符 13-gram Jaccard），0 表示关闭")
    args = ap.parse_args()

    files = sorted(CORPUS.glob("*.md"))
    errors: list[str] = []
    rows = []
    para_index: dict[str, list[str]] = {}
    all_paras: list[tuple[str, int, str]] = []

    if len(files) < args.min_files:
        errors.append(f"语料只有 {len(files)} 篇，少于要求的 {args.min_files} 篇")

    for path in files:
        if not NAME_RE.match(path.name):
            errors.append(f"{path.name}：文件名不符合 os-NN.md 约定")
        text = path.read_text(encoding="utf-8")
        if "\r" in text:
            errors.append(f"{path.name}：含 CRLF，应为 LF")
        for pat, label in MD_PATTERNS:
            if pat.search(text):
                errors.append(f"{path.name}：出现{label}")
        for phrase in BAD_PHRASES:
            if phrase in text:
                errors.append(f"{path.name}：出现不该有的字样「{phrase}」")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chars = len(re.sub(r"\s", "", text))
        if not (MIN_CHARS <= chars <= MAX_CHARS):
            errors.append(f"{path.name}：{chars} 字，超出 {MIN_CHARS}–{MAX_CHARS}")
        if len(paras) < 5:
            errors.append(f"{path.name}：只有 {len(paras)} 段，少于 5 段")
        for i, p in enumerate(paras):
            if "\n" in p:
                errors.append(f"{path.name} 第 {i} 段：段内换行")
            n = len(re.sub(r"\s", "", p))
            if not (MIN_PARA <= n <= MAX_PARA_LEN):
                errors.append(f"{path.name} 第 {i} 段：{n} 字，超出 {MIN_PARA}–{MAX_PARA_LEN}")
            key = re.sub(r"\s", "", p)[:60]
            if key in para_index and para_index[key][0] != path.name:
                errors.append(
                    f"段落重复：{path.name} 第 {i} 段与 {para_index[key][0]} 第 {para_index[key][1]} 段开头相同"
                )
            para_index.setdefault(key, (path.name, i))
            all_paras.append((path.stem, i, p))
        lengths = [len(re.sub(r"\s", "", p)) for p in paras]
        rows.append((path.name, chars, len(paras), min(lengths), max(lengths)))

    print(f"# {CORPUS}")
    print(f"{'文件':<14}{'字数':>6}{'段数':>6}{'最短段':>8}{'最长段':>8}")
    for name, chars, n, lo, hi in rows:
        print(f"{name:<14}{chars:>6}{n:>6}{lo:>8}{hi:>8}")
    total_chars = sum(r[1] for r in rows)
    total_paras = sum(r[2] for r in rows)
    print(f"\n合计 {len(rows)} 篇、{total_chars} 字、{total_paras} 段" + (f"（平均每篇 {total_chars // max(len(rows),1)} 字）" if rows else ""))

    if args.near_dup > 0:
        near = near_duplicate_pairs(all_paras, args.near_dup)
        print(f"\n近似重复检测（{NGRAM}-gram Jaccard ≥ {args.near_dup}）：{len(near)} 对")
        for sim, d1, i1, d2, i2 in near[:8]:
            print(f"  ! {sim:.2f}  {d1} p{i1}  ≈  {d2} p{i2}")
            errors.append(f"近似重复 {sim:.2f}：{d1} p{i1} 与 {d2} p{i2}（改写同一段会污染检索）")
        if len(near) > 8:
            print(f"  … 其余 {len(near) - 8} 对略")

    if errors:
        print(f"\n发现 {len(errors)} 个问题：")
        for e in errors[:40]:
            print(f"  ✗ {e}")
        if len(errors) > 40:
            print(f"  … 其余 {len(errors) - 40} 条略")
        return 1
    print("\n结构检查通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
