"""切分：段落识别 + 定长滑窗（零依赖）。

设计要点
--------
1. gold 标注在"段落"上（不是 chunk），所以这里只返回字符区间，不做语义判断；
2. 窗口边界优先落在句末标点/换行上，避免把一句话从中间劈开；
3. overlap 从上一个窗口尾部取固定字符数。

用法
----
    from chunker import paragraphs, chunk
    paras = paragraphs(text)          # [(start, end), ...]
    spans = chunk(text, 300, 60)      # [(start, end), ...]
"""

import re

SENT_END = "。！？；!?;\n"


def paragraphs(text):
    """按空行切段，返回每段的 [start, end) 字符区间。"""
    spans = []
    start = 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        end = m.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = m.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _snap(text, pos, lo, hi):
    """把切点 pos 挪到 [lo, hi) 内最近的一个句末标点之后。"""
    best = None
    for p in range(lo, min(hi, len(text))):
        if text[p] in SENT_END:
            if best is None or abs(p + 1 - pos) < abs(best - pos):
                best = p + 1
    return best if best is not None else pos


def chunk(text, size, overlap=0):
    """定长滑窗切分，返回 [(start, end), ...]。size/overlap 单位是字符。"""
    if size <= 0:
        raise ValueError("size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能为负")
    if overlap > size // 2:
        raise ValueError(
            "overlap 不能超过 size 的一半，否则窗口几乎不前进（chunk 数量爆炸）"
        )

    spans = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            lo = max(start + size // 2, start)
            hi = min(n, end + size // 4 + 1)
            end = _snap(text, end, lo, hi)
        if end <= start:
            end = min(start + size, n)
        if text[start:end].strip():
            spans.append((start, end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return spans
