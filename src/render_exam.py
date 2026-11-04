"""把真题 PDF 渲染成 PNG，供逐页读图抄录（扫描件没有文字层）。

用法：
    py -3 src/render_exam.py --pdf data/raw/pdf/2010.pdf            # 全部页
    py -3 src/render_exam.py --pdf data/raw/pdf/2010.pdf --pages 3-5
    py -3 src/render_exam.py --pdf data/raw/pdf/2010.pdf --dpi 160

输出：data/raw/pdf/<年份>_p<页码>.png
说明：这些 PDF 用的是内嵌子集字体且无 ToUnicode，pymupdf 直接 get_text() 得到的是乱码，
      不要走"先提取文字再清洗"的路，浪费时间；渲染成图后按页读更可靠。
"""

import argparse
import os
import sys

import pymupdf


def parse_pages(spec, total):
    if not spec:
        return list(range(1, total + 1))
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [p for p in out if 1 <= p <= total]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", default="")
    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    outdir = args.outdir or os.path.dirname(os.path.abspath(args.pdf))
    os.makedirs(outdir, exist_ok=True)
    pages = parse_pages(args.pages, doc.page_count)
    print(f"# {args.pdf}: 共 {doc.page_count} 页，渲染 {len(pages)} 页 @ {args.dpi}dpi")
    for p in pages:
        path = os.path.join(outdir, f"{stem}_p{p}.png")
        doc[p - 1].get_pixmap(dpi=args.dpi).save(path)
        print(f"  {path}")


if __name__ == "__main__":
    main()
