"""打印一篇语料的所有段落，带序号和首句 —— 给标注时查段号用。

段落切分口径与 src/eval.py 完全一致（空行分隔，序号从 0 开始），
所以这里看到的序号就是 gold 该填的序号。

用法：
    py -3 src/show_paras.py os-09          # 只看某一篇
    py -3 src/show_paras.py                # 列出所有篇名和段数
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import load_corpus, paragraphs      # noqa: E402


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    docs = load_corpus("data/corpus")
    if len(sys.argv) < 2:
        print(f"共 {len(docs)} 篇语料：")
        for doc in sorted(docs):
            ps = paragraphs(docs[doc])
            print(f"  {doc}  {len(ps):>2} 段")
        print("\n用法：py -3 src\\show_paras.py os-09")
        return

    key = sys.argv[1].strip()
    key = key[:-3] if key.endswith(".md") else key
    if key not in docs:
        print(f"没有这篇：{key}（可选：{'、'.join(sorted(docs))}）")
        sys.exit(1)

    ps = paragraphs(docs[key])
    full = "--full" in sys.argv
    print(f"{key}：{len(ps)} 段")
    for i, (s, e) in enumerate(ps):
        body = docs[key][s:e].replace("\n", " ")
        if full:
            print(f"\n[{i}] {len(body)} 字")
            print(f"  {body}")
        else:
            print(f"  [{i}] {len(body):>3} 字  {body[:40]}…")
    if not full:
        print("\n上面每段只显示前 40 字；要看全文加 --full，例如："
              f"py -3 src\\show_paras.py {key} --full")


if __name__ == "__main__":
    main()
