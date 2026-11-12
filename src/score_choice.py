"""人工三选一复核的判分脚本。

用法：
    py -3 src/score_choice.py ABCABCABCA

参数是一串字母，顺序与 data/raw/_relabel/choice.md 里的题号一致（1..10）。
答案键在 data/raw/_relabel/choice_key.json，由生成卷子的脚本写入。

输出：
    - 逐题对错
    - 命中数 / 总题数，以及随机基线（三选一 = 33.3%）

注意：这是"人工三选一复核"，不是双标一致率。它证明的是
"给定三段候选，人能不能挑出可回答的那段"，属于抽样复核，
写进文档时必须如实这样称呼。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
KEY = ROOT / "data" / "raw" / "_relabel" / "choice_key.json"


def main():
    if len(sys.argv) != 2:
        print("用法：py -3 src/score_choice.py ABCABCABCA（10 个字母，顺序同 choice.md）")
        return 1
    ans = sys.argv[1].strip().upper().replace(" ", "").replace(",", "")
    key = json.loads(KEY.read_text(encoding="utf-8"))
    qids = list(key.keys())
    if len(ans) != len(qids):
        print(f"答案长度 {len(ans)} 与题数 {len(qids)} 不符")
        return 1
    if any(c not in "ABC" for c in ans):
        print("只能包含 A/B/C")
        return 1

    hit, rows = 0, []
    for qid, a in zip(qids, ans):
        ok = a == key[qid]
        hit += ok
        rows.append((qid, a, key[qid], "✓" if ok else "✗"))
    for qid, a, k, mark in rows:
        print(f"  {qid}: 你选 {a}，正确 {k}  {mark}")
    n = len(qids)
    print(f"\n命中 {hit}/{n} = {hit / n:.0%}（三选一随机基线 33%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
