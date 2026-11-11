"""标注一致性抽检 · 第 2 步：对比两轮标注，算一致率。

对 20 条重标结果与已有 gold 逐一比较，给出三个层次的指标：
    完全一致   —— 两轮的段落集合完全相同（最严）
    有重叠     —— 至少有一段相同（宽松）
    Jaccard    —— 交集 / 并集，反映"差多少"

同时打印每条不一致的题，标出多标了哪段、漏标了哪段，方便你回看原文判断
"是我记错了"还是"原标注确实有问题"。

用法：
    py -3 src\\check_consistency.py                          # 默认读 round1.jsonl
    py -3 src\\check_consistency.py --relabel data/raw/_relabel/round1.jsonl
    py -3 src\\check_consistency.py --min 0.85                 # 低于阈值退出码 1
"""

import argparse
import json
import sys


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path} 第 {i} 行不是合法 JSON：{e}")
    return rows


def as_set(gold):
    return {(g["doc"], int(g["para"])) for g in gold if g.get("doc")}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--relabel", default="data/raw/_relabel/round1.jsonl")
    ap.add_argument("--min", type=float, default=0.0,
                    help="一致率低于这个值就以退出码 1 结束（默认不检查）")
    args = ap.parse_args()

    origin = {r["qid"]: r for r in load_jsonl(args.queries)}
    again = load_jsonl(args.relabel)

    # 填完的标志是 why 非空：gold 可以是空数组（表示"语料缺这块、标不出来"，
    # 这本身就是一个有效结论），但必须写理由，否则和"忘了填"分不开。
    empty = [r["qid"] for r in again if not r.get("why")]
    if empty:
        sys.exit(f"还有 {len(empty)} 条没写 why（没标完）：{', '.join(empty)}")

    exact = overlap = 0
    jaccard_sum = 0.0
    diffs = []
    for r in again:
        qid = r["qid"]
        if qid not in origin:
            sys.exit(f"{qid} 不在 queries.jsonl 里")
        a = as_set(origin[qid].get("gold", []))
        b = as_set(r.get("gold", []))
        if a == b:
            exact += 1
        if a & b:
            overlap += 1
        jaccard_sum += len(a & b) / len(a | b) if (a | b) else 1.0
        if a != b:
            diffs.append((qid, a - b, b - a, r.get("why", "")))

    n = len(again)
    print(f"对比 {n} 条（原始 gold vs 重标）")
    print(f"  完全一致：{exact}/{n} = {exact / n:.1%}")
    print(f"  有重叠：  {overlap}/{n} = {overlap / n:.1%}")
    print(f"  平均 Jaccard：{jaccard_sum / n:.3f}")

    if diffs:
        print(f"\n不一致 {len(diffs)} 条：")
        for qid, only_a, only_b, why in diffs:
            print(f"  {qid}")
            if only_a:
                print(f"    原标注有、重标没标：{', '.join(f'{d}/{p}' for d, p in sorted(only_a))}")
            if only_b:
                print(f"    重标有、原标注没标：{', '.join(f'{d}/{p}' for d, p in sorted(only_b))}")
            if why:
                print(f"    重标理由：{why}")
    else:
        print("\n全部一致 —— 注意：如果两次是同一天标的，这个数字会虚高。")

    rate = exact / n
    if rate < args.min:
        print(f"\n一致率 {rate:.1%} 低于阈值 {args.min:.0%}")
        sys.exit(1)


if __name__ == "__main__":
    main()
