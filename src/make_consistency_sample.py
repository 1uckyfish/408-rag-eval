"""标注一致性抽检 · 第 1 步：抽样 + 藏答案。

目的：常被问"gold 谁标的、可靠吗"。标准答法是"隔天重标 20 条，算一致率"。
这个脚本只做两件事：
    1. 从 120 条里按题型比例随机抽 20 条（固定随机种子，可复现）
    2. 生成一份**不含 gold** 的待标文件，和一份空的答案模板

为什么必须藏掉 gold：看见原答案就会照着抄，一致率必然虚高。
为什么必须隔天标：同一时段重标靠的是短期记忆，不算独立标注。

产出：
    data/raw/_relabel/round1.md          待标材料（query + 语料段落清单说明）
    data/raw/_relabel/round1.jsonl       答案模板，gold 为空，填完再跑 check_consistency.py

用法：
    py -3 src/make_consistency_sample.py            # 默认抽 20 条、种子 20261109
    py -3 src/make_consistency_sample.py --seed 42 --n 20
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

OUT_DIR = "data/raw/_relabel"
QUERIES = "data/queries.jsonl"

# 抽样比例按仓库里的题型分布走（概念 53 / 计算 44 / 综合 23），
# 这样抽出来的 20 条不会全是概念题，也不会漏掉综合题。
TYPES = ["概念", "计算", "综合"]


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def stratified_sample(queries, n, rng):
    """按题型比例分层抽样，剩余名额按余数补齐。"""
    pools = {t: [q for q in queries if q.get("type") == t] for t in TYPES}
    total = len(queries)
    picked = []
    for t in TYPES:                       # 先按比例分名额
        quota = round(n * len(pools[t]) / total)
        picked += rng.sample(pools[t], min(quota, len(pools[t])))
    # 名额可能因四舍五入少一两条，从"还没被抽中"的里补
    rest = [q for q in queries if q not in picked]
    rng.shuffle(rest)
    picked += rest[:max(0, n - len(picked))]
    return sorted(picked, key=lambda q: q["qid"])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=QUERIES)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    queries = load(args.queries)
    rng = random.Random(args.seed)
    sample = stratified_sample(queries, args.n, rng)

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 待标材料：只有 qid / 题型 / query，绝不出现 gold
    md = ["# 重标材料（第 1 轮）", "",
          f"共 {len(sample)} 条，随机种子 {args.seed}。",
          "**不要打开 data/queries.jsonl 或 docs/gold_review.md**，看见答案就白标了。",
          "",
          "填法：在 `round1.jsonl` 里给每条填 `gold`（段落序号，从 0 开始，1–3 段为佳）",
          "和 `why`（一句话理由）。段落序号按**空行分隔**，用下面命令查：",
          "",
          "```powershell",
          "py -3 src\\show_paras.py os-09        # 打印 os-09 的段落，带序号和首句",
          "```",
          "", "---", ""]
    for i, q in enumerate(sample, 1):
        md.append(f"## {i}. {q['qid']}　[{q.get('type', '')}／{q.get('year', '')}]")
        md.append("")
        md.append(f"> {q['query']}")
        md.append("")
    with open(os.path.join(args.outdir, "round1.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    # 2) 答案模板：gold 留空，qid 和 query 预填（防止填错题）
    with open(os.path.join(args.outdir, "round1.jsonl"), "w", encoding="utf-8") as f:
        for q in sample:
            row = {"qid": q["qid"], "query": q["query"], "gold": [], "why": ""}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dist = Counter(q.get("type", "?") for q in sample)
    print(f"已抽 {len(sample)} 条，题型分布："
          + "，".join(f"{t} {dist[t]}" for t in TYPES))
    print(f"  待标材料：{os.path.join(args.outdir, 'round1.md')}")
    print(f"  答案模板：{os.path.join(args.outdir, 'round1.jsonl')}")
    print("下一步：隔天（至少间隔 8 小时）再填 round1.jsonl，然后跑 "
          "py -3 src\\check_consistency.py")


if __name__ == "__main__":
    main()
