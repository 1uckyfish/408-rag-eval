"""图 5：最优 vs 默认的逐题差值分布。

要回答的问题：512/0（默认）换到 1024/64（网格最优）涨的 8.9 个点，
是**大多数题都涨一点**，还是**少数题暴涨撑起来的**？
如果是后者，这个"最优配置"就不稳，换一批题可能就没了。

做法：同一套语料和 query，两个配置各跑一遍，逐题算 recall@5 差值，画分布。

用法：
    py -3 src/plot_delta.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import build_index, load_corpus, load_queries    # noqa: E402
from metrics import recall_at_k                             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT = (512, 0)
BEST = (1024, 64)
FIG_OUT = "results/fig5_delta_distribution.png"


def per_query(docs, queries, size, overlap, k=5):
    chunks, index = build_index(docs, size, overlap)
    out = {}
    for q in queries:
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        hits = index.search(q["query"], top_k=k)
        ranked = [chunks[i]["covered"] for i, _ in hits]
        out[q["qid"]] = recall_at_k(ranked, gold, k)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    docs = load_corpus("data/corpus")
    queries = load_queries("data/queries.jsonl")
    base = per_query(docs, queries, *DEFAULT)
    best = per_query(docs, queries, *BEST)

    qids = [q["qid"] for q in queries]
    deltas = np.array([best[q] - base[q] for q in qids])

    up = int((deltas > 0).sum())
    down = int((deltas < 0).sum())
    same = int((deltas == 0).sum())
    gain = float(np.mean([best[q] for q in qids]) - np.mean([base[q] for q in qids]))

    print(f"默认 {DEFAULT[0]}/{DEFAULT[1]} (r@5={np.mean([base[q] for q in qids]):.4f}) "
          f"vs 最优 {BEST[0]}/{BEST[1]} (r@5={np.mean([best[q] for q in qids]):.4f})")
    print(f"差值 {gain:+.4f}；涨 {up} 条 / 跌 {down} 条 / 不变 {same} 条")
    losers = sorted(((best[q] - base[q], q) for q in qids))[:5]
    print("跌幅最大：" + "，".join(f"{q}({d:+.2f})" for d, q in losers))
    room = [q for q in qids if base[q] < 1]
    still = [q for q in room if best[q] < 1]
    print(f"默认配置已经满分的：{len(qids) - len(room)} 条；有提升空间的 {len(room)} 条，"
          f"换配置后仍不满分的 {len(still)} 条")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：逐题差值的直方图（只有 -1/-0.5/0/+0.5/+1 这几个取值，用柱状更清楚）
    vals, counts = np.unique(np.round(deltas, 3), return_counts=True)
    colors = ["#c0392b" if v < 0 else ("#7f8c8d" if v == 0 else "#27ae60") for v in vals]
    ax1.bar([f"{v:+.1f}" for v in vals], counts, color=colors)
    for x, c in zip(range(len(vals)), counts):
        ax1.text(x, c + 0.6, str(c), ha="center", fontsize=9)
    ax1.set_xlabel("单题 recall@5 差值（最优 − 默认）")
    ax1.set_ylabel("题数")
    ax1.set_title(f"涨 {up} 条 / 跌 {down} 条 / 不变 {same} 条")

    # 右：散点 —— 横轴默认配置的单题 recall，纵轴最优配置的单题 recall。
    # 对角线以上是变好、以下是变差；贴顶的一排是"两条都满分"，没有提升空间。
    xs = np.array([base[q] for q in qids])
    ys = np.array([best[q] for q in qids])
    ax2.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="y = x")
    ax2.scatter(xs[ys > xs], ys[ys > xs], s=45, c="#27ae60", label=f"变好 {up} 条", zorder=3)
    ax2.scatter(xs[ys < xs], ys[ys < xs], s=45, c="#c0392b", label=f"变差 {down} 条", zorder=3)
    flat = (ys == xs)
    ax2.scatter(xs[flat], ys[flat], s=18, c="#95a5a6", alpha=0.7, label=f"持平 {same} 条")
    room = int((xs < 1).sum())
    ax2.set_xlim(-0.05, 1.08)
    ax2.set_ylim(-0.05, 1.08)
    ax2.set_xlabel("默认 512/0 的单题 recall@5")
    ax2.set_ylabel("最优 1024/64 的单题 recall@5")
    ax2.set_title(f"有提升空间的 {room} 条里，{up} 条变好")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"图 5：{BEST[0]}/{BEST[1]} vs {DEFAULT[0]}/{DEFAULT[1]}，整体 "
                 f"{gain*100:+.1f}pt", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT) or ".", exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    print(f"已写出 {FIG_OUT}")


if __name__ == "__main__":
    main()
