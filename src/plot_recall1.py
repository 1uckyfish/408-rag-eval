"""recall@1 与 recall@5 的取舍（拐点图）。

为什么要单独画 recall@1：
    recall@5 随 size 单调上升（chunk 越大候选越少，越容易覆盖 gold），
    但 recall@1 是先升后降的——size 到 1024 时一个 chunk 塞进两三个考点，
    "第一条结果就是对的"反而变难。这个拐点才是能讲的取舍。

数据来源：
    results/grid_bm25.csv 里只有 recall@5，没有 recall@1，
    所以这里现算一遍 r@1 / r@3 / r@5，顺带写成 results/grid_recall_k.csv 备查。

用法：
    py -3 src/plot_recall1.py
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import build_index, load_corpus, load_queries          # noqa: E402
from metrics import recall_at_k                                  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

SIZES = [128, 256, 512, 1024]
OVERLAPS = [0, 64]
CSV_OUT = "results/grid_recall_k.csv"
FIG_OUT = "results/fig4_recall1_vs_5.png"


def measure(docs, queries):
    """跑一遍 8 组配置，返回 {(size, overlap): {"chunks":n, "r1":.., "r3":.., "r5":..}}。"""
    out = {}
    for size in SIZES:
        for overlap in OVERLAPS:
            chunks, index = build_index(docs, size, overlap)
            acc = {1: 0.0, 3: 0.0, 5: 0.0}
            for q in queries:
                gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
                hits = index.search(q["query"], top_k=5)
                ranked = [chunks[i]["covered"] for i, _ in hits]
                for k in acc:
                    acc[k] += recall_at_k(ranked, gold, k)
            n = max(1, len(queries))
            out[(size, overlap)] = {
                "chunks": len(chunks),
                "r1": acc[1] / n, "r3": acc[3] / n, "r5": acc[5] / n,
            }
            print(f"size={size:<5} overlap={overlap:<3} chunks={len(chunks):<5} "
                  f"r@1={out[(size, overlap)]['r1']:.4f} "
                  f"r@3={out[(size, overlap)]['r3']:.4f} "
                  f"r@5={out[(size, overlap)]['r5']:.4f}")
    return out


def write_csv(res):
    os.makedirs(os.path.dirname(CSV_OUT) or ".", exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "overlap", "chunks", "recall@1", "recall@3", "recall@5"])
        for (size, overlap), v in res.items():
            w.writerow([size, overlap, v["chunks"],
                        f"{v['r1']:.4f}", f"{v['r3']:.4f}", f"{v['r5']:.4f}"])


def plot(res):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # 左：折线，横轴 size 取对数（128/256/512/1024 等距），实线 r@5、虚线 r@1
    for overlap in OVERLAPS:
        xs = SIZES
        r5 = [res[(s, overlap)]["r5"] for s in SIZES]
        r1 = [res[(s, overlap)]["r1"] for s in SIZES]
        ax1.plot(xs, r5, "o-", label=f"recall@5  overlap={overlap}")
        ax1.plot(xs, r1, "s--", alpha=0.75, label=f"recall@1  overlap={overlap}")
        # 标出 r@1 的峰
        best = max(range(len(r1)), key=lambda i: r1[i])
        ax1.annotate(f"峰值 {r1[best]:.3f}", (xs[best], r1[best]),
                     textcoords="offset points", xytext=(0, -16),
                     fontsize=8, ha="center", color="#333333")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(SIZES, [str(s) for s in SIZES])
    ax1.set_xlabel("chunk size（对数轴）")
    ax1.set_ylabel("recall")
    ax1.set_title("recall@5 单调上升，recall@1 有拐点")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    # 右：recall@1 的 4×2 热力图，看拐点落在哪一格
    mat = np.zeros((len(SIZES), len(OVERLAPS)))
    for i, size in enumerate(SIZES):
        for j, overlap in enumerate(OVERLAPS):
            mat[i, j] = res[(size, overlap)]["r1"]
    im = ax2.imshow(mat, cmap="magma", aspect="auto",
                    vmin=mat.min() - 0.02, vmax=mat.max() + 0.02)
    ax2.set_xticks(range(len(OVERLAPS)), [f"overlap={o}" for o in OVERLAPS])
    ax2.set_yticks(range(len(SIZES)), [f"size={s}" for s in SIZES])
    for i in range(len(SIZES)):
        for j in range(len(OVERLAPS)):
            v = mat[i, j]
            ax2.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=11,
                     color="white" if v < mat.mean() else "black")
    ax2.set_title("recall@1：512 最好，1024 反而掉")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="recall@1")

    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG_OUT) or ".", exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    plt.close(fig)
    print(f"已写出 {CSV_OUT} 与 {FIG_OUT}")


def main():
    docs = load_corpus("data/corpus")
    queries = load_queries("data/queries.jsonl")
    res = measure(docs, queries)
    write_csv(res)
    plot(res)


if __name__ == "__main__":
    main()
