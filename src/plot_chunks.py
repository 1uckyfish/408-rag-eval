"""热力图进阶版：格子里同时标 recall 与 chunk 数。

和 src/plot.py 的差别只有三处（下面用 ← 标出）：
    1. 读 CSV 时多取一列 chunks
    2. 多摊一个矩阵 nums
    3. ax.text 里写成两行

用法：
    py -3 src/plot_chunks.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

SIZES = [128, 256, 512, 1024]
OVERLAPS = [0, 64]
CSV_PATH = "results/grid_bm25.csv"
OUT_PATH = "results/showcase/fig_heatmap_chunks.png"


def load_grid(path):
    """读 CSV，返回两个字典：(size, overlap) -> recall，以及 -> chunks。"""
    recalls, chunks = {}, {}                                # ← 两个字典
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (int(row["size"]), int(row["overlap"]))
            recalls[key] = float(row["recall"])
            chunks[key] = int(row["chunks"])                # ← 多读一列
    return recalls, chunks


def to_matrix(grid):
    """把字典摊成 4×2 矩阵；这里被调两次，一次给 recall、一次给 chunks。"""
    mat = np.zeros((len(SIZES), len(OVERLAPS)))
    for i, size in enumerate(SIZES):
        for j, overlap in enumerate(OVERLAPS):
            mat[i, j] = grid[(size, overlap)]
    return mat


def main():
    recalls, chunks = load_grid(CSV_PATH)
    mat = to_matrix(recalls)
    nums = to_matrix(chunks)                                # ← 第二个矩阵（浮点存整数，够用）

    fig, ax = plt.subplots(figsize=(6, 4.8))
    im = ax.imshow(mat, cmap="viridis", vmin=0.55, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(OVERLAPS)), [f"overlap={o}" for o in OVERLAPS])
    ax.set_yticks(range(len(SIZES)), [f"size={s}" for s in SIZES])
    ax.set_xlabel("overlap")
    ax.set_ylabel("chunk size")

    for i in range(len(SIZES)):
        for j in range(len(OVERLAPS)):
            v = mat[i, j]
            n = int(nums[i, j])
            # ← \n 换行：第一行 recall，第二行 chunk 数（括号包起来，和 recall 区分开）
            ax.text(j, i, f"{v:.3f}\n({n} chunks)", ha="center", va="center",
                    color="white" if v < 0.85 else "black", fontsize=10,
                    linespacing=1.4)

    ax.set_title("BM25 recall@5：size × overlap（括号内为 chunk 数）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall@5")
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"已写出 {OUT_PATH}")


if __name__ == "__main__":
    main()
