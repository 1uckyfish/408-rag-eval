"""
用法：
    py -3 src/plot.py
"""

import csv                 # 读 CSV 用
import os                  # 拼路径、建目录

import matplotlib
matplotlib.use("Agg")      # 不弹窗口，直接存图片（服务器/终端里也能跑）
import matplotlib.pyplot as plt
import numpy as np         # 把 CSV 摊成矩阵

# 这两行让图里的中文正常显示，不加会变成一个个方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

SIZES = [128, 256, 512, 1024]
OVERLAPS = [0, 64]
CSV_PATH = "results/grid_bm25.csv"
OUT_PATH = "results/fig_heatmap_hand.png"


def load_grid(path):
    """读 CSV，返回 {(size, overlap): recall}。"""
    grid = {}
    with open(path, encoding="utf-8-sig") as f:   # utf-8-sig 去掉 BOM，否则第一列名会带 \ufeff
        for row in csv.DictReader(f):
            size = int(row["size"])
            overlap = int(row["overlap"])
            grid[(size, overlap)] = float(row["recall"])
    return grid


def to_matrix(grid):
    """把字典摊成 4×2 的矩阵：行是 size，列是 overlap。"""
    mat = np.zeros((len(SIZES), len(OVERLAPS)))
    for i, size in enumerate(SIZES):         # i=0..3 对应四行
        for j, overlap in enumerate(OVERLAPS):  # j=0..1 对应两列
            mat[i, j] = grid[(size, overlap)]
    return mat


def main():
    grid = load_grid(CSV_PATH)
    mat = to_matrix(grid)

    fig, ax = plt.subplots(figsize=(6, 4.5))          # 画布 6×4.5 英寸
    im = ax.imshow(mat, cmap="viridis", vmin=0.55, vmax=1.0, aspect="auto")
    # imshow 把矩阵当图片画：数值映射成颜色，vmin/vmax 决定色标范围，
    # aspect="auto" 让格子随画布拉伸，而不是强制正方形

    ax.set_xticks(range(len(OVERLAPS)), [f"overlap={o}" for o in OVERLAPS])
    ax.set_yticks(range(len(SIZES)), [f"size={s}" for s in SIZES])
    ax.set_xlabel("overlap")
    ax.set_ylabel("chunk size")

    for i in range(len(SIZES)):            # 逐格写数字，让别人不用猜颜色
        for j in range(len(OVERLAPS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if v < 0.85 else "black", fontsize=11)

    ax.set_title("BM25 recall@5：size × overlap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall@5")
    fig.tight_layout()                      # 自动收紧边距，防止标签被裁掉

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)                          # 关掉画布，否则连续出图会吃内存
    print(f"已写出 {OUT_PATH}")


if __name__ == "__main__":
    main()
