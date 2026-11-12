"""出图：热力图、recall 与 chunk 数的关系、按题型拆分、recall@k 曲线。

数据从当前语料 + queries.jsonl 现算（不依赖旧 CSV），保证图与代码一致。

用法：
    py -3 src/plot_figures.py
    py -3 src/plot_figures.py --outdir results
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402
import numpy as np                                   # noqa: E402

from eval import load_corpus, load_queries, build_index   # noqa: E402
from metrics import recall_at_k, mrr_at_k                  # noqa: E402

SIZES = [128, 256, 512, 1024]
OVERLAPS = [0, 64]
TYPES = ["概念", "计算", "综合"]

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def setup_cjk_font() -> str:
    """挑一个能显示中文的字体，返回字体名。"""
    candidates = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                  "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simsun.ttc"]
    for path in candidates:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return ""


def measure(docs, queries, sizes=SIZES, overlaps=OVERLAPS, kmax=5):
    """返回 {(size, overlap): {"chunks": n, "recall@k": {k: v}, "rows": [...]}}。"""
    out = {}
    for size in sizes:
        for ov in overlaps:
            chunks, index = build_index(docs, size, ov)
            rows = []
            for q in queries:
                gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
                hits = index.search(q["query"], top_k=kmax)
                ranked = [chunks[i]["covered"] for i, _ in hits]
                rows.append({
                    "qid": q["qid"],
                    "type": q.get("type", ""),
                    "ranked": ranked,
                    "gold": gold,
                })
            rk = {k: np.mean([recall_at_k(r["ranked"], r["gold"], k) for r in rows])
                  for k in range(1, kmax + 1)}
            out[(size, ov)] = {
                "chunks": len(chunks),
                "recall@k": rk,
                "rows": rows,
                "mrr": float(np.mean([mrr_at_k(r["ranked"], r["gold"], kmax) for r in rows])),
            }
            print(f"size={size:<5} overlap={ov:<3} chunks={len(chunks):<4} "
                  f"recall@5={rk[5]:.4f} recall@1={rk[1]:.4f}")
    return out


def plot_heatmap(res, outdir):
    grid = np.array([[res[(s, o)]["recall@k"][5] for o in OVERLAPS] for s in SIZES])
    fig = plt.figure(figsize=(11, 4.6))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(grid, cmap="viridis", vmin=0.6, vmax=1.0, aspect="auto")
    ax1.set_xticks(range(len(OVERLAPS)), [f"overlap={o}" for o in OVERLAPS])
    ax1.set_yticks(range(len(SIZES)), [f"size={s}" for s in SIZES])
    for i in range(len(SIZES)):
        for j in range(len(OVERLAPS)):
            v = grid[i, j]
            ax1.text(j, i, f"{v:.3f}\n({res[(SIZES[i], OVERLAPS[j])]['chunks']} chunks)",
                     ha="center", va="center", color="white" if v < 0.92 else "black", fontsize=9)
    ax1.set_title("recall@5：size × overlap（格内为 chunk 数）")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(1, 2, 2)
    for ov in OVERLAPS:
        xs = [res[(s, ov)]["chunks"] for s in SIZES]
        ys = [res[(s, ov)]["recall@k"][5] for s in SIZES]
        ax2.plot(xs, ys, "o-", label=f"overlap={ov}（点自左向右 size=128/256/512/1024）")
    ax2.axhline(0.8, ls="--", c="gray", lw=1)
    ax2.text(ax2.get_xlim()[0], 0.805, "G1 上沿 0.8", fontsize=8, color="gray", ha="left")
    ax2.set_xlabel("chunk 总数（越少越容易命中）")
    ax2.set_ylabel("recall@5")
    ax2.set_title("recall@5 随 chunk 数上升 = 天花板效应")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出 {path}")


def plot_type_recall(res, outdir, size=512, ov=0):
    rows = res[(size, ov)]["rows"]
    vals = [np.mean([r["recall"] if "recall" in r else recall_at_k(r["ranked"], r["gold"], 5)
                     for r in rows if r["type"] == t]) for t in TYPES]
    ns = [sum(1 for r in rows if r["type"] == t) for t in TYPES]
    overall = np.mean([recall_at_k(r["ranked"], r["gold"], 5) for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = [f"{t}\n(n={n})" for t, n in zip(TYPES, ns)]
    bars = ax.bar(labels, vals, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.axhline(overall, ls="--", c="gray", lw=1, label=f"总体 {overall:.3f}")
    ax.legend(loc="lower right", fontsize=9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall@5")
    ax.set_title(f"分题型 recall@5（size={size}, overlap={ov}）")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig0_type_recall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出 {path}")


def plot_recall_vs_k(res, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for size in [256, 512, 1024]:
        ks = list(range(1, 6))
        ys = [res[(size, 0)]["recall@k"][k] for k in ks]
        ax.plot(ks, ys, "o-", label=f"size={size}（{res[(size, 0)]['chunks']} chunks）")
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_title("同一配置下 recall 随 k 的变化（k 越小越有区分度）")
    ax.set_xticks(range(1, 6))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_recall_vs_k.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出 {path}")


def plot_pareto(outdir):
    """recall@5 vs P95 延迟（回答 Q3：多花的延迟值不值）。读 CSV，不现算。"""
    import csv as _csv

    bm, dn = [], []
    grid = os.path.join(outdir, "grid_bm25.csv")
    if os.path.exists(grid):
        with open(grid, encoding="utf-8-sig") as f:
            for r in _csv.DictReader(f):
                bm.append((float(r["p95_ms"]), float(r["recall"]),
                           f"bm25 {r['size']}/{r['overlap']}"))

    for fn, tag in [("dense_256_0.csv", "dense 256/0"), ("dense_512_0.csv", "dense 512/0")]:
        p = os.path.join(outdir, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
        rec = np.mean([float(r["recall"]) for r in rows])
        lat = sorted(float(r["latency_ms"]) for r in rows)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        dn.append((p95, rec, tag))

    if not bm and not dn:
        print("跳过图 3：缺 results/grid_bm25.csv 或 dense CSV")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    seen = {}
    if bm:
        ax.scatter([p for p, _, _ in bm], [r for _, r, _ in bm], s=70, c="#4c72b0",
                   label="BM25（8 组配置）", zorder=3)
        # 手工错开：同一 size 的两档 y 差很小，靠上下偏移分开
        dy_map = {"128/0": -16, "128/64": 9, "256/0": 9, "256/64": -16,
                  "512/0": 9, "512/64": -16}
        for idx, (p, r, lab) in enumerate(bm):
            if lab.endswith("1024/64"):     # 与 1024/0 几乎重合，合并标注
                continue
            shown = lab + ",64" if lab.endswith("1024/0") else lab
            dy = dy_map.get(shown, 9 if idx % 2 == 0 else -16)
            ax.annotate(shown, (p, r), textcoords="offset points", xytext=(7, dy),
                        fontsize=8, color="#4c72b0")
    if dn:
        ax.scatter([p for p, _, _ in dn], [r for _, r, _ in dn], s=110, marker="s", c="#c44e52",
                   label="dense（bge-small-zh-v1.5）", zorder=4)
        for p, r, lab in dn:
            ax.annotate(lab, (p, r), textcoords="offset points", xytext=(-8, 6),
                        fontsize=8, color="#c44e52", ha="right")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xlim(right=max([p for p, _, _ in bm + dn]) * 1.55)  # 右侧留白，免得 dense 标注被裁掉
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("P95 延迟 / ms（对数轴，单机 CPU 测量）")
    ax.set_ylabel("recall@5")
    ax.set_title("recall@5 vs P95：BM25 在更低的延迟上拿到更高的召回")
    ax.grid(alpha=0.3, which="both")
    ax.margins(x=0.12, y=0.12)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_pareto.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已写出 {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    name = setup_cjk_font()
    print(f"中文字体：{name or '未找到（图里的中文会变成方框）'}")
    os.makedirs(args.outdir, exist_ok=True)

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    print(f"语料 {len(docs)} 篇 / {sum(len(t) for t in docs.values())} 字，query {len(queries)} 条\n")

    res = measure(docs, queries)
    plot_heatmap(res, args.outdir)
    plot_type_recall(res, args.outdir)
    plot_recall_vs_k(res, args.outdir)
    plot_pareto(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
