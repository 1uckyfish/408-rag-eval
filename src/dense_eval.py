"""dense 单点评测：切分 -> 编码（带缓存）-> 余弦检索 -> 打分。

用法（在项目根目录、激活 venv 后）：
    .\.venv\Scripts\python.exe src\dense_eval.py --size 512 --overlap 0 --k 5 \
        --out results\dense_512_0.csv

与 eval.py 的唯一区别是检索器：BM25 换成 bge-small-zh-v1.5 的句向量余弦相似度。
chunk 的 covered 判定、指标算法、CSV 列都与 eval.py 完全一致，保证可以直接对比。

缓存：data/cache/emb_<模型名>_<size>_<overlap>_<nchunks>_<hash8>.npy
只要语料或切分参数变了，hash 就变，自动重算，不会串味。
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunker import paragraphs, chunk          # noqa: E402
from metrics import recall_at_k, mrr_at_k, ndcg_at_k  # noqa: E402
from eval import load_corpus, load_queries     # noqa: E402

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
CACHE_DIR = os.path.join("data", "cache")


def build_chunks(docs, size, overlap):
    chunks = []
    for doc, text in docs.items():
        mids = [(s + e) / 2 for (s, e) in paragraphs(text)]
        for (s, e) in chunk(text, size, overlap):
            covered = {(doc, i) for i, m in enumerate(mids) if s <= m < e}
            chunks.append({"doc": doc, "start": s, "end": e,
                           "text": text[s:e], "covered": covered})
    return chunks


def cache_path(texts, model_name, size, overlap):
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8"))
    short = h.hexdigest()[:8]
    safe_model = model_name.replace("/", "_")
    return os.path.join(CACHE_DIR, f"emb_{safe_model}_{size}_{overlap}_{len(texts)}_{short}.npy")


def encode_chunks(model, texts, model_name, size, overlap):
    import numpy as np

    path = cache_path(texts, model_name, size, overlap)
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            print(f"# 复用 embedding 缓存 {path}  shape={arr.shape}")
            return arr
        print(f"# 缓存维度不匹配，重算：{path}")

    t0 = time.perf_counter()
    arr = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    dt = time.perf_counter() - t0
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, arr)
    print(f"# 编码 {len(texts)} 个 chunk 用了 {dt:.1f}s（{dt / max(1, len(texts)) * 1000:.1f} ms/个），已缓存 {path}")
    return arr


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if importlib.util.find_spec("sentence_transformers") is None:
        print("=> 缺 sentence-transformers。先按 docs/07-本地环境.md 第 3 节装依赖，"
              "并用 .venv 里的 python 运行本脚本。")
        return 2

    import numpy as np
    from sentence_transformers import SentenceTransformer

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    chunks = build_chunks(docs, args.size, args.overlap)

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model, device="cpu")
    print(f"# 模型 {args.model} 载入 {time.perf_counter() - t0:.1f}s，dim={model.get_sentence_embedding_dimension()}")

    emb = encode_chunks(model, [c["text"] for c in chunks], args.model, args.size, args.overlap)
    qvec = model.encode([q["query"] for q in queries], batch_size=32,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)

    sims = qvec @ emb.T                      # 都已归一化，点积即余弦
    rows, lat = [], []
    t_enc0 = time.perf_counter()
    for qi, q in enumerate(queries):
        gold = {(g["doc"], g["para"]) for g in q.get("gold", [])}
        # 延迟含"单条 query 编码 + 相似度排序"，与 eval.py 的 search 口径对齐
        t1 = time.perf_counter()
        qv = model.encode([q["query"]], batch_size=1, normalize_embeddings=True,
                          show_progress_bar=False, convert_to_numpy=True)[0]
        order = np.argsort(-(emb @ qv))[:args.k]
        dt = (time.perf_counter() - t1) * 1000
        for _ in range(3):
            qv = model.encode([q["query"]], batch_size=1, normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)[0]
            np.argsort(-(emb @ qv))[:args.k]
        lat.append(dt)
        ranked = [chunks[i]["covered"] for i in order]
        rows.append({
            "qid": q["qid"],
            "type": q.get("type", ""),
            "recall": recall_at_k(ranked, gold, args.k),
            "mrr": mrr_at_k(ranked, gold, args.k),
            "ndcg": ndcg_at_k(ranked, gold, args.k),
            "latency_ms": dt,
        })

    n = max(1, len(rows))
    summary = {
        "recall@k": sum(r["recall"] for r in rows) / n,
        "mrr@k": sum(r["mrr"] for r in rows) / n,
        "ndcg@k": sum(r["ndcg"] for r in rows) / n,
        "p95_ms": sorted(lat)[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
    }

    print(f"# 语料 {len(docs)} 篇 / {sum(len(t) for t in docs.values())} 字"
          f" | query {len(queries)} 条 | chunk {len(chunks)} 个")
    print(f"# dense {args.model} size={args.size} overlap={args.overlap} k={args.k}")
    print(f"{'qid':<6}{'type':<6}{'recall':>8}{'mrr':>8}{'ndcg':>8}{'ms':>9}")
    for r in rows:
        print(f"{r['qid']:<6}{r['type']:<6}{r['recall']:>8.3f}"
              f"{r['mrr']:>8.3f}{r['ndcg']:>8.3f}{r['latency_ms']:>9.2f}")
    print("-" * 45)
    print(f"{'MEAN':<12}{summary['recall@k']:>8.3f}{summary['mrr@k']:>8.3f}"
          f"{summary['ndcg@k']:>8.3f}{summary['p95_ms']:>9.2f}   (P95)")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
