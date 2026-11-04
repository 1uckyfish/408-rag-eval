"""检查本地环境是否具备跑 dense 检索的条件。

用法：python src/check_local_env.py
不需要 torch 也能运行（会告诉你缺什么）。
"""

import importlib.util
import sys
import time


def has(mod):
    return importlib.util.find_spec(mod) is not None


def main():
    print(f"python      {sys.version.split()[0]}")
    print(f"numpy       {'ok' if has('numpy') else 'MISSING'}")
    print(f"torch       {'ok' if has('torch') else 'MISSING'}")
    print(f"st          {'ok' if has('sentence_transformers') else 'MISSING'}")

    if not has("torch") or not has("sentence_transformers"):
        print("\n=> 还差依赖。按 docs/07-本地环境.md 第 3 节安装。")
        return

    import torch
    from sentence_transformers import SentenceTransformer

    print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         {torch.cuda.get_device_name(0)}")

    t0 = time.perf_counter()
    m = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"model load  {time.perf_counter() - t0:.1f}s  dim={m.get_sentence_embedding_dimension()}")

    texts = ["进程是资源分配的基本单位，线程是调度的基本单位。"] * 64
    t0 = time.perf_counter()
    m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    dt = time.perf_counter() - t0
    print(f"encode 64   {dt:.2f}s  -> 3000 chunk 预估 {dt / 64 * 3000:.0f}s")
    print("\n=> 环境可用。")


if __name__ == "__main__":
    main()
