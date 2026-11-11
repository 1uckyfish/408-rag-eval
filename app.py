r"""408 检索评测台的网页版（Streamlit）。

定位：**演示外壳**，不是产品。它只做三件事 —— 把问题喂给 BM25、把命中的段落
原文摆出来、把当前配置的指标挂在旁边。不做生成、不接大模型。

为什么单独一个文件、还要单独 requirements：核心评测链（eval.py / run_grid.py）
一直是零依赖，随手能跑；这个页面引入了 streamlit + pyarrow 约 60MB，
所以隔离在 `.venv` 里，README 标明"演示可选"。

启动：
    .\.venv\Scripts\python.exe -m streamlit run app.py
然后浏览器打开终端里给出的 http://localhost:8501
"""

import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from bm25 import tokenize                                    # noqa: E402
from eval import build_index, load_corpus, paragraphs        # noqa: E402

CORPUS = os.path.join(HERE, "data", "corpus")
SIZES = [128, 256, 512, 1024]
OVERLAPS = [0, 64]
K_EVAL = 5


@st.cache_resource(show_spinner=False)
def get_index(size, overlap):
    """建索引很贵（160 chunk 约 20ms，128/64 要 0.5s），按配置缓存住。"""
    docs = load_corpus(CORPUS)
    paras = {d: paragraphs(t) for d, t in docs.items()}
    chunks, index = build_index(docs, size, overlap)
    return docs, paras, chunks, index


def best_para(docs, paras, covered, query):
    """chunk 跨多段时，挑与问题词重合最多的那段当"被引用的段落"。"""
    qs = set(tokenize(query))
    best, best_hit = None, -1
    for doc, idx in covered:
        s, e = paras[doc][idx]
        hit = len(qs & set(tokenize(docs[doc][s:e])))
        if hit > best_hit:
            best, best_hit = (doc, idx), hit
    return best


def para_text(docs, paras, doc, idx):
    s, e = paras[doc][idx]
    return docs[doc][s:e].replace("\n", "")


def grid_metrics(size, overlap):
    """从现有 CSV 里读这组配置的指标，读不到就返回 None（不编数）。"""
    import csv
    row = {}
    # grid_bm25.csv 的列名是 recall/mrr/ndcg（无 @5 后缀），网格全量指标在这里
    path = os.path.join(HERE, "results", "grid_bm25.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    if int(float(r["size"])) == size and int(float(r["overlap"])) == overlap:
                        row["recall@5"] = r.get("recall", "")
                        row["mrr"] = r.get("mrr", "")
                        row["ndcg"] = r.get("ndcg", "")
                        row["chunks"] = r.get("chunks", "")
                except (KeyError, ValueError):
                    continue
    # recall@1 只有第二个 CSV 里有
    path = os.path.join(HERE, "results", "grid_recall_k.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    if int(float(r["size"])) == size and int(float(r["overlap"])) == overlap:
                        row["recall@1"] = r.get("recall@1", "")
                        row.setdefault("recall@5", r.get("recall@5", ""))
                except (KeyError, ValueError):
                    continue
    return row or None


def main():
    st.set_page_config(page_title="408 检索评测台", page_icon="🔎", layout="wide")

    with st.sidebar:
        st.subheader("切分配置")
        size = st.selectbox("chunk size（字）", SIZES, index=2)
        overlap = st.selectbox("overlap（字）", OVERLAPS, index=0)
        k = st.slider("返回前 k 段", 1, K_EVAL, 3)
        show_full = st.checkbox("显示整段原文", value=True)
        st.divider()
        st.subheader("这组配置的指标")
        m = grid_metrics(size, overlap)
        if m:
            st.metric("recall@5", m.get("recall@5", "—"))
            c1, c2 = st.columns(2)
            c1.metric("recall@1", m.get("recall@1", "—"))
            c2.metric("chunks", m.get("chunks", "—"))
        else:
            st.caption("这组没跑过网格，没有指标可报")
        st.divider()
        st.caption("检索器：BM25（字符 bigram），纯 Python 实现，零依赖。\n\n"
                   "答案 = 命中的段落原文，不做生成式改写。")

    st.title("408 检索评测台")
    st.caption("问一个操作系统问题，看它检索到哪一段。这是评测台的可视化外壳，"
               "结论性数字见 docs/04 与 results/。")

    docs, paras, chunks, index = get_index(size, overlap)
    st.caption(f"语料 {len(docs)} 篇 / {sum(len(v) for v in paras.values())} 段　"
               f"→ {len(chunks)} chunks（size={size}, overlap={overlap}）")

    query = st.text_input("问题", placeholder="例如：死锁的四个必要条件是什么",
                          label_visibility="collapsed")

    if not query.strip():
        st.info("输入一个问题后回车。示例：死锁的四个必要条件 / 进程和线程的区别 / "
                "FIFO 与 LRU 的区别")
        return

    hits = index.search(query, top_k=k)
    if not hits:
        st.warning("没有命中任何段落")
        return

    # 第一条做成"引用式回答"卡片
    ci, score = hits[0]
    c = chunks[ci]
    doc, idx = best_para(docs, paras, c["covered"], query)
    st.markdown(f"#### 根据 `{doc}` 第 {idx} 段")
    st.success(para_text(docs, paras, doc, idx))
    ps = sorted(p for d, p in c["covered"] if d == c["doc"])
    span = f"第 {ps[0]}–{ps[-1]} 段" if ps else "—"
    st.caption(f"出处：{doc} 第 {idx} 段 · 相关度 {score:.2f} · 该 chunk 覆盖 {span}"
               f" · 该段 {len(para_text(docs, paras, doc, idx))} 字")

    if len(hits) > 1:
        st.markdown("#### 其他候选")
        for rank, (ci2, sc2) in enumerate(hits[1:], 2):
            c2 = chunks[ci2]
            d2, p2 = best_para(docs, paras, c2["covered"], query)
            body = para_text(docs, paras, d2, p2)
            title = f"[{rank}] {sc2:.2f}　{d2} 第 {p2} 段"
            with st.expander(title):
                st.write(body if show_full else body[:120] + "…")

    st.divider()
    st.caption("说明：以上是检索结果，不是模型作答。评测口径只到「检索对不对」，"
               "不接 LLM 是为了不让生成质量混进检索指标。")


if __name__ == "__main__":
    main()
