# 408-RAG-Eval：用 408 真题评测 RAG 的检索策略

> 一句话：**把《操作系统》知识库 + 408 历年真题做成一个可复现的检索评测台，用网格实验回答"chunk 到底该切多大、rerank 到底值不值"。**

这不是一个 RAG demo，是一个**测量项目**。所有的价值在 `results/` 里的表格和图，不在界面。

---

## 1. 这个项目要回答的三个问题

| 编号 | 问题 | 交付形式 |
|---|---|---|
| Q1 | chunk size × overlap 怎么选？教材场景下默认的 512/0 是不是最优？ | 热力图 + 结论句 |
| Q2 | dense（向量）检索比 BM25 强多少？差在哪些题上？ | 对比表 + 失败案例分析 |
| Q3 | rerank 把 recall 提多少、延迟涨多少？值不值？ | 帕累托图（recall vs P95 延迟） |

**最终结论模板**（做完再填）：
> 在 200 条 408 真题上，chunk=**X** / overlap=**Y** 的 recall@5 比默认 512/0 高 **Z** 个百分点，P95 延迟只增加 **W** ms；rerank 带来 **+A** recall 但代价是 **+B** ms，在 5 条结果返回的场景下**不划算/划算**。

---

## 2. 目录

```
项目/
├── README.md               ← 你在这里
├── docs/
│   ├── 00-选题与定位.md     为什么是 C，答辩怎么讲，结果不显著怎么办
│   ├── 01-八周计划.md       周计划、里程碑、验收标准、3 周精简版
│   ├── 02-技术方案.md       架构、变量网格、指标定义、统计方法
│   ├── 03-数据与标注.md     语料、200 条 query、gold 段落标注协议、JSON 格式
│   ├── 04-实验记录.md       每次跑实验的记录模板（照抄）
│   └── 05-方法问答.md       三层追问 15 问 + 答题要点
├── src/
│   ├── chunker.py          切分（零依赖）
│   ├── bm25.py             字符 bigram BM25（零依赖）
│   ├── metrics.py          recall@k / MRR / nDCG（零依赖）
│   ├── eval.py             单次评测入口（零依赖，现在就能跑）
│   └── run_grid.py         网格实验（零依赖，现在就能跑）
├── data/
│   ├── corpus/*.md         语料（先放了两篇示例，换成你自己的笔记）
│   └── queries.jsonl       标注好的 query（先放了 5 条示例）
└── results/                每次实验的 CSV 和图表（.gitignore 建议只留最终版）
```

## 3. 现在就能跑（零依赖，2 分钟）

Windows（PowerShell）——**用 `py -3`，不要用 `python`**：
```powershell
cd "C:\Users\W2490\Desktop\文件夹\项目"
py -3 src\eval.py --corpus data\corpus --queries data\queries.jsonl --size 300 --overlap 60 --k 5
py -3 src\run_grid.py --corpus data\corpus --queries data\queries.jsonl --out results\grid_bm25.csv
```

> ⚠️ 你这台机器上的 `python` 指向 Microsoft Store 的空壳（运行无输出、退出码 1），
> 真正可用的是 `py -3`（Python 3.13.9，带 pip）。
> WSL 里则**根本没有 python3**（实测 `command not found`），要用得先 `sudo apt install python3`。
> 现阶段用 Windows 就够了，别为了一个零依赖脚本折腾 WSL。

跑通说明：切分 → 建索引 → 检索 → 打分 → 汇总，整条链路是通的，只是语料和 query 还是玩具规模。

> ⚠️ 示例数据上 recall 会全是 1.0——2 篇文档、5 条 query、k=5，闭着眼都能全召回。**这恰好说明为什么必须换成 200 条真题 + 10 篇以上语料**：指标饱和了就没有区分度。换数据的动作在 docs/03。

## 4. 节奏：已选定 3 周版（11/22 交付）

| 周 | 日期 | 做什么 | 交付物 |
|---|---|---|---|
| W1 | 11/02–11/08 | 120 条真题 query + gold 段落 + 语料整理 + BM25 默认配置 | `results/baseline_bm25.csv` |
| W2 | 11/09–11/15 | size × overlap 8 组网格 + 热力图 | `results/fig1_heatmap.png`、Q1 结论 |
| W3 | 11/16–11/22 | 失败分析 + README + 复现命令 | 可交付仓库 |

dense 与 rerank 降级为"提前做完才做"（见 docs/01 末节）。**小而完整 > 大而烂尾。**
八周版仍在 docs/01 里，作为参照与后续扩展（答辩前若还有时间，从 W3 dense 接上）。

## 5. 风险与预案（选 C 的核心风险就是"跑不出结论"）

| 风险 | 触发信号 | 预案 |
|---|---|---|
| 没预算/没网 | API 调不通 | 全本地：bge-small-zh + BM25，成本 0 |
| 结果不显著 | 各配置 recall 差 <2pt | 改问"为什么没差别"，做失败分析（短 query 下 dense 退化），负结果也是结论 |
| 标注太慢 | 200 题标两天还没完 | 降到 100 题，先跑通再扩 |
| 被质疑调包 | 说不清 splitter 规则 | 切分自己写（`src/chunker.py` 已经是自己的），不用 LangChain |

## 6. 状态

- [x] 项目骨架 + 零依赖评测链路跑通（2026-11-02）
- [x] 选定 3 周精简版，日期落到 11/22（2026-11-03）
- [x] 本地环境方案定稿：`docs/07-本地环境.md`（Windows venv + CPU torch + bge-small，可选 GPU）
- [x] `src/check_local_env.py` 环境自检脚本
- [x] 真题题库：2009–2021 共 159 道 OS 题（选择 130 / 综合 29）进 `data/raw/exam.md`，`check_exam.py` 通过（2026-11-04）
- [x] 检索式 query：`data/queries.jsonl` 120 条（概念 53 / 计算 44 / 综合 23），`check_queries.py --min 120` 通过（2026-11-06）
- [x] W1 数据补全：语料 18 篇 / 166 段（`check_corpus.py`），120 条 query 的 gold 全部标完（1 段 94 / 2 段 24 / 3 段 2），`check_queries.py --min 120` 通过（2026-11-07）
- [x] W1 基线：BM25 512/0 recall@5 = 0.9306（75 chunks）→ 判定为语料规模造成的天花板（2026-11-07）
- [x] W2 语料扩张：18 → **36 篇 / 306 段 / 6.8 万字**，近重复 0 对；512/0 recall@5 回落到 **0.8403**（160 chunks），recall@1 0.6264 → 0.5264（2026-11-08）
- [x] W2 出图：`results/fig0_type_recall.png`、`fig1_heatmap.png`、`fig2_recall_vs_k.png`（36 篇数据，`plot_figures.py` 现算）（2026-11-08）
- [x] W2 出图：分题型 recall、size×overlap 热力图 + recall vs chunk 数、recall@k 曲线（`results/fig0/1/2*.png`，`py -3 src\plot_figures.py`）（2026-11-08）
- [ ] W2 热力图结论句 + dense 单点对比　← 现在这里
- [ ] W3 失败分析 + README
- [ ] 可选：dense 单点对比 / rerank
