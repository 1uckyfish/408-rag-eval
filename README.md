# 408-RAG-Eval：用 408 真题评测 RAG 的检索策略

> 一句话：**把《操作系统》知识库 + 408 历年真题做成一个可复现的检索评测台，用网格实验回答"chunk 该切多大、dense 该不该换、瓶颈在召回还是在排序"。**

这不是一个 RAG demo，是一个**测量项目**。所有的价值在 `results/` 里的表格和图，不在界面。

---

## 1. 这个项目要回答的三个问题

| 编号 | 问题 | 交付形式 |
|---|---|---|
| Q1 | chunk size × overlap 怎么选？教材场景下默认的 512/0 是不是最优？ | 热力图 + 结论句 |
| Q2 | dense（向量）检索比 BM25 强多少？差在哪些题上？ | 对比表 + 失败案例分析 |
| Q3 | rerank 把 recall 提多少、延迟涨多少？值不值？ | **做了单点：负收益**（recall@5 −2.6pt、P95 ×1483），原因分析见 `docs/04` |

**结论（2026-11-11 实测：36 篇语料 / 306 段 / 120 条 query / 8 组配置）**

> **Q1 切分**：`size=512` 就是该选的档位，overlap 取 0 或 64 差异都在噪声内（512/0 作为默认值足够）。
> recall@5 确实随 size 一路涨到 0.9292（1024/64），但**这是假象**——chunk 总数从 591 掉到 93，top-5 占全库的比例从 0.8% 升到 5.4%，分母越来越小自然容易命中。换成更严的 recall@1 就露馅了：512 见顶 **0.5264**，1024 两档都回落（0.5028 / 0.4903）。overlap=64 在 512 上只 +3.9pt（0.8403 → 0.8792），相当于 4–5 道题，**样本量不足以下"overlap 有用"的结论**。
>
> **Q2 dense vs BM25**：BM25 全面胜出，**没有理由换**。recall@5 **0.8403 vs 0.7600**、MRR **0.6830 vs 0.5590**、P95 **1.5ms vs 15.9ms**（慢 10 倍还更不准）。原因是 query 是术语密集的短句、语料又按考点组织，字符 bigram 恰好能吃到关键词；换成真实用户的长问句，这个结论未必成立。
>
> **Q3 rerank**：**做了，是负收益**——这条负结果比"没做"更有信息量。BM25 top-20 交给 `bge-reranker-base` 重排后，recall@5 **0.8403 → 0.8139**（−2.6pt）、MRR **0.6829 → 0.6494**（−3.3pt）、recall@1 0.5264 → 0.4806（−4.6pt），而 P95 从 **2.69 ms 涨到 3 988 ms**（×1483）。逐题看是涨 8 / 跌 14 / 平 98。
>
> 为什么立项时觉得该做：120 条里 **37 条"召回到了却没排第一"**（桶 B），只有 **16 条完全没召回**（桶 A）——瓶颈确实在排序；单纯清洗 query 编号噪声只换来 +0.83pt，也说明改 query 没用。但**"该重排"不等于"这个 reranker 会重排对"**：它按通用语义相关性打分，而我们要的是"这一段能不能答这道题"。三个可能原因（打分单元是 chunk 而 gold 是段、通用 reranker 与教材式密集术语不匹配、512 token 截断）与三条下一步验证写在 `docs/04` 的 Q3 节。

**读这三个数的前提**（面试被追问时的诚实边界）：

- 延迟**不可复现**：同机同代码两次跑差 3.3 倍（2.81ms vs 9.38ms），所以只讲"个位数 ms 对十几 ms"，不讲小数；
- 语料是**按 408 大纲自撰**的 36 篇（规避教材版权），措辞比真教材干净，BM25 可能偏乐观；
- gold 是我自己标的，**隔日盲标一致率 40%**，但 `recall@5` 只差 2.5pt（主结论稳），`MRR` 差 16.7pt（所以对外只讲 recall@5）。


---

## 2. 目录

```
项目/
├── README.md               ← 你在这里
├── docs/
│   ├── 00-选题与定位.md     为什么是 C，答辩怎么讲，结果不显著怎么办
│   ├── 01-八周计划.md       周计划、里程碑、验收标准、3 周精简版
│   ├── 02-技术方案.md       架构、变量网格、指标定义、统计方法
│   ├── 03-数据与标注.md     语料、120 条 query、gold 段落标注协议、JSON 格式
│   ├── 04-实验记录.md       每次跑实验的记录（8 组网格、Q1/Q2、拐点、稳定性）
│   ├── 05-方法问答.md       三层追问 15 问 + 答题要点
│   ├── 06-失败案例.md       6 条具体案例 + A/B/C 三类分布 + 消融实验
│   ├── 07-本地环境.md       venv、torch、模型下载的踩坑记录
│   ├── 08-真题收集.md       195 题真题怎么从 PDF 变成 exam.md
│   ├── 09-执行清单.md       11 步线性清单与三个 Gate
│   └── 10-口径与细节笔记.md  滚动记录口径细节（将来合并进 05）
├── src/                   26 个脚本 / 约 2 200 行（核心链零依赖）
│   ├── chunker.py          切分（零依赖）
│   ├── bm25.py             字符 bigram BM25（零依赖）
│   ├── metrics.py          recall@k / MRR / nDCG（零依赖）
│   ├── eval.py             单次评测入口（零依赖）
│   ├── run_grid.py         8 组网格（零依赖）
│   ├── ask.py              当检索器用：问一句，打印 top-k 段落（--cite 出引用）
│   ├── dense_eval.py       dense 检索（需要 .venv）
│   ├── find_failures.py    失败案例挖掘（A/B/C 分桶）
│   ├── ablation_idnoise.py 消融：剥掉 query 里的进程编号
│   ├── plot_figures.py     图 0–3 一键出图
│   ├── plot_recall1.py     图 4：recall@1 vs recall@5 拐点
│   ├── plot_delta.py       图 5：最优 vs 默认的逐题差值
│   ├── plot.py / plot_chunks.py  手写版热力图（学习用）
│   ├── check_corpus.py / check_queries.py / check_exam.py   三道数据校验
│   ├── suggest_gold.py / apply_gold.py    gold 候选与回填
│   ├── make_consistency_sample.py / check_consistency.py    一致率抽检
│   ├── show_paras.py       打印某篇的段落序号（标注用）
│   ├── render_exam.py      扫描版 PDF → PNG
│   └── merge_parts.py / merge_queries.py   分片合并
├── app.py                  Streamlit 演示外壳（需 .venv，只包装不产结论）
├── requirements-demo.txt   演示页依赖（核心评测不需要）
├── data/
│   ├── corpus/os-01…36.md  36 篇 / 306 段 / 6.8 万字（按 408 大纲自撰）
│   ├── raw/exam.md         195 道真题（2009–2024，不进公开仓库）
│   └── queries.jsonl       120 条检索式 query + gold 段落
└── results/                CSV 与图（fig0–fig5 + 两份手写版）
```

## 3. 现在就能用

**它不是问答系统，没有界面。** 它有两层用法：一层是"当检索器用"（问一句，看检索到哪几段），一层是"当评测台用"（对比配置，看哪个更好）。

### 3.1 当检索器用：问一句，看命中的段落

```powershell
cd <项目根目录>   # 例：cd "D:\code\408-rag-eval"
py -3 src\ask.py "页表项 2B、页大小 2^10B 时页目录最少多少项"
py -3 src\ask.py "进程和线程的区别是什么" --k 3 --full
py -3 src\ask.py "死锁的四个必要条件" --cite   # 引用式回答：段落原文 + 出处 + 相关度
py -3 src\ask.py --interactive          # 连续提问，回车退出
```

输出形如：

```
语料：36 篇 / 306 段  切分：size=512 overlap=0 → 160 chunks

[1]   29.92  os-27  第 2–3 段  （字符 514–1011）
    转换时先由页表基址寄存器找到页目录，用页目录索引乘 4 加基址得到目录项地址…

[2]   29.89  os-27  第 0–1 段  （字符 0–514）
    32 位逻辑地址、页大小 4KB、页表项 4B 时，页内偏移 12 位…
```

它返回的是**段落原文**（带篇号、段号、BM25 分数），不是生成的答案。要生成答案得再接一个 LLM，那是另一件事，本项目的评测口径只到"检索对不对"。

加 `--cite` 则输出"引用式回答"——把 top-1 命中的那一段原文直接给你，并注明出处和相关度：

```
问题：死锁的四个必要条件是什么
────────────────────────────────────────────────────────
根据 os-06 第 1 段（共 192 字）：

  死锁必须同时具备四个必要条件。互斥条件指……循环等待条件指……

── 出处：os-06 第 1 段 · 相关度 30.42 · 该 chunk 覆盖第 0–2 段
── 其他候选：os-06 第 2 段（17.2）、os-06 第 5 段（10.2）
────────────────────────────────────────────────────────
说明：答案即检索到的原文段落，未做生成式改写（BM25 无生成能力）。
```

注意最后那行：**这不是"模型作答"，是"把检索到的那段原样引用"**。故意不接 LLM，因为一接生成，评测对象就从"检索策略"变成"生成质量"，三周的结论就作废了。答辩被问"为什么不做成问答机器人"，这就是答案。

### 3.2 当评测台用：对比配置

```powershell
# 单次评测（一组配置，写 CSV）
py -3 src\eval.py --corpus data\corpus --queries data\queries.jsonl --size 512 --overlap 0 --k 5 --out results\baseline_bm25.csv

# 8 组网格（size 4 档 × overlap 2 档）
py -3 src\run_grid.py --out results\grid_bm25.csv

# 失败案例挖掘 / 出图
py -3 src\find_failures.py
py -3 src\plot_figures.py
```

> ⚠️ 你这台机器上的 `python` 指向 Microsoft Store 的空壳（运行无输出、退出码 1），
> 真正可用的是 `py -3`（Python 3.13.9，带 pip）。dense 相关脚本要用 `.\.venv\Scripts\python.exe`。

### 3.3 它覆盖了多少 408 内容

36 篇 / 306 段 / 6.8 万字，按 408 操作系统大纲自己写的章节摘要（规避教材版权），不是教材全文。**能检索到讲解和算例，但覆盖面比王道/教材窄** —— 这也是 docs/06 里几类失败案例的根因之一。

### 3.4 演示页（可选，需要额外依赖）

想要"能点的界面"而不是终端输出，用 Streamlit 外壳。**核心评测链不依赖它**（BM25 部分仍然零依赖），所以单独装在 `.venv` 里：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开终端提示的 `http://localhost:8501`。页面上能选 size/overlap、调 k、输入问题，第一条结果按"引用式回答"展示（段落原文 + 出处 + 相关度），侧栏显示这组配置的 recall@5 / recall@1 / chunk 数。

它**只是把 `ask.py --cite` 的结果画成卡片**，不产生任何新结论；离线的、可复现的结论仍然只认 `results/` 里的 CSV 和 `docs/04`。

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
- [x] 真题题库：2009–2024 共 195 道 OS 题（选择 160 / 综合 35）进 `data/raw/exam.md`，`check_exam.py` 通过（2026-11-04）
- [x] 检索式 query：`data/queries.jsonl` 120 条（概念 53 / 计算 44 / 综合 23），`check_queries.py --min 120` 通过（2026-11-06）
- [x] W1 数据补全：语料 18 篇 / 166 段（`check_corpus.py`），120 条 query 的 gold 全部标完（1 段 94 / 2 段 24 / 3 段 2），`check_queries.py --min 120` 通过（2026-11-07）
- [x] W1 基线：BM25 512/0 recall@5 = 0.9306（75 chunks）→ 判定为语料规模造成的天花板（2026-11-07）
- [x] W2 语料扩张：18 → **36 篇 / 306 段 / 6.8 万字**，近重复 0 对；512/0 recall@5 回落到 **0.8403**（160 chunks），recall@1 0.6264 → 0.5264（2026-11-08）
- [x] W2 出图：分题型 recall、size×overlap 热力图 + recall vs chunk 数、recall@k 曲线、recall vs P95 帕累托（`results/fig0..fig3*.png`，`py -3 src\plot_figures.py`）（2026-11-08）
- [x] W2 环境：`.venv` + torch 2.15.0+cpu + sentence-transformers 6.2.0，bge-small-zh-v1.5（512 维）跑通（2026-11-08）
- [x] W2 Q2 对照：dense 512/0 recall@5 **0.760 / P95 16ms** vs BM25 **0.840 / 1.5ms** —— BM25 又准又快，不换（2026-11-08）
- [x] W2 补图：recall@1 拐点（512 见顶）、最优 vs 默认的逐题差值、手写版热力图两张（练习用，在 `results/showcase/`）（2026-11-09）
- [x] W3 失败分析：A 16 / B 37 / C 4，**瓶颈在排序不在召回**；6 条案例写进 `docs/06`；编号噪声消融 +0.83pt（2026-11-09）
- [x] 标注一致性抽检：20 条藏答案重标，严格一致 40%，但 recall@5 只差 2.5pt、MRR 差 16.7pt（`docs/04`）（2026-11-10）
- [x] 检索入口与演示：`src/ask.py`（含 `--cite` 引用式回答）、`app.py` Streamlit 外壳（2026-11-10）
- [x] 冷启动复现验证：干净 clone 下三条命令全绿，`grid_bm25.csv` 数值逐项一致（2026-11-11）
- [ ] 结论句汇总 + `docs/05` 按实测数字重写　← 现在这里
- [ ] 可选：rerank 单点、2022–2024 真题补录

## 7. 复现（2026-11-11 冷启动实测）

在一个**全新的 `git clone`** 里、只用系统 Python（不带 `.venv`、不带任何本地缓存）跑通，记录如下。

### 7.1 要求

- Python ≥ 3.11（实测 3.13.9）。核心评测链**只用标准库**；画图需要 `numpy` + `matplotlib`。
- 不需要联网、不需要 GPU、不需要大模型。

### 7.2 三条命令（实测耗时）

```powershell
git clone <仓库地址> && cd 项目
py -3 src\check_corpus.py --min-files 36    # 约 1s，期望 36 篇 / 306 段 / 0 对近似重复
py -3 src\run_grid.py                       # 约 5s，8 组配置，写 results/grid_bm25.csv
py -3 src\plot_figures.py                   # 约 3s，写 results/fig0~fig3
```

想连图 4、图 5 一起复现，再加两条（各约 10s）：

```powershell
py -3 src\plot_recall1.py                   # results/fig4_recall1_vs_5.png + grid_recall_k.csv
py -3 src\plot_delta.py                     # results/fig5_delta_distribution.png
```

### 7.3 复现判据：数值一致，延迟仅供参考

干净 clone 跑出的 `grid_bm25.csv` 与原仓库**逐项相同**（4 档 size × 2 档 overlap 全对，含 chunks/MRR/nDCG）。

但 **P95 延迟复现不出来**：同机同代码，128/0 这次是 2.81ms，原仓库记的是 9.38ms（3.3 倍差，取决于当时机器负载）。所以：

> **对外只讲"指标可复现"，延迟只讲量级（个位数 ms 对十几 ms），不讲具体小数。**

### 7.4 冷启动查出的三个真问题

| 问题 | 现象 | 处理 |
|---|---|---|
| 语料被 `.gitignore` 排除 | clone 后 `run_grid.py` 报 `FileNotFoundError: 'data/corpus'`，三条命令全挂 | 已让 `data/corpus/` 入库（自撰内容不涉版权） |
| embedding 缓存被提交 | 仓库白带 16MB `.npy` 二进制 | 已 `git rm --cached` 并加入 `.gitignore` |
| 延迟数字不可复现 | 同上 3.3 倍差 | 见 7.3，只报量级 |

### 7.5 已知的"需要自备"部分

- `data/raw/`（真题 PDF、`exam.md`）**不入库**：扫描原卷有版权，只在本地。所以干净 clone 里跑 `py -3 src\check_exam.py` 会报"找不到 data/raw/exam.md"，这是预期行为。
- 派生的 `data/queries.jsonl`（120 条检索式 query）**已入库**，评测不依赖真题原件。
- `app.py` 需要 `.venv` 与 `requirements-demo.txt`，见 §3.4。

---

**在线仓库**：<https://github.com/1uckyfish/408-rag-eval>（2026-11-11 首次公开，32 次提交、101 个文件）

`data/raw/` 与 `data/cache/` 不入库：前者是真题 PDF/转录的版权考虑，后者是可再生的 embedding 缓存。因此 clone 下来后 `check_exam.py` 会提示"找不到 exam.md"，这是预期行为，不影响评测链（`queries.jsonl` 已入库）。
