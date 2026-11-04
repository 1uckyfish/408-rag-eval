"""校验 data/raw/exam.md 的结构与覆盖度（零依赖）。

用法：
    py -3 src/check_exam.py                # 只报告
    py -3 src/check_exam.py --min 150      # 不足 150 题则退出码 1（CI/自检用）

exam.md 约定
------------
    ## 2009                     <- 年份段
    ### 2009-23                 <- 一道题（id 必须形如 <年份>-<题号>）
    - 题型: 选择 | 综合
    - 考点: 进程管理/调度        <- 一级章/二级考点，用于分层分析
    - 难度: 易 | 中 | 难
    - 题干: ...
    - 选项A: ...                <- 选择题必须有 A-D
    - 答案: D                   <- 选择题答案字母；综合题写"见参考答案"
    - 解析: ...
    - 参考答案: ...             <- 综合题用，可多行（缩进续行）
"""

import argparse
import os
import re
import sys
from collections import Counter, OrderedDict

FIELDS = ("题型", "考点", "难度", "题干")

CHAPTERS = ("概述", "进程", "内存", "文件", "IO")


def parse(path):
    """返回 (items, errors)。item = dict(id, year, fields, extra)"""
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    items, errors = [], []
    year, cur, last_key = None, None, None
    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if line.startswith("## "):
            if cur:
                items.append(cur)
            year = line[3:].strip()
            cur, last_key = None, None
            continue
        if line.startswith("### "):
            if cur:
                items.append(cur)
            cur = {"id": line[4:].strip(), "year": year, "fields": {}, "line": lineno}
            last_key = None
            continue
        m = re.match(r"^- ([^:：]+)[:：]\s*(.*)$", line)
        if m and cur is not None:
            k, v = m.group(1).strip(), m.group(2).strip()
            cur["fields"][k] = v
            last_key = k
            continue
        if cur is not None and last_key and raw.startswith((" ", "\t")) and line.strip():
            cur["fields"][last_key] += "\n" + line.strip()
    if cur:
        items.append(cur)

    for it in items:
        f = it["fields"]
        miss = [k for k in FIELDS if not f.get(k)]
        if it["year"] is None:
            errors.append(f"{it['id']}: 不在任何 ## 年份段内")
        elif not it["id"].startswith(str(it["year"]) + "-"):
            errors.append(f"{it['id']}: id 与所在年份 {it['year']} 不一致")
        if miss:
            errors.append(f"{it['id']}: 缺字段 {','.join(miss)}")
        if f.get("题型") == "选择":
            opts = [k for k in f if re.fullmatch(r"选项[A-D]", k)]
            if len(opts) != 4:
                errors.append(f"{it['id']}: 选择题应有 4 个选项，实际 {len(opts)}")
            ans = f.get("答案", "")
            if not ans:
                errors.append(f"{it['id']}: 选择题缺 答案")
            elif not re.fullmatch(r"[A-D]", ans):
                errors.append(f"{it['id']}: 选择题答案应为 A-D，实际 {ans!r}")
        elif f.get("题型") == "综合":
            if not f.get("参考答案"):
                errors.append(f"{it['id']}: 综合题缺 参考答案")
        elif f.get("题型"):
            errors.append(f"{it['id']}: 题型只能是 选择/综合，实际 {f['题型']!r}")
    return items, errors


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", default="data/raw/exam.md")
    ap.add_argument("--min", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.exam):
        print(f"找不到 {args.exam}")
        return 1
    items, errors = parse(args.exam)

    sel = [i for i in items if i["fields"].get("题型") == "选择"]
    comp = [i for i in items if i["fields"].get("题型") == "综合"]
    print(f"# {args.exam}: 共 {len(items)} 题（选择 {len(sel)} / 综合 {len(comp)}）")

    by_year = Counter(i["year"] for i in items)
    print("\n## 每年题数")
    for y in sorted(by_year):
        n_s = sum(1 for i in items if i["year"] == y and i["fields"].get("题型") == "选择")
        print(f"  {y}: {by_year[y]:>3}  (选择 {n_s})")

    print("\n## 章节分布")
    chap = Counter()
    for i in items:
        c = (i["fields"].get("考点") or "未标").split("/")[0]
        chap[c] += 1
    for c, n in chap.most_common():
        print(f"  {c}: {n}")

    print("\n## 难度分布")
    for d, n in Counter(i["fields"].get("难度", "未标") for i in items).most_common():
        print(f"  {d}: {n}")

    dup = [q for q, n in Counter(re.sub(r"\s+", "", i["fields"].get("题干", ""))
                                 for i in items).items() if n > 1]
    if dup:
        print(f"\n!! 题干重复 {len(dup)} 处（建索引前需去重）")

    if errors:
        print(f"\n## 结构问题 {len(errors)} 处")
        for e in errors[:40]:
            print("  - " + e)
        if len(errors) > 40:
            print(f"  ... 其余 {len(errors) - 40} 处省略")
    else:
        print("\n结构检查通过 ✓")

    if args.min and len(items) < args.min:
        print(f"\n!! 题量 {len(items)} < 目标 {args.min}")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
