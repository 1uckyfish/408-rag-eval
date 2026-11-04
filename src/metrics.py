"""检索指标：recall@k / MRR@k / nDCG@k。

约定：ranked 是"已排序的结果列表"，每个元素是**命中的 gold 集合**。
因为一个 chunk 可能覆盖多个段落，也可能一个都不覆盖（空集）。
gold 是被标注的段落集合，元素形如 ("os-001", 3)。
"""

import math


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    found = set()
    for s in ranked[:k]:
        found |= (s & gold)
    return len(found) / len(gold)


def mrr_at_k(ranked, gold, k):
    for i, s in enumerate(ranked[:k], start=1):
        if s & gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    dcg, seen = 0.0, set()
    for i, s in enumerate(ranked[:k], start=1):
        new = (s & gold) - seen
        if new:
            seen |= new
            dcg += 1.0 / math.log2(i + 1)   # 二值增益，每条 gold 只算一次
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / idcg if idcg else 0.0
