"""字符 bigram 的 BM25（零依赖，中文不用分词也能打）。

中文不分词时，bigram 已经能抓住大部分术语匹配（"页表"→"页表"，
"缺页中断"→"缺页/页中/中断"），对一个基线来说够用了。
上限：不理解同义词，所以"TLB"搜不到"快表"。这正好是分析素材。
"""

import math
from collections import Counter


def tokenize(s):
    s = "".join(ch for ch in s.lower() if not ch.isspace())
    if len(s) < 2:
        return [s] if s else []
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(1, self.N)
        self.f = [Counter(d) for d in docs_tokens]
        df = Counter()
        for c in self.f:
            df.update(c.keys())
        self.idf = {
            t: math.log(1 + (self.N - df[t] + 0.5) / (df[t] + 0.5))
            for t in df
        }

    def _one(self, i, q):
        c, dl = self.f[i], len(self.docs[i])
        s = 0.0
        for t, qtf in q.items():
            tf = c.get(t)
            if not tf:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf * tf * (self.k1 + 1) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            ) * qtf
        return s

    def search(self, query, top_k=5):
        """返回 [(chunk_index, score), ...]，按分数降序。"""
        q = Counter(tokenize(query))
        scored = [(i, self._one(i, q)) for i in range(self.N)]
        scored.sort(key=lambda x: -x[1])
        return [x for x in scored[:top_k] if x[1] > 0]
