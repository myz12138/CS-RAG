import re, time, numpy as np

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def f1_score(pred, gold):
    def tok(s):
        return re.findall(r"\w+", s.lower())
    p = tok(pred); g = tok(gold)
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = num_same / len(p)
    recall = num_same / len(g)
    return 2*precision*recall/(precision+recall+1e-8)

def recall_at_k(retrieved_ids, gold_ids, k):
    R = set(retrieved_ids[:k]); G = set(gold_ids)
    return int(len(R & G) > 0)

def avg(lst):
    return sum(lst)/max(1,len(lst))
