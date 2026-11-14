"""Metrics, standard library + torch only (no sklearn needed on Kaggle).

    python src/metrics.py      # self-test
"""

from __future__ import annotations

import math
import random


def accuracy(y, p):
    return sum(int(a == b) for a, b in zip(y, p)) / len(y) if y else float("nan")


def macro_f1(y, p, labels):
    f1s = []
    for c in labels:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        if tp + fp + fn == 0:
            continue  # class absent from both gold and predictions: not counted
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def confusion(y, p, labels):
    idx = {c: i for i, c in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for a, b in zip(y, p):
        m[idx[a]][idx[b]] += 1
    return m


def nll(gold_idx, probs):
    return sum(-math.log(max(pr[g], 1e-12)) for g, pr in zip(gold_idx, probs)) / len(probs)


def brier(gold_idx, probs):
    return sum(sum((pk - (1.0 if k == g else 0.0)) ** 2 for k, pk in enumerate(pr)) for g, pr in zip(gold_idx, probs)) / len(probs)


def ece(gold_idx, probs, bins=15):
    """Top-label expected calibration error (equal-width bins), in [0, 1]."""
    conf = [max(pr) for pr in probs]
    corr = [int(max(range(len(pr)), key=pr.__getitem__) == g) for g, pr in zip(gold_idx, probs)]
    n, tot = len(probs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            tot += len(sel) / n * abs(sum(corr[i] for i in sel) / len(sel) - sum(conf[i] for i in sel) / len(sel))
    return tot


def reliability(gold_idx, probs, bins=10):
    conf = [max(pr) for pr in probs]
    corr = [int(max(range(len(pr)), key=pr.__getitem__) == g) for g, pr in zip(gold_idx, probs)]
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if sel:
            out.append({"bin": [lo, hi], "n": len(sel), "conf": sum(conf[i] for i in sel) / len(sel),
                        "acc": sum(corr[i] for i in sel) / len(sel)})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative), ties count half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    allv = sorted((s, l) for s, l in zip(scores, labels))
    ranks, i = {}, 0
    while i < len(allv):  # average ranks for ties
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks.setdefault(allv[k][0], (i + j) / 2 + 1)
        i = j + 1
    rp = sum(ranks[s] for s in pos)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def average_precision(scores, labels):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    npos = sum(labels)
    if npos == 0:
        return float("nan")
    hit, ap = 0, 0.0
    for i, (_, l) in enumerate(pairs, 1):
        if l:
            hit += 1
            ap += hit / i
    return ap / npos


def norm_entropy(pr):
    k = len(pr)
    return -sum(p * math.log(p) for p in pr if p > 0) / math.log(k) if k > 1 else 0.0


def risk_coverage(gold_idx, probs, points=(1.0, 0.9, 0.8, 0.7, 0.5)):
    """Accuracy on the most confident fraction of examples (abstain on the rest)."""
    order = sorted(range(len(probs)), key=lambda i: -max(probs[i]))
    out = []
    for cov in points:
        k = max(1, round(cov * len(order)))
        sel = order[:k]
        out.append({"coverage": cov, "accuracy": sum(int(max(range(len(probs[i])), key=probs[i].__getitem__) == gold_idx[i]) for i in sel) / k})
    return out


def chance_skill(raw, chance):
    return max(0.0, min(1.0, (raw - chance) / (1 - chance))) if chance < 1 else float("nan")


def paired_bootstrap(a_correct, b_correct, n=1000, seed=0):
    """95% CI of mean(a) - mean(b) over resampled examples (paired)."""
    rng = random.Random(seed)
    m = len(a_correct)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(a_correct[i] - b_correct[i] for i in idx) / m)
    diffs.sort()
    return {"diff": sum(a - b for a, b in zip(a_correct, b_correct)) / m, "lo": diffs[int(0.025 * n)], "hi": diffs[int(0.975 * n) - 1]}


def cluster_bootstrap(values, clusters, n=1000, seed=0):
    """95% CI of the mean of `values` when examples come in clusters (e.g. all renders/tiers of one script):
    resamples whole clusters, so near-duplicates don't make the interval look tighter than it is."""
    rng = random.Random(seed)
    by = {}
    for v, c in zip(values, clusters):
        by.setdefault(c, []).append(v)
    keys = list(by)
    means = []
    for _ in range(n):
        pick = [by[keys[rng.randrange(len(keys))]] for _ in keys]
        tot = sum(sum(x) for x in pick)
        cnt = sum(len(x) for x in pick)
        means.append(tot / cnt)
    means.sort()
    return {"mean": sum(values) / len(values), "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n) - 1],
            "n_clusters": len(keys)}


def paired_cluster_bootstrap(a_correct, b_correct, clusters, n=1000, seed=0):
    """95% CI of mean(a) - mean(b), resampling clusters (scripts) with all their clips."""
    d = [x - y for x, y in zip(a_correct, b_correct)]
    out = cluster_bootstrap(d, clusters, n, seed)
    return {"diff": out["mean"], "lo": out["lo"], "hi": out["hi"], "n_clusters": out["n_clusters"]}


def _selftest():
    y, p = ["a", "a", "b", "c"], ["a", "b", "b", "c"]
    assert accuracy(y, p) == 0.75
    assert abs(macro_f1(y, p, ["a", "b", "c"]) - (2 / 3 + 2 / 3 + 1) / 3) < 1e-9
    probs = [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]]
    g = [0, 1, 1]
    assert abs(nll(g, probs) - (-(math.log(0.9) + math.log(0.4) + math.log(0.8)) / 3)) < 1e-9
    assert ece([0, 0], [[1.0, 0.0], [1.0, 0.0]]) == 0.0
    lo, hi = wilson(30, 30)
    assert abs(lo - 0.886) < 0.002 and hi == 1.0, (lo, hi)
    assert abs(wilson(10, 10)[0] - 0.722) < 0.002
    assert auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0 and auroc([0.5, 0.5], [1, 0]) == 0.5
    assert abs(average_precision([0.9, 0.8, 0.7], [1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-9
    bs = paired_bootstrap([1, 1, 1, 0] * 10, [0, 1, 0, 0] * 10, n=200)
    assert bs["lo"] <= bs["diff"] <= bs["hi"]
    # clusters of identical values: the cluster CI must be wider than the clip-level one
    vals = [1] * 10 + [0] * 10 + [1] * 10 + [1] * 10
    cl = ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10
    cb = cluster_bootstrap(vals, cl, n=400)
    clip = cluster_bootstrap(vals, list(range(40)), n=400)
    assert cb["n_clusters"] == 4 and (cb["hi"] - cb["lo"]) > (clip["hi"] - clip["lo"]), (cb, clip)
    print("metrics self-test OK")


if __name__ == "__main__":
    _selftest()
