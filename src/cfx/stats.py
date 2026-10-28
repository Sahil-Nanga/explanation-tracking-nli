import math

import numpy as np


def _clusters(cluster_ids):
    codes, inverse = np.unique(np.asarray(cluster_ids), return_inverse=True)
    return len(codes), inverse


def bootstrap_mean(values, cluster_ids, reps=10000, seed=42, alpha=0.05):
    v = np.asarray(values, dtype=float)
    keep = ~np.isnan(v)
    v, ids = v[keep], np.asarray(cluster_ids)[keep]
    if len(v) == 0:
        return (math.nan, math.nan, math.nan, 0)
    k, inv = _clusters(ids)
    sums = np.bincount(inv, weights=v, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(reps, k))
    num, den = sums[draws].sum(1), counts[draws].sum(1)
    boot = num / den
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return (float(v.mean()), float(lo), float(hi), int(len(v)))


def bootstrap_stat(stat_fn, n_rows, cluster_ids, reps=2000, seed=42, alpha=0.05):
    k, inv = _clusters(cluster_ids)
    members = [np.flatnonzero(inv == c) for c in range(k)]
    rng = np.random.default_rng(seed)
    boot = np.empty(reps)
    for r in range(reps):
        idx = np.concatenate([members[c] for c in rng.integers(0, k, size=k)])
        boot[r] = stat_fn(idx)
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    return (float(stat_fn(np.arange(n_rows))), float(lo), float(hi), int(n_rows))


def bootstrap_diff(a, b, cluster_ids, reps=10000, seed=42, alpha=0.05):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d, ids = a[keep] - b[keep], np.asarray(cluster_ids)[keep]
    if len(d) == 0:
        return {"diff": math.nan, "lo": math.nan, "hi": math.nan, "p": math.nan, "n": 0}
    k, inv = _clusters(ids)
    sums = np.bincount(inv, weights=d, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(reps, k))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    if np.all(d == 0):
        p = 1.0
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(p), "n": int(len(d))}


def mcnemar_exact(a_correct, b_correct):
    a, b = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    idx = [i for i in np.argsort(p) if not np.isnan(p[i])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def cohen_kappa(a, b, labels=None):
    pairs = [(x, y) for x, y in zip(a, b) if x not in (None, "") and y not in (None, "")]
    if not pairs:
        return math.nan
    labels = labels or sorted({x for p in pairs for x in p})
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == l for x, _ in pairs) / n) * (sum(y == l for _, y in pairs) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def macro_f1(gold, pred, labels):
    gold, pred = np.asarray(gold, dtype=object), np.asarray(pred, dtype=object)
    f1s = []
    for l in labels:
        tp = np.sum((pred == l) & (gold == l))
        fp = np.sum((pred == l) & (gold != l))
        fn = np.sum((pred != l) & (gold == l))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def weighted_kappa(a, b, order):
    pos = {v: i for i, v in enumerate(order)}
    pairs = [(pos[x], pos[y]) for x, y in zip(a, b) if x in pos and y in pos]
    if not pairs:
        return math.nan
    k = len(order)
    obs = np.zeros((k, k))
    for x, y in pairs:
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / max(1, k - 1)
    denom = (w * exp).sum()
    return 1.0 if denom == 0 else float(1 - (w * obs).sum() / denom)
