#!/usr/bin/env python3
"""Signals ledger: small pre-round patterns that went with a policy doing better or worse (Daryl, 5 Nov).

A signal is (policy, feature, direction): in rounds where the feature was higher, the policy's K4 caught 3+
in more (direction +) or fewer (direction -) of its 40 test games. Each is recorded when found, with its
in-sample correlation, permutation p-value and its correlation in each half of the history, then tracked
forward on rounds scored after it was found. Only the forward correlation can tell a real pattern from luck:
416 features were tried on 5 Nov and about 21 would pass p < 0.05 by luck alone.

usage: python lab/signals.py update        refresh forward results in ledger/signals.csv (run after scoring)
       python lab/signals.py today R0NN    which way each signal leans for a frozen round (for the AI pick)
"""
import csv, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aipick

ROOT = aipick.ROOT
LEDGER = os.path.join(ROOT, "ledger", "signals.csv")
POL = list(aipick.FAMILY)


def features(r):
    """{policy: {feature: value}} for round r, from what was known at its freeze."""
    f = aipick.load(r, "frozen.json"); X = aipick.window(f)
    scan = {k: v["observed"] for k, v in f["window_vs_chance"].items()}
    c100, c20, c10, c3 = X.sum(0), X[-20:].sum(0), X[-10:].sum(0), X[-3:].sum(0)
    gap = np.array([int(X[::-1, i].argmax()) if X[:, i].any() else 100 for i in range(80)])

    def usual(i):
        idx = np.flatnonzero(X[:, i]); return np.diff(idx).mean() if len(idx) > 1 else 50

    def longest(i):
        return np.diff(np.r_[-1, np.flatnonzero(X[:, i]), 100]).max()
    steady = np.array([min(X[q * 25:(q + 1) * 25, i].sum() for q in range(4)) for i in range(80)])
    C, last = X.T @ X, X[-1]
    w = f["weights_at_freeze"]; order = sorted(POL, key=lambda p: -w[p])
    ag = set(f["policies"]["Agreement"][:4]) if "Agreement" in f["policies"] else set()
    top4 = {p: set(v[:4]) for p, v in f["policies"].items() if p in POL}
    out = {}
    for p in POL:
        t = f["policies"][p][:4]; i = np.array(t) - 1
        d = {"draws in 100": c100[i].mean(), "draws in last 20": c20[i].mean(), "draws in last 10": c10[i].mean(),
             "draws in last 3": c3[i].mean(), "current gap": gap[i].mean(),
             "longest gap in window": np.mean([longest(j) for j in i]),
             "due ratio (gap/usual gap)": np.mean([gap[j] / usual(j) for j in i]),
             "steadiness (min per quarter)": steady[i].mean(),
             "pairs together": np.mean([C[a, b] for a, b in itertools.combinations(i, 2)]),
             "all 4 together count": int((X[:, i].sum(1) == 4).sum()), "3+ together count": int((X[:, i].sum(1) >= 3).sum()),
             "in last game": int(last[i].sum()), "odd numbers": sum(n % 2 for n in t), "low half (1-40)": sum(n <= 40 for n in t),
             "ticket sum": sum(t), "ticket spread": max(t) - min(t), "same board row": 4 - len({(n - 1) // 10 for n in t}),
             "same board column": 4 - len({n % 10 for n in t}), "shared with Agreement": len(set(t) & ag),
             "shared with other policies": sum(len(set(t) & o) for q, o in top4.items() if q != p),
             "weight": w[p], "weight rank": order.index(p) + 1}
        d.update({"window " + k: v for k, v in scan.items()})
        out[p] = {k: float(v) for k, v in d.items()}
    return out


def outcome(r, p):
    s = aipick.load(r, "scores.json")
    return sum(h >= 3 for h in s["results"][p]["sizes"]["4"]["per_game"]) if s else None


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() and y.std() else float("nan")


def load():
    return list(csv.DictReader(open(LEDGER))) if os.path.exists(LEDGER) else []


def save(rows):
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)


def update():
    rows = load()
    cache = {}
    for s in rows:
        start = int(s["forward_from"].lstrip("R"))
        x, y = [], []
        r = start
        while aipick.load(r, "scores.json"):
            cache.setdefault(r, features(r))
            x.append(cache[r][s["policy"]][s["feature"]]); y.append(outcome(r, s["policy"]))
            r += 1
        c = corr(x, y)
        s["forward_rounds"] = len(x)
        s["forward_r"] = "" if c != c else f"{c:+.2f}"
        s["forward_holds"] = "" if c != c or len(x) < 10 else ("yes" if np.sign(c) == (1 if s["direction"] == "+" else -1) else "no")
    save(rows)
    return rows


def today(r):
    """For each signal: this round's feature value against its in-sample median, and which way it leans."""
    f = features(r); out = []
    for s in load():
        v = f[s["policy"]][s["feature"]]; med = float(s["median"])
        lean = "neutral" if v == med else ("better" if (v > med) == (s["direction"] == "+") else "worse")
        out.append((s["id"], s["policy"], s["feature"], v, med, lean))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "update":
        for s in update():
            print(f"{s['id']:>3} {s['policy']:24s} {s['feature']:34s} in-sample {s['r']} forward {s['forward_r'] or '-':>6} ({s['forward_rounds']} rds) {s['forward_holds']}")
    elif sys.argv[1] == "today":
        for i, p, feat, v, med, lean in today(int(sys.argv[2].lstrip("R"))):
            print(f"{i:>3} {p:24s} {feat:34s} {v:7.2f} (median {med:.2f}) -> {lean}")
