#!/usr/bin/env python3
"""AI pick: a side K4 built at every freeze from R040 (Daryl, 30 Oct), outside the weights and Agreement.

From the round's frozen policies and training window: the pool is every non-random policy's top 5. Each
4-number group from the pool is scored on signal (sum over its numbers of look-alike balanced weight x
strength), togetherness (average times each pair was drawn together in the 100 training games, and the
games where 3 or more of the four landed together) and balance (how many policy families back it: hot,
recent, neighbour, cold). Each measure is standardised over all groups and combined
1 x signal + 0.7 x pair + 0.5 x 3-together + 0.5 x families; the best group is the ticket (ties to the
higher signal, then the lower numbers). Built only from frozen.json, draws.csv and ledger tables, so it can
be rebuilt at any time.

usage: python lab/aipick.py build R0NN     write rounds/R0NN/aipick.json (run at freeze)
       python lab/aipick.py report          replay every scored round that has one
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agreement as ag, replay as rp, round as rd
from strength import raw, strength

ROOT = rp.ROOT
FIRST = 40
FAMILY = {"Hot (most drawn)": "hot", "Pairs with last game": "hot", "Pair cluster": "hot", "Persistence": "hot",
          "Not in last game": "hot", "Surge (last 20 games)": "recent", "Hot last 10": "recent",
          "Hot last 3": "recent", "Rising (2nd half)": "recent", "Just drawn (last game)": "recent",
          "Neighbours of last game": "neighbour", "Cold (100)": "cold", "Overdue (longest gap)": "cold"}


def load(r, name):
    p = os.path.join(ROOT, "rounds", f"R{r:03d}", name)
    return json.load(open(p)) if os.path.exists(p) else None


def window(f):
    a, b = f["training_closed_utc"]
    return rd.matrix([x for x in rd.load(os.path.join(ROOT, "draws.csv")) if a <= x["closed"] <= b]).astype(int)


def ticket(r):
    f = load(r, "frozen.json")
    X = window(f)
    pol = {p: v for p, v in f["policies"].items() if p in FAMILY}
    w = f["weights_at_freeze"]
    d = ag.divisors(json.load(open(ag.OVERLAP))["overlap"])
    wb = {p: w[p] / d.get(p, 1.0) for p in pol}
    T = sum(wb.values())
    R = raw(X)
    cand = {}
    for p in pol:
        for n in pol[p][:5]:
            c = cand.setdefault(n, {"sig": 0.0, "fams": set()})
            c["sig"] += strength(p, float(R[p][n - 1])) * wb[p] / T
            c["fams"].add(FAMILY[p])
    C = X.T @ X
    groups = list(itertools.combinations(sorted(cand), 4))
    sig = np.array([sum(cand[n]["sig"] for n in g) for g in groups])
    pair = np.array([sum(C[a - 1, b - 1] for a, b in itertools.combinations(g, 2)) / 6 for g in groups])
    idx = [np.array(g) - 1 for g in groups]
    tri = np.array([int((X[:, i].sum(1) >= 3).sum()) for i in idx])
    fam = np.array([len(set().union(*[cand[n]["fams"] for n in g])) for g in groups])
    z = lambda a: (a - a.mean()) / a.std() if a.std() > 0 else a * 0
    total = z(sig) + 0.7 * z(pair) + 0.5 * z(tri) + 0.5 * z(fam)
    best = min(range(len(groups)), key=lambda i: (-round(float(total[i]), 9), -sig[i], groups[i]))
    g = groups[best]
    t = sorted(g, key=lambda n: -cand[n]["sig"])
    return {"round": f"R{r:03d}", "ticket": t, "frozen_utc": f["frozen_utc"], "pool": len(cand),
            "pair_avg": round(float(pair[best]), 2), "three_together": int(tri[best]), "families": int(fam[best])}


def result(r, t):
    f, s = load(r, "frozen.json"), load(r, "scores.json")
    if s is None:
        return None
    after = [x for x in rd.load(os.path.join(ROOT, "draws.csv")) if x["closed"] > f["frozen_utc"]]
    i = next(i for i, x in enumerate(after) if x["game"] == s["test_games"][0])
    hits = [len(set(t) & set(x["draw"])) for x in after[i:i + 40]]
    pay = rp.paytable()[4]
    return rp.replay(4, hits, [pay.get(h, 0) for h in hits], s["test_games"][0])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        r = int(sys.argv[2].lstrip("R"))
        d = ticket(r)
        json.dump(d, open(os.path.join(ROOT, "rounds", d["round"], "aipick.json"), "w"), indent=1)
        print(d)
    else:
        n = b = net = 0
        r = FIRST
        while load(r, "scores.json") and load(r, "aipick.json"):
            d = load(r, "aipick.json"); x = result(r, d["ticket"])
            n += 1; b += rp.big(x); net += x["net"]
            print(d["round"], d["ticket"], x["end"], x["net"])
            r += 1
        print(f"{n} rounds, {b} wins, ${net:+d}")
