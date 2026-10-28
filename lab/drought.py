#!/usr/bin/env python3
"""Drought ticket: a side K4 tracked from R032 to the R037 review (Daryl, 28 Oct). Outside the weights,
the GRPO group and Agreement, and not on the dashboard.

At each freeze, take the 4 non-random policies that have gone longest without a K4 cutoff win (counting
from R018; never since R018 counts as R017; ties to the higher weight at freeze, then name) and use each
one's best number not already on the ticket. Built only from frozen.json and earlier rounds' scores, so it
can be rebuilt at any time. R018-R031 were a hindsight replay (3 wins, +$124); R032 on is the forward test.

usage: python lab/drought.py build R0NN     write rounds/R0NN/drought.json (run at freeze)
       python lab/drought.py report          replay every round from R018 that has scores
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay as rp

ROOT = rp.ROOT
FIRST, FORWARD = 18, 32


def load(r, name):
    p = os.path.join(ROOT, "rounds", f"R{r:03d}", name)
    return json.load(open(p)) if os.path.exists(p) else None


def ticket(r):
    f = load(r, "frozen.json")
    last = {}
    for q in range(FIRST, r):
        for p, v in rp.round_replays(load(q, "scores.json")).items():
            if rp.big(v[4]):
                last[p] = q
    w = f["weights_at_freeze"]
    pol = {p: v for p, v in f["policies"].items() if p != "Agreement" and not p.startswith("Random")}
    pick = sorted(pol, key=lambda p: (last.get(p, FIRST - 1), -w[p], p))[:4]
    t = []
    for p in pick:
        t.append(next(n for n in pol[p] if n not in t))
    return {"round": f"R{r:03d}", "policies": pick, "ticket": t, "frozen_utc": f["frozen_utc"]}


def result(r, t):
    s = load(r, "scores.json")
    if s is None:
        return None
    import round as rd
    draws = {x["game"]: set(x["draw"]) for x in rd.load(os.path.join(ROOT, "draws.csv")) if x["closed"] > load(r, "frozen.json")["frozen_utc"]}
    first = s["test_games"][0]
    games = [(first + i) % 1000 for i in range(40)]
    hits = [len(set(t) & draws[g]) for g in games]
    pay = rp.paytable()[4]
    return rp.replay(4, hits, [pay.get(h, 0) for h in hits], first)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        r = int(sys.argv[2].lstrip("R"))
        d = ticket(r)
        json.dump(d, open(os.path.join(ROOT, "rounds", d["round"], "drought.json"), "w"), indent=1)
        print(d)
    else:
        tot = {True: [0, 0, 0], False: [0, 0, 0]}
        r = FIRST
        while load(r, "scores.json"):
            d = load(r, "drought.json") or ticket(r)
            x = result(r, d["ticket"])
            fw = r >= FORWARD
            tot[fw][0] += 1; tot[fw][1] += rp.big(x); tot[fw][2] += x["net"]
            print(d["round"], d["ticket"], x["end"], x["net"], "" if fw else "(hindsight)")
            r += 1
        for fw, (n, b, net) in tot.items():
            print(("forward" if fw else "hindsight"), f"{n} rounds, {b} wins, ${net:+d}")
