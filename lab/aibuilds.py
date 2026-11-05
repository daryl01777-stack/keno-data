#!/usr/bin/env python3
"""AI builds: the AI pick ideas turned into fixed rules, run side by side as their own GRPO group (Daryl, 5 Nov).

Each build reads only what a freeze has: the round's frozen policies (top 10s), the 100 training games and the
scores of earlier rounds. It returns a ranked top 10, nested as K4-K10 like a policy. All builds are scored
with the replay rule and weighted with the points rule (start 100, +5 a round with a cutoff win, +1 if no win
but a ticket timed out, -1 if every ticket busted). They are outside the main weights, Agreement and the
locked rules. Live from R055; the hindsight replay over R018-R054 is information only.

usage: python lab/aibuilds.py build R0NN      write rounds/R0NN/aibuilds.json (run at freeze)
       python lab/aibuilds.py score R0NN      score it and update ledger/aibuilds.json + ledger/aibuilds.csv
       python lab/aibuilds.py hindsight A B   replay every build over rounds A-B (no files written)
       python lab/aibuilds.py init A B        start ledger/aibuilds.json with points carried from rounds A-B

Builds added later (Last-3 skip and Signal board from R056, built on the 5 Nov signals) join on the group's
median points, because their hindsight replay would be in-sample.

Starting points (Daryl, 5 Nov, as for the main points weights): every build starts on 100 at R018 and carries
its R018-R054 hindsight points into R055. The top-weighted build's K4 is the AI pick from R055 (ties to more
rounds won, then more K4 wins, then the build listed first).
"""
import csv, itertools, json, os, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aipick, replay as rp, round as rd

ROOT = rp.ROOT
LIVE = 55
LEDGER = os.path.join(ROOT, "ledger", "aibuilds.json")
BOARD = os.path.join(ROOT, "ledger", "aibuilds.csv")
START, WIN, SURVIVE, BUST = 100, 5, 1, -1
SIGNAL = [p for p in aipick.FAMILY]           # the 13 non-random policies
COLD = ("Cold (100)",)


class Ctx:
    """What a build can see at a freeze."""
    def __init__(self, r):
        self.r = r
        self.f = aipick.load(r, "frozen.json")
        self.X = aipick.window(self.f)                       # 100 x 80, oldest first
        self.pol = {p: v for p, v in self.f["policies"].items() if p in SIGNAL}
        X = self.X
        self.c100 = X.sum(0); self.c20 = X[-20:].sum(0); self.c10 = X[-10:].sum(0); self.c3 = X[-3:].sum(0)
        self.h1 = X[:50].sum(0); self.h2 = X[50:].sum(0)
        self.last = set(np.flatnonzero(X[-1]) + 1)
        self.gap = np.array([int(X[::-1, i].argmax()) if X[:, i].any() else 100 for i in range(80)])
        self.top6 = Counter(n for v in self.pol.values() for n in v[:6])
        self.C = X.T @ X

    def c(self, a, n): return int(getattr(self, a)[n - 1])

    def mean_gap(self, n):
        idx = np.flatnonzero(self.X[:, n - 1])
        return float(np.diff(idx).mean()) if len(idx) > 1 else 50.0


def z(v):
    v = np.asarray(v, float); s = v.std()
    return (v - v.mean()) / s if s else v * 0


def rank(score, ctx, ties=("c20", "c100")):
    """Numbers 1-80 by score, ties to the listed counts, then the lower number; top 10."""
    key = lambda n: (-score[n - 1],) + tuple(-ctx.c(t, n) for t in ties) + (n,)
    return sorted(range(1, 81), key=key)[:10]


def fill(head, ctx, score=None):
    """Keep head's order, then fill to 10 by score (default: c100 + c20 z-mix)."""
    score = score if score is not None else z(ctx.c100) + z(ctx.c20)
    out = list(dict.fromkeys(head))
    for n in rank(score, ctx) + sorted(range(1, 81), key=lambda n: -score[n - 1]):
        if n not in out:
            out.append(n)
        if len(out) == 10:
            break
    return out[:10]


def votes(ctx):
    return np.array([ctx.top6[n] for n in range(1, 81)], float)


# ---- the builds (one per AI pick idea, R040-R054) ----

def momentum(ctx):                     # R040: heavy over 100 games and hot in the last 20
    return rank(z(ctx.c100) + z(ctx.c20), ctx)


def cluster(ctx):                      # R041: the most-backed number plus the numbers that land with it
    a = rank(votes(ctx), ctx)[0]
    return fill([a] + rank(ctx.C[a - 1] - 1000 * (np.arange(1, 81) == a), ctx), ctx)


def streakers(ctx):                    # R042: the hottest short runs
    return rank(ctx.c10.astype(float), ctx, ("c20", "c100"))


def all_rounders(ctx):                 # R043: in the most policies' top 6, steadiest first
    steady = np.array([min(ctx.X[q * 25:(q + 1) * 25, n].sum() for q in range(4)) for n in range(80)])
    return rank(votes(ctx) + steady / 100, ctx, ("c100",))


def proven_quartet(ctx):               # R044: top-drawn numbers that have landed as a full set
    top = [n for n in rank(ctx.c100.astype(float), ctx, ("c20",)) + sorted(range(1, 81), key=lambda n: -ctx.c100[n - 1])][:20]
    top = list(dict.fromkeys(top))[:20]
    rows = [set(np.flatnonzero(x) + 1) & set(top) for x in ctx.X]
    q = Counter(t for s in rows for t in itertools.combinations(sorted(s), 4))
    best = max(q, key=lambda t: (q[t], sum(ctx.c100[n - 1] for n in t))) if q else tuple(top[:4])
    return fill(sorted(best, key=lambda n: -ctx.c100[n - 1]), ctx, ctx.c100.astype(float))


def rising_tide(ctx):                  # R045: biggest climb from the first 50 games to the last 50
    return rank((ctx.h2 - ctx.h1).astype(float), ctx)


def cold_awakening(ctx):               # R046: cold over 100 games but woken up in the last 20
    cold = ctx.c100 <= np.median(ctx.c100)
    return rank(np.where(cold, ctx.c20 - ctx.c100 / 5, -99), ctx)


def spring_loaded(ctx):                # R047: two in form plus two regulars gone quiet
    form = rank(votes(ctx) + z(ctx.c20), ctx)
    due = np.array([ctx.gap[n - 1] / ctx.mean_gap(n) if ctx.c100[n - 1] >= np.median(ctx.c100) else -1 for n in range(1, 81)])
    rest = [n for n in rank(due, ctx, ("c100",)) if n not in form[:2]]
    return fill([form[0], form[1], rest[0], rest[1], form[2], rest[2], form[3], rest[3]], ctx)


def fresh_press(ctx):                  # R048: heavyweights that were all in the last game
    s = z(ctx.c100) + z(ctx.c20)
    return fill(rank(np.array([s[n - 1] if n in ctx.last else -99 for n in range(1, 81)]), ctx), ctx)


def short_clock(ctx):                  # R049: the hottest number now plus its partners in the last 40 games
    a = rank(ctx.c20.astype(float), ctx, ("c10", "c100"))[0]
    C40 = ctx.X[-40:].T @ ctx.X[-40:]
    return fill([a] + rank(C40[a - 1] - 1000 * (np.arange(1, 81) == a), ctx), ctx)


def other_pond(ctx):                   # R050: overdue regulars and cold numbers, alternating
    reg = ctx.c100 >= np.median(ctx.c100)
    over = rank(np.where(reg, ctx.gap, -1).astype(float), ctx, ("c100",))
    cold = rank(-ctx.c100.astype(float), ctx, ("c20",))
    out = []
    for a, b in zip(over, cold):
        out += [a, b]
    return list(dict.fromkeys(out))[:10] if len(set(out)) >= 10 else fill(out, ctx)


def engine_room(ctx):                  # R051: the most-backed numbers on both clocks
    return rank(votes(ctx) + z(ctx.c100) + z(ctx.c20), ctx)


def near_miss(ctx):                    # R052: #1s of the policies with the best 3/4 rates over the last 20 rounds
    rate = Counter(); n = Counter()
    for q in range(max(18, ctx.r - 20), ctx.r):
        s = aipick.load(q, "scores.json")
        if not s:
            continue
        for p, v in s["results"].items():
            if p in SIGNAL:
                g = v["sizes"]["4"]["per_game"]; rate[p] += sum(h >= 3 for h in g); n[p] += len(g)
    order = sorted(ctx.pol, key=lambda p: -(rate[p] / n[p] if n[p] else 0))
    out = []
    for depth in range(10):
        for p in order[:4]:
            out.append(ctx.pol[p][depth])
    return list(dict.fromkeys(out))[:10]


def barbell(ctx):                      # R053: strongest hot numbers plus Cold's numbers, alternating
    hot = rank(votes(ctx) + z(ctx.c100), ctx)
    cold = [n for n in ctx.pol["Cold (100)"] if n not in hot[:5]]
    out = []
    for a, b in zip(hot, cold):
        out += [a, b]
    return fill(out, ctx)


def head_count(ctx):                   # R054: straight count of policies' top 6
    return rank(votes(ctx), ctx, ("c10", "c20", "c100"))


# ---- builds suggested on 5 Nov, never played as an AI pick ----

def rhythm(ctx):                       # due now by each number's own usual gap
    due = np.array([ctx.gap[n - 1] / ctx.mean_gap(n) if ctx.c100[n - 1] >= 15 else -1 for n in range(1, 81)])
    return rank(due, ctx, ("c100",))


def board_shape(ctx):                  # the hottest row or column of the 8 x 10 board, last 20 games
    lines = [list(range(10 * i + 1, 10 * i + 11)) for i in range(8)] + [list(range(j, 81, 10)) for j in range(1, 11)]
    heat = lambda L: sum(ctx.c20[n - 1] for n in L) / len(L)
    best = max(lines, key=heat)
    return fill(sorted(best, key=lambda n: (-ctx.c20[n - 1], -ctx.c100[n - 1], n)), ctx)


def winner_echo(ctx):                  # the numbers on last round's winning tickets
    s, f = aipick.load(ctx.r - 1, "scores.json"), aipick.load(ctx.r - 1, "frozen.json")
    tally = Counter()
    if s and f:
        for p, v in s["results"].items():
            if p not in f["policies"]:
                continue
            for k, x in v["sizes"].items():
                if x["replay"]["end"].startswith("big"):
                    tally.update(f["policies"][p][:int(k)])
    return fill(rank(np.array([tally[n] for n in range(1, 81)], float), ctx)[:max(1, min(10, len(tally)))], ctx)


def odd_one_out(ctx):                  # strong numbers that only one policy backs
    s = z(ctx.c100) + z(ctx.c20)
    return rank(np.where(votes(ctx) == 1, s, -99), ctx)


def pair_only(ctx):                    # the pair drawn together most, then its best partners
    C = ctx.C.astype(float).copy(); np.fill_diagonal(C, -1)
    a, b = np.unravel_index(int(C.argmax()), C.shape)
    both = C[a] + C[b]; both[[a, b]] = -1
    return fill([a + 1, b + 1] + rank(both, ctx), ctx)


def track_record(ctx):                 # each policy's top 4, weighted by its 3/4 rate over the last 20 rounds
    rate = Counter(); n = Counter()
    for q in range(max(18, ctx.r - 20), ctx.r):
        s = aipick.load(q, "scores.json")
        for p, v in (s or {"results": {}})["results"].items():
            if p in SIGNAL:
                g = v["sizes"]["4"]["per_game"]; rate[p] += sum(h >= 3 for h in g); n[p] += len(g)
    sc = np.zeros(80)
    for p, v in ctx.pol.items():
        for i, x in enumerate(v[:4]):
            sc[x - 1] += (rate[p] / n[p] if n[p] else 0) * (4 - i)
    return rank(sc, ctx)


# ---- builds from the signals ledger (5 Nov) ----

def last3_skip(ctx):                   # signal 1-3: Hot last 3 did better when its numbers were not in the last game
    return rank(np.array([ctx.c3[n - 1] if n not in ctx.last else -1 for n in range(1, 81)], float), ctx, ("c10", "c20", "c100"))


def signal_board(ctx):                 # policies ordered by how many recorded signals lean their way this round
    import signals
    net = {p: 0 for p in ctx.pol}
    for _, p, _, _, _, lean in signals.today(ctx.r):
        net[p] += {"better": 1, "worse": -1}.get(lean, 0)
    order = sorted(ctx.pol, key=lambda p: (-net[p], -ctx.f["weights_at_freeze"].get(p, 0)))
    out = []
    for depth in range(10):
        for p in order[:4]:
            out.append(ctx.pol[p][depth])
    return list(dict.fromkeys(out))[:10]


BUILDS = {"Momentum": momentum, "Cluster": cluster, "Streakers": streakers, "All-rounders": all_rounders,
          "Proven quartet": proven_quartet, "Rising tide": rising_tide, "Cold awakening": cold_awakening,
          "Spring-loaded": spring_loaded, "Fresh off the press": fresh_press, "Short-clock crew": short_clock,
          "The other pond": other_pond, "Engine room": engine_room, "Near-miss leaders": near_miss,
          "Barbell": barbell, "Head count": head_count, "Rhythm": rhythm, "Board shape": board_shape,
          "Winner echo": winner_echo, "Odd one out": odd_one_out, "Pair-only": pair_only, "Track record": track_record,
          "Last-3 skip": last3_skip, "Signal board": signal_board}
LATE = {"Last-3 skip": "R056", "Signal board": "R056"}   # built from 5 Nov findings: no hindsight points (they'd be in-sample)


def build(r):
    ctx = Ctx(r)
    out = {}
    for name, fn in BUILDS.items():
        t = [int(n) for n in fn(ctx)]
        assert len(t) == 10 and len(set(t)) == 10, (name, t)
        out[name] = t
    return out


# ---- scoring ----

def test_games(r):
    f, s = aipick.load(r, "frozen.json"), aipick.load(r, "scores.json")
    after = [x for x in rd.load(os.path.join(ROOT, "draws.csv")) if x["closed"] > f["frozen_utc"]]
    i = next(i for i, x in enumerate(after) if x["game"] == s["test_games"][0])
    return [set(x["draw"]) for x in after[i:i + 40]], s["test_games"][0]


def score_ticket(top10, games, first):
    pay = rp.paytable()
    res = {}
    for k in range(4, 11):
        h = [len(set(top10[:k]) & g) for g in games]
        res[k] = rp.replay(k, h, [pay[k].get(x, 0) for x in h], first)
    return res


def outcome(res):
    if any(rp.big(x) for x in res.values()):
        return "win", WIN
    if any(x["end"].startswith("timed out") for x in res.values()):
        return "survive", SURVIVE
    return "bust", BUST


def score(r, tickets=None):
    tickets = tickets or aipick.load(r, "aibuilds.json")["builds"]
    games, first = test_games(r)
    return {name: score_ticket(t, games, first) for name, t in tickets.items()}


def update(r):
    led = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"start_round": f"R{LIVE:03d}", "points": {b: START for b in BUILDS}, "rounds": []}
    rid = f"R{r:03d}"
    assert rid not in led["rounds"], f"{rid} already scored"
    res = score(r)
    rows = []
    for name, x in res.items():
        o, pts = outcome(x)
        led["points"][name] = led["points"].get(name, START) + pts
        t = led.setdefault("tally", {}).setdefault(name, {})
        t[o] = t.get(o, 0) + 1; t["k4"] = t.get("k4", 0) + rp.big(x[4])
        rows.append({"round": rid, "build": name, "outcome": o, "points": led["points"][name],
                     "k4": x[4]["end"], "k4_net": x[4]["net"],
                     "wins": " ".join(f"K{k}" for k, v in x.items() if rp.big(v))})
    led["rounds"].append(rid)
    T = sum(led["points"].values())
    led["weights"] = {b: round(p / T, 6) for b, p in led["points"].items()}
    json.dump(led, open(LEDGER, "w"), indent=1, sort_keys=True)
    new = not os.path.exists(BOARD)
    with open(BOARD, "a", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); new and w.writeheader(); w.writerows(rows)
    return rows, led


def ensure(led):
    """Builds added after the start join on the group's median points, since their hindsight would be in-sample."""
    med = int(np.median([v for k, v in led["points"].items() if k not in LATE])) if led["points"] else START
    for b in BUILDS:
        if b not in led["points"]:
            led["points"][b] = med; led.setdefault("joined", {})[b] = {"round": LATE.get(b, ""), "points": med}
    T = sum(led["points"].values()); led["weights"] = {b: round(p / T, 6) for b, p in led["points"].items()}
    return led


def leader(led):
    t = led.get("tally", {})
    return max(BUILDS, key=lambda b: (led["points"][b], t.get(b, {}).get("win", 0), t.get(b, {}).get("k4", 0), -list(BUILDS).index(b)))


def hindsight(a, b):
    pts = {n: START for n in BUILDS}; tally = {n: Counter() for n in BUILDS}
    for r in range(a, b + 1):
        if not aipick.load(r, "scores.json"):
            continue
        res = score(r, build(r))
        for name, x in res.items():
            o, p = outcome(x); pts[name] += p; tally[name][o] += 1
            tally[name]["k4"] += rp.big(x[4]); tally[name]["k4net"] += x[4]["net"]
    return pts, tally


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        r = int(sys.argv[2].lstrip("R"))
        d = {"round": f"R{r:03d}", "frozen_utc": aipick.load(r, "frozen.json")["frozen_utc"], "builds": build(r)}
        json.dump(d, open(os.path.join(ROOT, "rounds", d["round"], "aibuilds.json"), "w"), indent=1)
        for k, v in d["builds"].items():
            print(f"{k:20s} {v[:4]}")
    elif cmd == "join":
        led = ensure(json.load(open(LEDGER))); json.dump(led, open(LEDGER, "w"), indent=1, sort_keys=True); print(led.get("joined"))
    elif cmd == "score":
        rows, led = update(int(sys.argv[2].lstrip("R")))
        for x in sorted(rows, key=lambda x: -x["points"]):
            print(f"{x['build']:20s} {x['outcome']:8s} {x['points']:4d}  K4 {x['k4']:16s} {x['k4_net']:+d}  {x['wins']}")
    elif cmd == "init":
        a, b = int(sys.argv[2].lstrip("R")), int(sys.argv[3].lstrip("R"))
        pts, tally = hindsight(a, b)
        T = sum(pts.values())
        led = {"start_round": f"R{a:03d}", "carried": f"R{a:03d}-R{b:03d} hindsight", "points": pts,
               "tally": {n: {k: int(v) for k, v in t.items() if k != "k4net"} for n, t in tally.items()},
               "rounds": [f"R{r:03d}" for r in range(a, b + 1) if aipick.load(r, "scores.json")],
               "weights": {n: round(p / T, 6) for n, p in pts.items()}}
        json.dump(led, open(LEDGER, "w"), indent=1, sort_keys=True)
        print("leader", leader(led))
    elif cmd == "hindsight":
        pts, tally = hindsight(int(sys.argv[2].lstrip("R")), int(sys.argv[3].lstrip("R")))
        for n in sorted(pts, key=lambda n: -pts[n]):
            t = tally[n]
            print(f"{n:20s} pts {pts[n]:4d}  rounds won {t['win']:2d} survived {t['survive']:2d} bust {t['bust']:2d}  K4 wins {t['k4']:2d} K4 net {t['k4net']:+5d}")
