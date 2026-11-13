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
       python lab/aipick.py ledger          rebuild ledger/aipick.csv, the learning ledger (run after scoring)
       python lab/aipick.py summary         what the ledger adds up to (also written to ledger/aipick_summary.json)
       python lab/aipick.py brief R0NN      pre-pick briefing for a frozen round: summary, last lessons, candidates

The learning ledger (Daryl, 2 Nov; rebuilt 11 Nov) has one row per AI pick, including the holistic side picks of
R056-R060: its kind (free-form R040-R054, build leader R055-R060, holistic from R061), the recipe used, the idea,
size and ticket; how it did (replay result, best catch, near misses = games with 3+ of a K4 or 4+ of a K6, hit
rate = share of its numbers drawn per game against 25% by chance, each number's draws in the 40 test games, the
best number and the one that let it down); where each number came from (policies with it in their top 6 and
how many builds had it in theirs); which policies the signals leaned for and against; the policies that won at
the pick's size and how many numbers they shared; and a lesson. recipe, idea and lesson are written by hand and
kept when the ledger is rebuilt. The summary adds it up by kind, recipe and source so each pick can use what
the earlier ones showed: run `brief` before every pick.
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


def scored(start):
    """Round numbers from start with scores.json and aipick.json, stepping over void rounds (rounds/R0NN/void.json)."""
    r = start
    while load(r, "frozen.json"):
        if not load(r, "void.json"):
            if not (load(r, "scores.json") and load(r, "aipick.json")):
                return
            yield r
        r += 1


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


def hits(r, t):
    """Per-game catches of ticket t over the round's 40 test games, and every number's draw count there."""
    f, s = load(r, "frozen.json"), load(r, "scores.json")
    after = [x for x in rd.load(os.path.join(ROOT, "draws.csv")) if x["closed"] > f["frozen_utc"]]
    i = next(i for i, x in enumerate(after) if x["game"] == s["test_games"][0])
    games = after[i:i + 40]
    return [len(set(t) & set(x["draw"])) for x in games], {n: sum(n in x["draw"] for x in games) for n in t}


LEDGER = os.path.join(ROOT, "ledger", "aipick.csv")
SUMMARY = os.path.join(ROOT, "ledger", "aipick_summary.json")
COLS = ["round", "kind", "recipe", "idea", "size", "ticket", "end", "net", "best_catch", "near_misses", "hit_rate",
        "draws_in_test", "best_number", "letdown", "sources", "signals_for", "signals_against", "winners", "lesson"]
SIDE = "holistic (side)"


def kind(r):
    """Which era a main AI pick belongs to."""
    return "free-form" if r <= 54 else "build leader" if r <= 60 else "holistic"


def picks():
    """Every scored AI pick, oldest first: (round number, kind, the pick's json). Side holistic picks follow their round's main pick."""
    for r in scored(FIRST):
        yield r, kind(r), load(r, "aipick.json")
        h = load(r, "holistic.json")
        if h:
            yield r, SIDE, h


def sources(r, t):
    """{number: (policies with it in their top 6, how many AI builds have it in their top 6)} at the freeze."""
    f, b = load(r, "frozen.json"), load(r, "aibuilds.json")
    out = {}
    for n in t:
        pol = [p for p in FAMILY if p in f["policies"] and n in f["policies"][p][:6]]
        out[n] = (pol, sum(n in v[:6] for v in b["builds"].values()) if b else None)
    return out


def short(p):
    return p.split(" (")[0]


def ledger():
    """Rebuild the learning ledger. recipe, idea and lesson are hand-written and kept (keyed by round and kind)."""
    import csv
    old = {}
    if os.path.exists(LEDGER):
        for row in csv.DictReader(open(LEDGER)):
            old[(row["round"], row.get("kind") or kind(int(row["round"][1:])))] = row
    rows = []
    for r, kd, d in picks():
        f, s = load(r, "frozen.json"), load(r, "scores.json")
        t = d["ticket"]; k = len(t); x = result(r, t); h, cnt = hits(r, t); src = sources(r, t)
        win = []
        for p, v in s["results"].items():          # winners at the pick's own size (K4, or K6 from R064)
            if p in f["policies"] and v["sizes"][str(k)]["replay"]["end"].startswith("big"):
                win.append(f"{p} {' '.join(map(str, f['policies'][p][:k]))} (shares {len(set(t) & set(f['policies'][p][:k]))})")
        rid = f"R{r:03d}"; o = old.get((rid, kd), {}); sg = d.get("signals") or {}
        hi, lo = max(t, key=lambda n: cnt[n]), min(t, key=lambda n: cnt[n])
        rows.append({
            "round": rid, "kind": kd, "recipe": o.get("recipe") or d.get("recipe", ""), "idea": o.get("idea") or d.get("idea", ""),
            "size": k, "ticket": " ".join(map(str, t)), "end": x["end"], "net": x["net"], "best_catch": max(h),
            "near_misses": sum(c >= (3 if k == 4 else k - 2) for c in h),        # 3+ of a K4, 4+ of a K6
            "hit_rate": f"{100 * sum(cnt.values()) / (40 * k):.1f}",              # % of the ticket's numbers drawn per game; chance 25
            "draws_in_test": " ".join(f"{n}:{c}" for n, c in cnt.items()),
            "best_number": f"{hi} ({cnt[hi]} draws)", "letdown": f"{lo} ({cnt[lo]} draws)",
            "sources": "; ".join(f"{n}={'+'.join(short(p) for p in src[n][0]) or 'none'}" + (f" [{src[n][1]} builds]" if src[n][1] is not None else "") for n in t),
            "signals_for": ", ".join(sg.get("favour", [])), "signals_against": ", ".join(sg.get("against", [])),
            "winners": "; ".join(win), "lesson": o.get("lesson", "")})
    with open(LEDGER, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS); w.writeheader(); w.writerows(rows)
    summary(rows)
    return rows


def summary(rows=None):
    """What the ledger adds up to: record by kind and by recipe, and how the numbers did by where they came from.
    Written to ledger/aipick_summary.json; chance is 10 draws per number in 40 games (25%)."""
    import csv
    rows = rows or list(csv.DictReader(open(LEDGER)))

    def group(key):
        g = {}
        for x in rows:
            a = g.setdefault(x[key] or "(none)", {"picks": 0, "wins": 0, "net": 0, "hit": 0.0, "near": 0})
            a["picks"] += 1; a["wins"] += x["end"].startswith("big"); a["net"] += int(x["net"]); a["hit"] += float(x["hit_rate"]); a["near"] += int(x["near_misses"])
        return {k: {"picks": a["picks"], "wins": a["wins"], "net": a["net"], "hit_rate": round(a["hit"] / a["picks"], 2),
                    "near_misses_per_pick": round(a["near"] / a["picks"], 2)} for k, a in g.items()}
    pol, nb, npol, sig = {}, {}, {}, {}
    for r, kd, d in picks():
        t = d["ticket"]; _, cnt = hits(r, t); src = sources(r, t); fav = set((d.get("signals") or {}).get("favour", []))
        ag = set((d.get("signals") or {}).get("against", []))
        for n in t:
            ps, b = src[n]
            for p in ps:
                pol.setdefault(short(p), []).append(cnt[n])
            npol.setdefault("0" if not ps else "1" if len(ps) == 1 else "2" if len(ps) == 2 else "3+", []).append(cnt[n])
            if b is not None:
                nb.setdefault("0-2" if b <= 2 else "3-6" if b <= 6 else "7-12" if b <= 12 else "13+", []).append(cnt[n])
            if "signals" in d:
                names = {short(p) for p in ps}
                key = "backed by a favoured policy" if names & fav else "backed by an against policy" if names & ag else "no signal either way"
                sig.setdefault(key, []).append(cnt[n])
    avg = lambda m: {k: {"numbers": len(v), "avg_draws": round(sum(v) / len(v), 2), "hit_rate": round(2.5 * sum(v) / len(v), 2)} for k, v in sorted(m.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))}
    out = {"picks": len(rows), "by_kind": group("kind"), "by_recipe": group("recipe"), "by_size": group("size"),
           "numbers_by_source_policy": avg(pol), "numbers_by_policies_backing": avg(npol), "numbers_by_builds_backing": avg(nb),
           "numbers_by_signal_lean": avg(sig),
           "note": "Chance is 10 draws per number in 40 games (hit rate 25%). Small counts swing a lot; treat any line under about 40 numbers as a hint only."}
    json.dump(out, open(SUMMARY, "w"), indent=1)
    return out


def show_summary(o):
    print(f"{o['picks']} picks in the ledger. Chance: hit rate 25%, 10 draws per number.")
    for title, key in (("By kind", "by_kind"), ("By recipe", "by_recipe"), ("By size", "by_size")):
        print(title)
        for k, a in o[key].items():
            print(f"  {str(k):34s} {a['picks']:2d} picks  {a['wins']} wins  ${a['net']:+5d}  hit rate {a['hit_rate']:.1f}%  near misses {a['near_misses_per_pick']:.1f} a pick")
    for title, key in (("Numbers by source policy (in its top 6 at the freeze)", "numbers_by_source_policy"),
                       ("Numbers by how many policies backed them", "numbers_by_policies_backing"),
                       ("Numbers by how many builds backed them", "numbers_by_builds_backing"),
                       ("Numbers by signal lean of their policies", "numbers_by_signal_lean")):
        print(title)
        for k, a in o[key].items():
            print(f"  {k:34s} {a['numbers']:3d} numbers  {a['avg_draws']:5.2f} draws each  ({a['hit_rate']:.1f}%)")


def brief(r):
    """Pre-pick briefing for a frozen round: the ledger's summary, the last lessons, and every candidate number
    with who backs it (policies with it in their top 6, builds with it in their top 6, signal leans of those policies)."""
    import csv, signals
    o = summary(); show_summary(o)
    rows = list(csv.DictReader(open(LEDGER)))
    print("Last lessons")
    for x in rows[-4:]:
        print(f"  {x['round']} {x['kind']} [{x['recipe']}] {x['ticket']} -> {x['end']} {x['net']}: {x['lesson']}")
    f, b = load(r, "frozen.json"), load(r, "aibuilds.json")
    lean = {}
    for _, p, _, _, _, l in signals.today(r):
        lean[p] = lean.get(p, 0) + {"better": 1, "worse": -1}.get(l, 0)
    lift = {k: a["avg_draws"] - 10 for k, a in o["numbers_by_source_policy"].items()}
    X = window(f); c100, c20 = X.sum(0), X[-20:].sum(0); cand = []
    for n in range(1, 81):
        ps = [p for p in FAMILY if p in f["policies"] and n in f["policies"][p][:6]]
        nb = sum(n in v[:6] for v in b["builds"].values()) if b else 0
        if ps or nb >= 3:
            cand.append((n, ps, nb, sum(lean.get(p, 0) for p in ps), sum(lift.get(short(p), 0) for p in ps)))
    for k, c in ((b or {}).get("confidence") or {}).items():
        print(f"{k} build's confidence in its ticket {b['builds'][k][:6]}: {c['confidence']} (forecast {c['lift']:+.1%} against a typical ticket; "
              f"{c['views']} of {len(c['trees'])} copies found something{'; treat its numbers as a random six' if c['confidence'] == 'none' else ''})")
    print(f"Signal lean by policy: " + ", ".join(f"{short(p)} {v:+d}" for p, v in sorted(lean.items(), key=lambda kv: -kv[1]) if v))
    print(f"Candidates for R{r:03d} (number: policies | builds | signal lean | ledger lift | draws in 100 / last 20)")
    for n, ps, nb, sl, ll in sorted(cand, key=lambda c: -(2 * len(c[1]) + c[2] / 3 + c[3]))[:24]:
        print(f"  {n:2d}: {len(ps)} {'+'.join(short(p) for p in ps):60s} | {nb:2d} | {sl:+d} | {ll:+.1f} | {int(c100[n - 1])}/{int(c20[n - 1])}")


def result(r, t):
    f, s = load(r, "frozen.json"), load(r, "scores.json")
    if s is None:
        return None
    after = [x for x in rd.load(os.path.join(ROOT, "draws.csv")) if x["closed"] > f["frozen_utc"]]
    i = next(i for i, x in enumerate(after) if x["game"] == s["test_games"][0])
    hits = [len(set(t) & set(x["draw"])) for x in after[i:i + 40]]
    k = len(t)                                     # K4 to R063; K6 from R064 (Daryl, 11 Nov)
    pay = rp.paytable()[k]
    return rp.replay(k, hits, [pay.get(h, 0) for h in hits], s["test_games"][0])


if __name__ == "__main__":
    if sys.argv[1] == "build":
        r = int(sys.argv[2].lstrip("R"))
        d = ticket(r)
        json.dump(d, open(os.path.join(ROOT, "rounds", d["round"], "aipick.json"), "w"), indent=1)
        print(d)
    elif sys.argv[1] == "ledger":
        for row in ledger():
            print(row["round"], row["kind"], row["ticket"], row["end"], row["net"], row["best_catch"], row["near_misses"], row["hit_rate"])
    elif sys.argv[1] == "summary":
        show_summary(summary())
    elif sys.argv[1] == "brief":
        brief(int(sys.argv[2].lstrip("R")))
    else:
        n = b = net = 0
        for r in scored(FIRST):
            d = load(r, "aipick.json"); x = result(r, d["ticket"])
            n += 1; b += rp.big(x); net += x["net"]
            print(d["round"], d["ticket"], x["end"], x["net"])
        print(f"{n} rounds, {b} wins, ${net:+d}")
