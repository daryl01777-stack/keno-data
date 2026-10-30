#!/usr/bin/env python3
"""Agreement: the primary ticket from R018 (Daryl, 25 Oct). From R040 the top-half design (pick_top_half);
R018-R039 the consensus design described here (pick_consensus), the numbers the weighted policies agree on.

At freeze, every non-random policy (the 12 rules plus Pair cluster) scores each number in its top 10
by signal strength: -log10 of the chance a random 100-game history gives that policy's raw score or
more (lab/strength.py, table in ledger/strength_null.json), floored at 0. Each policy's strengths are
multiplied by its weight in ledger/weights.json, renormalised over those policies, and summed per
number. (Rank points 10..1 were used for a few minutes before the R018 freeze; Daryl chose strength.) The ticket is the numbers in descending score
(ties to the lower number), nested K4-K10 like the other policies. round.py adds it to the round's
policies; score.py scores it but keeps it out of the GRPO group (it is built from the weights), and
its ideal size comes from ledger/sizes.json like any other ticket, K4 until it has scored rounds.
Per-number scores are not shown on the dashboard (Daryl).

Look-alike balancing (from R019, Daryl, 25 Oct): several policies often pick the same numbers (Hot,
Pairs with last game, Pair cluster, Not in last game...), so together they used to outvote the rest.
ledger/agreement_overlap.json holds each pair's average top-10 overlap over past 100-game windows;
overlap above chance (1.25 of 10) counts as redundancy, and each policy's weight is divided by its
total redundancy (1 for itself plus its excess overlap with every other policy) before renormalising.
The table is built once (python lab/agreement.py overlap) so it doesn't drift between rounds.

usage: python lab/agreement.py overlap [LAST_GAME]   build ledger/agreement_overlap.json from draws.csv
"""
import json, os, sys

LABEL = "Agreement"
CONTROLS = ("Random A", "Random B")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERLAP = os.path.join(ROOT, "ledger", "agreement_overlap.json")
CHANCE = 10 / 80                      # expected share of a top 10 that two unrelated policies share


def build_overlap(last_game=None, step=20):
    """Average top-10 overlap between every pair of policies, over 100-game windows every `step`
    games of draws.csv (up to last_game if given)."""
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import round as rd, shadow
    games = rd.load(os.path.join(ROOT, "draws.csv"))
    if last_game is not None:
        games = games[:max(i for i, g in enumerate(games) if g["game"] == last_game) + 1]
    X = rd.matrix(games)
    names, acc, n = None, None, 0
    for e in range(rd.N_TRAIN, len(X) + 1, step):
        W = X[e - rd.N_TRAIN:e]
        s = rd.stats(W)
        pol = {"Hot (most drawn)": rd.ranking(s["freq"], s["rec"])[:10],
               "Overdue (longest gap)": rd.ranking(s["gap"], -s["freq"])[:10],
               "Surge (last 20 games)": rd.ranking(s["z_rec"], s["freq"])[:10],
               "Pair cluster": rd.pair_cluster(s), "Persistence": rd.persistence(W, s)[:10]}
        r = shadow.rules(W.astype(int))
        pol.update({k: shadow.top10(r[k]) for k in rd.NEW_RULES})
        if names is None:
            names, acc = list(pol), np.zeros((len(pol), len(pol)))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                acc[i, j] += len(set(pol[a]) & set(pol[b])) / 10
        n += 1
    o = acc / n
    out = {"windows": n, "step": step, "last_game": games[-1]["game"], "last_closed_utc": games[-1]["closed"],
           "overlap": {a: {b: round(float(o[i, j]), 4) for j, b in enumerate(names) if b != a}
                       for i, a in enumerate(names)}}
    json.dump(out, open(OVERLAP, "w"), indent=1, sort_keys=True)
    print(f"wrote {OVERLAP}: {n} windows to game {out['last_game']}")
    for a, d in divisors(out["overlap"]).items():
        print(f"  {a:26s} divisor {d:.2f}")


def divisors(overlap):
    """{policy: 1 + sum of its above-chance overlap with every other policy}."""
    return {a: 1 + sum(max(0.0, (x - CHANCE) / (1 - CHANCE)) for x in row.values())
            for a, row in overlap.items()}


def pick(policies, weights, X):
    """The Agreement ticket. From R040 the top-half design; R018-R039 the consensus design below."""
    return pick_top_half(policies, weights, X)


def _consensus(policies, w, r, voters):
    """{number: summed weight x strength} and {number: votes} over `voters`, weights renormalised."""
    from strength import strength
    total = sum(w[p] for p in voters)
    score, votes = {}, {}
    for p in voters:
        for n in policies[p][:10]:
            score[n] = score.get(n, 0.0) + strength(p, float(r[p][n - 1])) * w[p] / total
            votes[n] = votes.get(n, 0) + 1
    return score, votes


def _number_key(n, score, votes, X):
    """Ordering for numbers: higher consensus score, then more votes, more draws in the 100 games, drawn
    more recently, more draws in the last 20, then the last 10; lower number only if all of those tie."""
    import numpy as np
    col = np.asarray(X)[:, n - 1]
    seen = int(col[::-1].argmax()) if col.any() else len(col)
    return (-round(score.get(n, 0.0), 9), -votes.get(n, 0), -int(col.sum()), seen,
            -int(col[-20:].sum()), -int(col[-10:].sum()), n)


def pick_top_half(policies, weights, X):
    """Top-half design (Daryl, 30 Oct, from R040). Only the top half of the 13 policies by weight (13 -> 6,
    rounded down; Random A/B never count) take part. The #1 policy gives its #1 and #2 numbers, the #2 and
    #3 policies their #1 each (a number already on the ticket passes to that policy's next one): that is the
    K4. K5-K10 are filled by the consensus design below run over the top 6 only (look-alike balanced).
    Weight ties go to the tied policy whose #1 number has the most consensus among the tied policies and
    every policy above them; equal consensus scores are broken by _number_key, never arbitrarily."""
    from strength import raw
    r = raw(X)
    pol = [p for p in policies if p not in CONTROLS and p != LABEL]
    missing = [p for p in pol if p not in weights]
    assert not missing, f"no weight for {missing}: recompute ledger/weights.json first"
    wb = {p: weights[p] for p in pol}
    if os.path.exists(OVERLAP):
        d = divisors(json.load(open(OVERLAP))["overlap"])
        wb = {p: x / d.get(p, 1.0) for p, x in wb.items()}
    rank = []
    for v in sorted({weights[p] for p in pol}, reverse=True):
        tied = [p for p in pol if weights[p] == v]
        if len(tied) > 1:
            score, votes = _consensus(policies, wb, r, [p for p in pol if weights[p] >= v])
            tied.sort(key=lambda p: _number_key(policies[p][0], score, votes, X))
        rank += tied
    top = rank[:len(rank) // 2]
    ticket = []
    for p in (top[0], top[0], top[1], top[2]):
        ticket.append(next(n for n in policies[p] if n not in ticket))
    score, votes = _consensus(policies, wb, r, top)
    for n in sorted(score, key=lambda n: _number_key(n, score, votes, X)):
        if len(ticket) == 10:
            break
        if n not in ticket:
            ticket.append(n)
    total = sum(wb[p] for p in top)
    return {"policies": {LABEL: ticket}, "weights_used": {p: round(wb[p] / total, 6) for p in top},
            "top_half": top, "seats": {top[0]: ticket[:2], top[1]: ticket[2:3], top[2]: ticket[3:4]}}


def pick_consensus(policies, weights, X):
    """R018-R039 Agreement. policies: {name: ranked top 10}; weights: ledger/weights.json weights; X: training window."""
    from strength import raw, strength
    r = raw(X)
    w = {p: weights[p] for p in policies if p not in CONTROLS and p != LABEL and p in weights}
    missing = [p for p in policies if p not in CONTROLS and p != LABEL and p not in weights]
    assert not missing, f"no weight for {missing}: recompute ledger/weights.json first"
    if os.path.exists(OVERLAP):           # look-alike balancing, from R019
        d = divisors(json.load(open(OVERLAP))["overlap"])
        w = {p: x / d.get(p, 1.0) for p, x in w.items()}
    total = sum(w.values())
    score = {}
    for p, x in w.items():
        for i, n in enumerate(policies[p][:10]):
            score[n] = score.get(n, 0.0) + strength(p, float(r[p][n - 1])) * x / total
    ticket = sorted(score, key=lambda n: (-score[n], n))[:10]
    return {"policies": {LABEL: ticket}, "weights_used": {p: round(x / total, 6) for p, x in w.items()}}


if __name__ == "__main__":
    if sys.argv[1] == "overlap":
        build_overlap(int(sys.argv[2]) if len(sys.argv) > 2 else None)
