#!/usr/bin/env python3
"""Agreement: the primary ticket from R018 (Daryl, 25 Oct), the numbers the weighted policies agree on.

At freeze, every non-random policy (the 12 rules plus Pair cluster) scores each number in its top 10
by signal strength: -log10 of the chance a random 100-game history gives that policy's raw score or
more (lab/strength.py, table in ledger/strength_null.json), floored at 0. Each policy's strengths are
multiplied by its weight in ledger/weights.json, renormalised over those policies, and summed per
number. (Rank points 10..1 were used for a few minutes before the R018 freeze; Daryl chose strength.) The ticket is the numbers in descending score
(ties to the lower number), nested K4-K10 like the other policies. round.py adds it to the round's
policies; score.py scores it but keeps it out of the GRPO group (it is built from the weights), and
its ideal size comes from ledger/sizes.json like any other ticket, K4 until it has scored rounds.
Per-number scores are not shown on the dashboard (Daryl).
"""
LABEL = "Agreement"
CONTROLS = ("Random A", "Random B")


def pick(policies, weights, X):
    """policies: {name: ranked top 10}; weights: ledger/weights.json weights; X: training window."""
    from strength import raw, strength
    r = raw(X)
    w = {p: weights[p] for p in policies if p not in CONTROLS and p != LABEL and p in weights}
    missing = [p for p in policies if p not in CONTROLS and p != LABEL and p not in weights]
    assert not missing, f"no weight for {missing}: recompute ledger/weights.json first"
    total = sum(w.values())
    score = {}
    for p, x in w.items():
        for i, n in enumerate(policies[p][:10]):
            score[n] = score.get(n, 0.0) + strength(p, float(r[p][n - 1])) * x / total
    ticket = sorted(score, key=lambda n: (-score[n], n))[:10]
    return {"policies": {LABEL: ticket}, "weights_used": {p: round(x / total, 6) for p, x in w.items()}}
