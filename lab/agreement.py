#!/usr/bin/env python3
"""Agreement: the primary ticket from R018 (Daryl, 25 Oct), the numbers the weighted policies agree on.

At freeze, every non-random policy (the 12 rules plus Pair cluster) gives its top 10 rank points,
10 for #1 down to 1 for #10. Each policy's points are multiplied by its weight in ledger/weights.json,
renormalised over those policies, and summed per number. The ticket is the numbers in descending score
(ties to the lower number), nested K4-K10 like the other policies. round.py adds it to the round's
policies; score.py scores it but keeps it out of the GRPO group (it is built from the weights), and
its ideal size comes from ledger/sizes.json like any other ticket, K4 until it has scored rounds.
Agreement only, no strength term; per-number scores are not shown on the dashboard (Daryl).
"""
LABEL = "Agreement"
CONTROLS = ("Random A", "Random B")


def pick(policies, weights, rid=None):
    """policies: {name: ranked top 10}; weights: ledger/weights.json weights."""
    w = {p: weights[p] for p in policies if p not in CONTROLS and p != LABEL and p in weights}
    missing = [p for p in policies if p not in CONTROLS and p != LABEL and p not in weights]
    assert not missing, f"no weight for {missing}: recompute ledger/weights.json first"
    total = sum(w.values())
    score = {}
    for p, x in w.items():
        for i, n in enumerate(policies[p][:10]):
            score[n] = score.get(n, 0.0) + (10 - i) * x / total
    ticket = sorted(score, key=lambda n: (-score[n], n))[:10]
    return {"policies": {LABEL: ticket}, "weights_used": {p: round(x / total, 6) for p, x in w.items()}}
