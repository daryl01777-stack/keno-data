#!/usr/bin/env python3
"""Agreement: a shadow ticket of the numbers the weighted signal policies agree on (from R018, Daryl, 25 Oct).

At freeze, each non-random policy (Anomaly composite, Hot, Surge, Pair cluster, Overdue, Persistence)
gives its top 10 rank points, 10 for #1 down to 1 for #10. Each policy's points are multiplied by its
weight in ledger/weights.json, renormalised over those six, and summed per number. The ticket is the
numbers in descending score (ties to the lower number), nested K4-K10 like the other policies. Its
ideal size comes from its own scored rounds (last 10; most cutoff wins, then average net, then the
smaller ticket), K4 until it has any. It sits outside the GRPO group, the weights and the blend;
round.py stores it under "agreement" in frozen.json, so the fingerprint covers it.

usage: python lab/agreement.py R018    score a round's Agreement ticket once its test games are in
                                       draws.csv (after score.py); writes rounds/<R>/agreement_scores.json,
                                       appends ledger/agreement.csv
"""
import csv, json, os, sys

from replay import WINDOW, big, paytable, reward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL = "Agreement"
VOTERS = ("Anomaly composite", "Hot (most drawn)", "Surge (last 20 games)", "Pair cluster",
          "Overdue (longest gap)", "Persistence")


def ideal_size(before=None):
    """(best size, rounds used) from the Agreement's own scored rounds (the last WINDOW before
    `before`); (4, 0) if none."""
    d = os.path.join(ROOT, "rounds")
    ids = sorted(r for r in os.listdir(d) if os.path.exists(os.path.join(d, r, "agreement_scores.json"))
                 and (before is None or r < before))[-WINDOW:]
    if not ids:
        return 4, 0
    tot = {k: [0, 0] for k in range(4, 11)}
    for r in ids:
        for k, x in json.load(open(os.path.join(d, r, "agreement_scores.json")))["sizes"].items():
            tot[int(k)][0] += big(x["replay"])
            tot[int(k)][1] += x["replay"]["net"]
    return max(tot, key=lambda k: (tot[k][0], tot[k][1], -k)), len(ids)


def pick(policies, weights, rid=None):
    """policies: {name: ranked top 10}; weights: ledger/weights.json weights."""
    w = {p: weights[p] for p in VOTERS if p in weights and p in policies}
    total = sum(w.values())
    score = {}
    for p, x in w.items():
        for i, n in enumerate(policies[p][:10]):
            score[n] = score.get(n, 0.0) + (10 - i) * x / total
    ticket = sorted(score, key=lambda n: (-score[n], n))[:10]
    size, rounds = ideal_size(rid)
    return {"policies": {LABEL: ticket}, "ideal_size": size, "ideal_rounds": rounds,
            "weights_used": {p: round(x / total, 6) for p, x in w.items()}}


def score(rid):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from round import load, matrix
    from shadow import catches, replays
    frozen = json.load(open(os.path.join(ROOT, "rounds", rid, "frozen.json")))
    ticket = frozen["agreement"]["policies"][LABEL]
    first, last = frozen["test_games"]
    want = [(first + i) % 1000 for i in range((last - first) % 1000 + 1)]
    games = [g for g in load(os.path.join(ROOT, "draws.csv")) if g["closed"] > frozen["frozen_utc"]]
    test = [next(g for g in games if g["game"] == n) for n in want]
    pay = paytable()
    pg = catches(ticket, matrix(test).astype(int))
    reps = replays(pg, pay, first)
    group = {}
    sp = os.path.join(ROOT, "rounds", rid, "scores.json")
    if os.path.exists(sp):
        from replay import round_replays
        group = {p: reward(v) for p, v in round_replays(json.load(open(sp)), pay).items()}
    out = {
        "round": rid, "policy": LABEL, "ticket": ticket,
        "k10_hits": sum(pg[10]), "k10_hit_rate": round(sum(pg[10]) / (10 * len(test)), 4),
        "reward": reward(reps),
        "sizes": {str(k): {"per_game": pg[k], "replay": reps[k]} for k in pg},
        "compare": {p: group[p] for p in ("Random A", "Random B", "Weighted blend") if p in group},
    }
    with open(os.path.join(ROOT, "rounds", rid, "agreement_scores.json"), "w") as f:
        json.dump(out, f, indent=1)
    led = os.path.join(ROOT, "ledger", "agreement.csv")
    new = not os.path.exists(led)
    with open(led, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["round", "policy", "k10_hits", "k10_hit_rate", "reward",
                        "random_a_reward", "random_b_reward", "big_sizes"])
        w.writerow([rid, LABEL, out["k10_hits"], out["k10_hit_rate"], out["reward"],
                    group.get("Random A", ""), group.get("Random B", ""),
                    " ".join(f"K{k}" for k in reps if big(reps[k]))])
    print(json.dumps({k: v for k, v in out.items() if k != "sizes"}, indent=1))


if __name__ == "__main__":
    score(sys.argv[1])
