#!/usr/bin/env python3
"""Keno round builder: anomaly scan on a 100-game training window, policy tickets,
random baselines, and a chance (null) comparison for every anomaly statistic.

usage: python round.py DRAWS_CSV LAST_TRAINING_GAME ROUND_ID OUT_JSON
"""
import csv, hashlib, json, os, sys
from datetime import datetime, timedelta, timezone
import numpy as np

import shadow
import agreement

N_TRAIN, N_RECENT, SIMS = 100, 20, 2000
P = 0.25                          # chance a given number is drawn in a game
SIZES = list(range(4, 11))        # nested tickets K4..K10 (paytable covers K4-K10)
PRIMARY = 4                       # pre-declared ticket size for the reward (K6 up to R006)
AGREE = agreement.LABEL           # primary ticket from R018: scored, but outside the GRPO group
NEW_RULES = ("Cold (100)", "Hot last 10", "Not in last game", "Hot last 3", "Rising (2nd half)",
             "Just drawn (last game)", "Pairs with last game", "Neighbours of last game")


def load(path):
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = {"game": int(r["game"]), "closed": r["closed_utc"],
                 "draw": sorted(int(r[f"n{i}"]) for i in range(1, 21))}
            rows[g["closed"]] = g
    return sorted(rows.values(), key=lambda g: g["closed"])


def matrix(games):
    X = np.zeros((len(games), 80), dtype=np.int8)
    for i, g in enumerate(games):
        X[i, np.array(g["draw"]) - 1] = 1
    return X


def stats(X):
    n = X.shape[0]
    freq = X.sum(0)
    z_freq = (freq - n * P) / np.sqrt(n * P * (1 - P))
    rec = X[-N_RECENT:].sum(0)
    z_rec = (rec - N_RECENT * P) / np.sqrt(N_RECENT * P * (1 - P))
    seen = X[::-1].argmax(0)
    gap = np.where(X.any(0), seen, n)                     # games since last drawn
    C = X.T.astype(np.int32) @ X.astype(np.int32)         # pair co-occurrence
    np.fill_diagonal(C, 0)
    pp = (20 * 19) / (80 * 79)
    z_pair = (C - n * pp) / np.sqrt(n * pp * (1 - pp))
    np.fill_diagonal(z_pair, -np.inf)
    nums = np.arange(1, 81)
    heads = (X[:, :40].sum(1))                            # numbers 1-40 per game
    sums = (X * nums).sum(1)
    odd = X[:, 0::2].sum(1)
    consec = (X[:, :-1] & X[:, 1:]).sum(1)
    # longest run of consecutive games each number appeared in
    run = np.zeros(80, int); best = np.zeros(80, int)
    for row in X:
        run = np.where(row == 1, run + 1, 0); best = np.maximum(best, run)
    iu = np.triu_indices(80, 1)
    return {"freq": freq, "z_freq": z_freq, "rec": rec, "z_rec": z_rec, "gap": gap,
            "C": C, "z_pair": z_pair, "pairs_upper": C[iu], "heads": heads, "sums": sums,
            "odd": odd, "consec": consec, "streak": best}


def summary(s):
    """Window-level anomaly statistics compared against chance."""
    return {
        "max |z| number frequency": float(np.abs(s["z_freq"]).max()),
        "numbers with |z| > 2": int((np.abs(s["z_freq"]) > 2).sum()),
        "longest current gap (games)": int(s["gap"].max()),
        "max pair count": int(s["pairs_upper"].max()),
        "pairs seen 13+ times": int((s["pairs_upper"] >= 13).sum()),
        "max |z| last-20 frequency": float(np.abs(s["z_rec"]).max()),
        "longest appearance streak": int(s["streak"].max()),
        "heads-tails spread (sd)": float(s["heads"].std()),
        "mean draw sum dev": float(abs(s["sums"].mean() - 810)),
        "mean consecutive pairs dev": float(abs(s["consec"].mean() - 79 * pp_adj())),
    }


def pp_adj():
    return (20 * 19) / (80 * 79)


def null_compare(obs, sims, rng):
    """For each statistic: share of random 100-game windows at least as extreme."""
    keys = list(obs)
    vals = {k: [] for k in keys}
    for _ in range(sims):
        X = np.zeros((N_TRAIN, 80), dtype=np.int8)
        for i in range(N_TRAIN):
            X[i, rng.choice(80, 20, replace=False)] = 1
        sm = summary(stats(X))
        for k in keys:
            vals[k].append(sm[k])
    out = {}
    for k in keys:
        v = np.array(vals[k])
        out[k] = {"observed": round(obs[k], 3), "chance_median": round(float(np.median(v)), 3),
                  "p_chance": round(float((v >= obs[k]).mean()), 3)}
    return out


def ranking(score, tiebreak):
    order = sorted(range(80), key=lambda i: (-score[i], -tiebreak[i], i))
    return [i + 1 for i in order]


def pair_cluster(s):
    C = s["C"].astype(float)
    i, j = np.unravel_index(np.argmax(np.triu(C, 1)), C.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < 10:
        rest = [k for k in range(80) if k not in chosen]
        k = max(rest, key=lambda k: (C[k, chosen].sum(), s["freq"][k], -k))
        chosen.append(int(k))
    return [c + 1 for c in chosen]


def persistence(X, s):
    """Numbers that run above average in every quarter of the window (7+ of 25 games, 6.25
    expected), steadiest first, then most drawn. Rewards consistency rather than one hot run."""
    q = X.reshape(4, -1, 80).sum(1)
    above, spread = (q >= 7).sum(0), q.std(0)
    order = sorted(range(80), key=lambda i: (-above[i], spread[i], -s["freq"][i], i))
    return [i + 1 for i in order]


def ideal_sizes():
    """Each policy's best ticket size so far (most take-win cutoff hits over the last 10 scored
    rounds), from ledger/sizes.json. Recorded at freeze so the recommended size is frozen too."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ledger", "sizes.json")
    if not os.path.exists(p):
        return {}
    return {k: {"size": v["ideal"], "avg_net": v["avg_net"][str(v["ideal"])], "rounds": v["rounds"],
                **({"big": v["big"][str(v["ideal"])], "x_chance": v["x_chance"][str(v["ideal"])]} if "big" in v else {})}
            for k, v in json.load(open(p))["policies"].items()}


def seeded_random(label, data_hash):
    seed = int(hashlib.sha256(f"{label}|{data_hash}".encode()).hexdigest()[:16], 16)
    r = np.random.default_rng(seed)
    return [int(x) + 1 for x in r.permutation(80)[:10]], seed


def main(path, last_game, round_id, out_path):
    games = load(path)
    idx = max(i for i, g in enumerate(games) if g["game"] == last_game)
    train = games[idx - N_TRAIN + 1: idx + 1]
    nums = [g["game"] for g in train]
    assert len(train) == N_TRAIN and all((b - a) % 1000 == 1 for a, b in zip(nums, nums[1:]))
    data_hash = hashlib.sha256(json.dumps([g["draw"] for g in train]).encode()).hexdigest()
    X = matrix(train)
    s = stats(X)

    # From R018 (Daryl, 25 Oct) the policies are the 12 Retro rules plus Pair cluster. Hot, Overdue,
    # Surge and Persistence keep their own builders (and weight history); the other eight rules come
    # from shadow.rules(). Anomaly composite, the Sainte-Lague blend and the Retro pick shadow were
    # dropped (restore point: branch pre-r018-restructure).
    rules = shadow.rules(X.astype(int))
    policies = {
        "Hot (most drawn)": ranking(s["freq"], s["rec"]),
        "Overdue (longest gap)": ranking(s["gap"], -s["freq"]),
        "Surge (last 20 games)": ranking(s["z_rec"], s["freq"]),
        "Pair cluster": pair_cluster(s),
        "Persistence": persistence(X, s),
        **{r: shadow.top10(rules[r]) for r in NEW_RULES},
    }
    for label in ("Random A", "Random B"):
        pick, seed = seeded_random(label, data_hash)
        policies[label] = pick

    # Agreement (primary ticket from R018): every non-random policy's top 10 gets rank points 10..1,
    # times its current weight renormalised over those policies, summed per number. Scored but kept
    # out of GRPO because it is built from the weights.
    weights = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "ledger", "weights.json")))["weights"]
    agree = agreement.pick(policies, weights, round_id)
    policies[AGREE] = agree["policies"][AGREE]

    obs = summary(s)
    nullc = null_compare(obs, SIMS, np.random.default_rng(12345))

    flagged = {
        "hot": [(int(i + 1), int(s["freq"][i]), round(float(s["z_freq"][i]), 2))
                for i in np.argsort(-s["z_freq"])[:5]],
        "cold": [(int(i + 1), int(s["freq"][i]), round(float(s["z_freq"][i]), 2))
                 for i in np.argsort(s["z_freq"])[:5]],
        "overdue": [(int(i + 1), int(s["gap"][i])) for i in np.argsort(-s["gap"])[:5]],
        "top pairs": [],
    }
    iu = np.triu_indices(80, 1)
    order = np.argsort(-s["C"][iu])[:5]
    for o in order:
        a, b = iu[0][o], iu[1][o]
        flagged["top pairs"].append((int(a + 1), int(b + 1), int(s["C"][a, b])))

    rec = {
        "round": round_id,
        "training_games": [nums[0], nums[-1]],
        "training_closed_utc": [train[0]["closed"], train[-1]["closed"]],
        "training_data_sha256": data_hash,
        "primary_ticket_size": PRIMARY,
        "nested_sizes": SIZES,
        "policies": {k: v[:10] for k, v in policies.items()},
        "verdict_rule": ("Reward is 1 if any K4-K10 replay ticket reaches the take-win cutoff (a single prize "
                         "over $20) in the round, else 0; replay dollars reported for reference. K10 hit rate is labelled against the exact luck band: over 40 "
                         "games, 95% of random K10 tickets land between 21% and 29%. The band is a "
                         "label only, never a gate. With 7 policies, one landing outside the band by "
                         "chance is common, so only repeats across rounds count."),
        "window_vs_chance": nullc,
        "ideal_sizes": {**ideal_sizes(), **({} if AGREE in ideal_sizes() else
                        {AGREE: {"size": 4, "rounds": 0, "note": "K4 until the Agreement has scored rounds"}})},
        "agreement_weights": agree["weights_used"],
        "flagged": flagged,
    }
    json.dump(rec, open(out_path, "w"), indent=1)
    print(json.dumps({"window_vs_chance": nullc, "flagged": flagged,
                      "policies": rec["policies"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
