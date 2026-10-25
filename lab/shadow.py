#!/usr/bin/env python3
"""Retro pick: a shadow policy that automates the retrospective (from R010, Daryl, 23 Oct).

At freeze, a library of simple rules is run over every completed 100+40 game window before the
round (non-overlapping, stepping forward 40 games from the oldest game in draws.csv, so windows stay
put as history grows; from R018, before that they stepped back from the last training game and were
re-cut every freeze). The rule with the most windows with a take-win cutoff hit (replay.reward) (ties: higher K10 catches per game, then library order)
picks the round's ticket from the training window. round.py stores the result under "shadow" in
frozen.json, so the fingerprint covers it. It sits outside the GRPO group, the weights and the
blend; score.py ignores it.

usage: python lab/shadow.py R010                score a round's shadow once its test games are in
                                                draws.csv (after score.py); writes
                                                rounds/<R>/shadow_scores.json, appends ledger/shadow.csv
       python lab/shadow.py freeze R009 308     late shadow ticket for a round already frozen, built
                                                from draws up to game 308 only (rounds/<R>/shadow.json)
"""
import csv, json, os, sys
import numpy as np

from replay import big, paytable, replay, reward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN, TEST = 100, 40
LABEL = "Retro pick"


def rules(W):
    """{rule: score per number}, higher is better, for a 100-game window W (games x 80)."""
    f = W.sum(0)
    seen = W[::-1].argmax(0)
    gap = np.where(W.any(0), seen, len(W))
    q = W.reshape(4, 25, 80).sum(1)
    last = W[-1]
    nb = np.zeros(80)
    for i in np.flatnonzero(last):
        for j in (i - 1, i + 1):
            if 0 <= j < 80:
                nb[j] += 1
    C = W.T @ W
    np.fill_diagonal(C, 0)
    return {
        "Hot (100)": f,
        "Cold (100)": -f,
        "Surge (last 20)": W[-20:].sum(0),
        "Hot last 10": W[-10:].sum(0),
        "Overdue": gap,
        "Just drawn (last game)": last * 1000 + f,
        "Not in last game": -last * 1000 + f,
        "Hot last 3": W[-3:].sum(0),
        "Persistence": q.min(0) * 1000 + f,
        "Neighbours of last game": nb,
        "Pairs with last game": C[:, last == 1].sum(1),
        "Rising (2nd half)": W[50:].sum(0) - W[:50].sum(0),
    }


def top10(score):
    """Best ten numbers, ties to the smaller number (stable sort keeps number order)."""
    return [int(i) + 1 for i in np.argsort(-score, kind="stable")[:10]]


def catches(ticket, T):
    """Per-game catches for nested K4-K10 tickets over test games T."""
    hit = T[:, np.array(ticket) - 1]
    return {k: [int(x) for x in hit[:, :k].sum(1)] for k in range(4, 11)}


def replays(per_game, pay, first=0):
    return {k: replay(k, h, [pay[k].get(x, 0) for x in h], first) for k, h in per_game.items()}


def pick(X):
    """X: matrix of every game up to and including the last training game, oldest first."""
    pay = paytable()
    X = np.asarray(X, dtype=int)
    n = len(X)
    starts = list(range(0, n - TRAIN - TEST + 1, TEST))   # anchored at the oldest game
    names = list(rules(X[-TRAIN:]))
    cut = {r: 0 for r in names}
    k10 = {r: 0.0 for r in names}
    by_size = {r: {k: [0, 0] for k in range(4, 11)} for r in names}   # [cutoff hits, net]
    for s in starts:
        T = X[s + TRAIN:s + TRAIN + TEST]
        for r, v in rules(X[s:s + TRAIN]).items():
            pg = catches(top10(v), T)
            reps = replays(pg, pay)
            cut[r] += reward(reps)
            k10[r] += sum(pg[10]) / TEST
            for k, x in reps.items():
                by_size[r][k][0] += big(x)
                by_size[r][k][1] += x["net"]
    best = max(names, key=lambda r: (cut[r], k10[r], -names.index(r)))
    # ideal size for the chosen rule, as in replay.size_table: most cutoff hits, then net, then smaller
    ideal = max(range(4, 11), key=lambda k: (by_size[best][k][0], by_size[best][k][1], -k))
    return {
        "policies": {LABEL: top10(rules(X[-TRAIN:])[best])},
        "rule": best,
        "ideal_size": ideal,
        "windows": len(starts),
        "library": {r: {"cutoff_hits": cut[r], "k10_per_game": round(k10[r] / len(starts), 3)}
                    for r in names},
    }


def freeze(rid, last_game):
    """Shadow ticket for a round frozen before Retro pick existed (R009): built only from draws up to
    last_game, written to rounds/<R>/shadow.json (canonical) with its sha256."""
    from datetime import datetime, timezone
    from round import load, matrix
    games = load(os.path.join(ROOT, "draws.csv"))
    idx = max(i for i, g in enumerate(games) if g["game"] == last_game)
    rec = pick(matrix(games[:idx + 1]))
    rec.update(round=rid, built_from_last_game=last_game, added_after_freeze=True,
               frozen_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    d = os.path.join(ROOT, "rounds", rid)
    open(os.path.join(d, "shadow.json"), "w").write(body)
    sha = __import__("hashlib").sha256(body.encode()).hexdigest()
    open(os.path.join(d, "shadow.sha256"), "w").write(sha)
    print(sha)
    print(json.dumps({k: v for k, v in rec.items() if k != "library"}))


def score(rid):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from round import load, matrix
    frozen = json.load(open(os.path.join(ROOT, "rounds", rid, "frozen.json")))
    sp = os.path.join(ROOT, "rounds", rid, "shadow.json")
    sh = json.load(open(sp)) if os.path.exists(sp) else frozen["shadow"]
    ticket = sh["policies"][LABEL]
    first, last = frozen["test_games"]
    want = [(first + i) % 1000 for i in range((last - first) % 1000 + 1)]
    games = [g for g in load(os.path.join(ROOT, "draws.csv")) if g["closed"] > frozen["frozen_utc"]]
    test = [next(g for g in games if g["game"] == n) for n in want]
    pay = paytable()
    pg = catches(ticket, matrix(test).astype(int))
    reps = replays(pg, pay, first)
    group = {}                                   # the round's own rewards, for comparison
    sp = os.path.join(ROOT, "rounds", rid, "scores.json")
    if os.path.exists(sp):
        from replay import round_replays
        group = {p: reward(v) for p, v in round_replays(json.load(open(sp)), pay).items()}
    out = {
        "round": rid, "policy": LABEL, "rule": sh["rule"], "ticket": ticket,
        "k10_hits": sum(pg[10]), "k10_hit_rate": round(sum(pg[10]) / (10 * len(test)), 4),
        "reward": reward(reps),
        "sizes": {str(k): {"per_game": pg[k], "replay": reps[k]} for k in pg},
        "compare": {p: group[p] for p in ("Random A", "Random B", "Weighted blend") if p in group},
    }
    with open(os.path.join(ROOT, "rounds", rid, "shadow_scores.json"), "w") as f:
        json.dump(out, f, indent=1)
    led = os.path.join(ROOT, "ledger", "shadow.csv")
    new = not os.path.exists(led)
    with open(led, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["round", "policy", "rule", "k10_hits", "k10_hit_rate", "reward",
                        "random_a_reward", "random_b_reward", "big_sizes"])
        w.writerow([rid, LABEL, sh["rule"], out["k10_hits"], out["k10_hit_rate"], out["reward"],
                    group.get("Random A", ""), group.get("Random B", ""),
                    " ".join(f"K{k}" for k in reps if big(reps[k]))])
    print(json.dumps({k: v for k, v in out.items() if k != "sizes"}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        freeze(sys.argv[2], int(sys.argv[3]))
    else:
        score(sys.argv[1])
