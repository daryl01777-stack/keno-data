#!/usr/bin/env python3
"""Signal strength for the Agreement ticket (from R018, Daryl, 25 Oct).

Each policy has a raw score per number on the 100-game training window (higher = stronger for that
policy). Strength = -log10(P(score >= observed) on a random 100-game history), from a chance table
built once by simulation and stored in ledger/strength_null.json. Top-10 numbers of any policy are
in the upper tail by construction, so even on random draws a #1 pick gets some strength; what the
strength adds over rank points is how far out in that tail the number sits compared with chance.

usage: python lab/strength.py build [SIMS]    rebuild ledger/strength_null.json (seeded, reproducible)
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "ledger", "strength_null.json")
N_TRAIN, SIMS, SEED = 100, 20000, 20261025


def raw(W):
    """{policy: raw score per number (80,)} for a 100-game window W (games x 80, 0/1)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from round import pair_cluster
    W = np.asarray(W, dtype=np.int64)
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
    chosen = np.array(pair_cluster({"C": C, "freq": f})) - 1
    return {
        "Hot (most drawn)": f,
        "Overdue (longest gap)": gap,
        "Surge (last 20 games)": W[-20:].sum(0),
        "Persistence": (q >= 7).sum(0) * 1000 + f,
        "Pair cluster": C[:, chosen].sum(1),                # co-occurrence with the cluster's 10 numbers
        "Cold (100)": -f,
        "Hot last 10": W[-10:].sum(0),
        "Not in last game": -last * 1000 + f,
        "Hot last 3": W[-3:].sum(0),
        "Rising (2nd half)": W[50:].sum(0) - W[:50].sum(0),
        "Just drawn (last game)": last * 1000 + f,
        "Pairs with last game": C[:, last == 1].sum(1),
        "Neighbours of last game": nb,
    }


def build(sims=SIMS):
    rng = np.random.default_rng(SEED)
    vals = {}
    for _ in range(sims):
        W = np.zeros((N_TRAIN, 80), dtype=np.int64)
        for g in range(N_TRAIN):
            W[g, rng.choice(80, 20, replace=False)] = 1
        for p, v in raw(W).items():
            vals.setdefault(p, []).extend(np.asarray(v, dtype=float).tolist())
    out = {"sims": sims, "seed": SEED, "n_values": sims * 80, "policies": {}}
    for p, v in vals.items():
        v = np.array(v)
        u = np.unique(v)
        ge = len(v) - np.searchsorted(np.sort(v), u, side="left")      # values >= u
        out["policies"][p] = {"values": u.tolist(), "n_ge": ge.astype(int).tolist()}
    json.dump(out, open(TABLE, "w"), separators=(",", ":"))
    print(f"wrote {TABLE}: {sims} random histories, {len(out['policies'])} policies")


_table = None


def strength(policy, value):
    """-log10 P(score >= value) under chance, floored at 0; add-one smoothing so it stays finite."""
    global _table
    if _table is None:
        _table = json.load(open(TABLE))
    t = _table["policies"][policy]
    i = int(np.searchsorted(t["values"], value, side="left"))
    n_ge = t["n_ge"][i] if i < len(t["values"]) else 0
    return max(0.0, -np.log10((n_ge + 1) / (_table["n_values"] + 1)))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else SIMS)
