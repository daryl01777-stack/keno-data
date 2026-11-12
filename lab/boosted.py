#!/usr/bin/env python3
"""Boosted trees build (Daryl, 13 Nov): a machine-learning K6 ticket run as one of the AI builds.

This is the optimised version (Daryl, 13 Nov: "replace the old boosted trees with the optimized
version"); the first version, which forecast single numbers, never played a live round.

What it does at a freeze, using only games up to the round's last training game:
1. Looks back at every third past game and deals 150 six-number tickets there (half at random, half
   from a pool of 14 numbers picked by a random mix of the numbers' form).
2. For each ticket it records 47 facts as they stood then: the six numbers' own form (rates over 10 to
   300 games and all-time, gap, last draw position: average, lowest, highest), how they behaved
   TOGETHER (how often 3, 4 or 5 of them landed in the same game over the last 100, 300, 1,000 and all
   games; their tightest and loosest pair), how many were in the last game and the last three, and the
   state of play (Brisbane hour, place in the 0-999 cycle, games since the last long pause).
3. The target is what a K6 lives on: in how many games of a 40-game window placed like the test window
   the ticket caught 4 or more.
4. Five copies of gradient-boosted trees (4 leaves each) learn that link and are averaged, so the ticket
   does not flip on the model's random start. Each copy stops adding trees when the most recent 15% of
   its training period stops improving (kept apart by a gap, because overlapping windows fool a
   shuffled check).
5. It searches for the best-scoring six: 20,000 dealt tickets, then swapping one number at a time.
   The six are ordered by how much each one adds; numbers 7-10 are the best four to swap in. That
   ranked top 10 is the build, nested K4-K10 like the others, though K6 is the size it is built for.

Equal scores are settled by a seeded shuffle, so when the model has no view the ticket is in effect a
random six (a draft settled ties towards lower numbers and kept handing back 1 2 3 4 5 6). It has no
facts about a ticket's shape on the board. Needs scikit-learn (`pip install scikit-learn`); takes two to three minutes. The ticket
is written into aibuilds.json at the freeze, so scoring never rebuilds it.

Tested before it went in, as the draft with the board-shape facts (tuned on games up to 10 Nov, then
a held-back exam on the latest 1,500 games played as 37 rounds, then 129 earlier walk-forward rounds):
steadier and aimed at K6, but its tickets landed no more often than chance (166 rounds: 13 K6 wins
under the lab rule against 13.9 expected).

usage: python lab/boosted.py R0NN            print the ranked top 10 for a frozen round
       python lab/boosted.py replay A B      replay rounds A-B (no files written; about 2 minutes a round)
"""
import csv, itertools, json, os, sys
from datetime import datetime, timedelta
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW, FIRST, STEP, PER, COPIES, DEAL = 40, 1000, 3, 150, 5, 20000
PAIRS = list(itertools.combinations(range(6), 2))
_cache = {}


def draws():
    """All games in close order and what the build reads from them."""
    if "d" not in _cache:
        seen, R = set(), []
        for r in sorted(csv.DictReader(open(os.path.join(ROOT, "draws.csv"))), key=lambda r: r["closed_utc"]):
            if r["closed_utc"] not in seen:
                seen.add(r["closed_utc"]); R.append(r)
        N = len(R)
        POS = np.zeros((N, 80), np.int8)                      # draw position 1-20, 0 if not drawn
        for i, r in enumerate(R):
            for k in range(1, 21):
                POS[i, int(r[f"n{k}"]) - 1] = k
        X = (POS > 0).astype(np.float32)
        secs = np.array([np.datetime64(r["closed_utc"][:19]) for r in R]).astype("datetime64[s]").astype(np.int64)
        brk = np.r_[0, np.flatnonzero(np.diff(secs) > 900) + 1]   # first game after each long pause
        _cache["d"] = dict(
            R=R, POS=POS, X=X, cs=np.vstack([np.zeros((1, 80), np.float32), np.cumsum(X, 0)]),
            hour=np.array([int(r["closed_brisbane"][11:13]) for r in R]),
            game=np.array([int(r["game"]) for r in R]),
            since=np.arange(N) - brk[np.searchsorted(brk, np.arange(N), side="right") - 1],
            at={r["closed_utc"]: i for i, r in enumerate(R)})
    return _cache["d"]


def number_facts(t):
    """80 x 8 single-number facts from games before game t."""
    d = draws(); cs, X = d["cs"], d["X"]
    r = lambda w: (cs[t] - cs[max(t - w, 0)]) / min(t, w)
    back = X[:t][::-1]
    gap = np.where(back.any(0), back.argmax(0), 60).astype(np.float32)
    return np.stack([r(10), r(20), r(50), r(100), r(300), cs[t] / t, np.minimum(gap, 60), d["POS"][t - 1].astype(np.float32)], 1)


def deal(t, n, rng):
    """n six-number tickets (sorted, 0-79): half uniform, half from a 14-number pool chosen by a random mix of form."""
    T = np.argsort(rng.random((n, 80)), 1)[:, :6]
    nf = number_facts(t); z = (nf - nf.mean(0)) / (nf.std(0) + 1e-9)
    for i in range(n // 2):
        T[i] = rng.choice(np.argsort(-(z @ rng.normal(size=z.shape[1])))[:14], 6, replace=False)
    return np.sort(T, 1)


def ticket_facts(t, T):
    """tickets x 47 facts as they stood before game t."""
    d = draws(); X = d["X"]; out = []
    for i in range(0, len(T), 3000):                           # in slices, to keep memory small
        S = T[i:i + 3000]; n = len(S)
        M = np.zeros((80, n), np.float32); M[S.T, np.arange(n)[None, :]] = 1
        nf = number_facts(t)[S]
        F = [nf.mean(1), nf.min(1), nf.max(1)]
        for w, pairs in ((100, False), (300, True), (1000, False), (t, True)):
            Xw = X[max(t - w, 0):t]; c = Xw @ M                # the ticket's catch in each past game
            F.append(np.stack([(c ** 2).mean(0), (c >= 3).mean(0), (c >= 4).mean(0), (c >= 5).mean(0)], 1))
            if pairs:
                C = (Xw.T @ Xw) / len(Xw); pc = np.stack([C[S[:, a], S[:, b]] for a, b in PAIRS], 1)
                F.append(np.stack([pc.max(1), pc.min(1)], 1))
        F.append(np.stack([X[t - 1] @ M, X[t - 3:t].sum(0) @ M], 1))
        F.append(np.tile(np.array([d["hour"][t - 1], d["game"][t - 1] % 1000, min(d["since"][t - 1], 600)], np.float32), (n, 1)))
        out.append(np.hstack(F).astype(np.float32))
    return np.vstack(out)


def examples(L, gap):
    """Past tickets and how many games with 4+ each caught in its window; every window ends by game L."""
    d = draws(); X = d["X"]; Fs, ys, ts = [], [], []
    for t in range(FIRST, L + 2 - gap - WINDOW, STEP):
        key = (t, gap)
        if key not in _cache:                                  # the same deal at a time point whichever round asks
            T = deal(t, PER, np.random.default_rng(t))
            M = np.zeros((80, PER), np.float32); M[T.T, np.arange(PER)[None, :]] = 1
            _cache[key] = (ticket_facts(t, T), ((X[t + gap:t + gap + WINDOW] @ M) >= 4).sum(0).astype(np.float32))
        Fs.append(_cache[key][0]); ys.append(_cache[key][1]); ts.append(np.full(PER, t))
    return np.vstack(Fs), np.concatenate(ys), np.concatenate(ts)


def learn(F, y, t, gap, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    p = dict(max_iter=300, learning_rate=0.03, max_leaf_nodes=4, min_samples_leaf=500, l2_regularization=1.0,
             early_stopping=False, random_state=seed)
    cut = np.quantile(t, 0.85); tr = t < cut - gap - WINDOW; va = t >= cut
    rng = np.random.default_rng(seed); keep = tr & (rng.random(len(t)) < 0.8)     # each copy sees a different 80%
    m = HistGradientBoostingRegressor(**p).fit(F[keep], y[keep])
    err = [np.mean((y[va] - q) ** 2) for q in m.staged_predict(F[va])]
    best = int(np.argmin(err)) + 1
    p["max_iter"] = best if err[best - 1] < np.mean((y[va] - y[keep].mean()) ** 2) else 1
    keep = rng.random(len(t)) < 0.8
    return HistGradientBoostingRegressor(**p).fit(F[keep], y[keep]), p["max_iter"]


def forecast(L, gap, seed=0, detail=False):
    """Ranked top 10 (numbers 1-80) for the window L+1+gap .. L+40+gap, from games 0..L only."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        raise SystemExit("The Boosted trees build needs scikit-learn: run `pip install scikit-learn` and build again.")
    if L + 2 - gap - WINDOW - FIRST < 600:
        raise SystemExit(f"Boosted trees needs more history: {L + 1} games before the freeze is too few.")
    F, y, tt = examples(L, gap)
    fits = [learn(F, y, tt, gap, seed + s) for s in range(COPIES)]
    score = lambda T: np.mean([m.predict(ticket_facts(L + 1, T)) for m, _ in fits], 0)
    rng = np.random.default_rng(seed + L)
    T = deal(L + 1, DEAL, rng)
    # Best first. Equal scores are common (small trees score in steps, and a copy with nothing to say scores every
    # ticket alike), so ties are settled by a seeded shuffle of the 80 numbers: never by lower numbers, which would
    # hand back 1 2 3 4 5 6 whenever the model has no view.
    shuffle = rng.permutation(80)
    order = lambda T, p: T[np.lexsort(tuple(np.sort(shuffle[T], 1).T[::-1]) + (-np.round(p, 9),))]
    T = order(T, score(T))[:15]
    for _ in range(4):                                         # swap one number at a time
        cand = [T]
        for row in T:
            new = np.setdiff1d(np.arange(80), row)
            for pos in range(6):
                cand.append(np.sort(np.column_stack([np.tile(np.delete(row, pos), (len(new), 1)), new]), 1))
        C = np.unique(np.vstack(cand), axis=0); p = score(C); T = order(C, p)[:15]
    best = T[0]; top = float(score(best[None, :])[0])
    new = np.setdiff1d(np.arange(80), best)
    swaps = np.stack([np.sort(np.column_stack([np.tile(np.delete(best, pos), (len(new), 1)), new]), 1) for pos in range(6)])
    sw = score(swaps.reshape(-1, 6)).reshape(6, len(new))
    adds = top - sw.mean(1)                                    # what each of the six adds over a typical replacement
    six = [int(best[i]) for i in sorted(range(6), key=lambda i: (-round(float(adds[i]), 9), shuffle[best[i]]))]
    rest = [int(new[j]) for j in sorted(range(len(new)), key=lambda j: (-round(float(sw[:, j].mean()), 9), shuffle[new[j]]))[:4]]
    top10 = [n + 1 for n in six + rest]
    return (top10, dict(trees=[n for _, n in fits], score=top, typical=float(y.mean()), examples=int(len(y)))) if detail else top10


def window(rid):
    """(L, gap) for a frozen round: index of its last training game and games skipped before the test window."""
    f = json.load(open(os.path.join(ROOT, "rounds", rid, "frozen.json")))
    at = draws()["at"]; last = f["training_closed_utc"][-1]
    if last not in at:
        raise SystemExit(f"{rid}: the last training game ({last}) is not in draws.csv.")
    utc = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    gap = round((utc(f["test_expected_close_utc"][0]) - utc(last)).total_seconds() / 160) - 1
    return at[last], int(min(10, max(0, gap)))


def ticket(r, detail=False):
    """Ranked top 10 for frozen round r (int or 'R0NN')."""
    rid = r if isinstance(r, str) else f"R{r:03d}"
    L, gap = window(rid)
    return forecast(L, gap, detail=detail)


if __name__ == "__main__":
    if sys.argv[1] == "replay":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import aibuilds as ab, aipick, replay as rp
        a, b = (int(x.lstrip("R")) for x in sys.argv[2:4])
        pts, wins, sizes, net6 = ab.START, 0, {k: 0 for k in range(4, 11)}, 0
        for r in range(a, b + 1):
            if not aipick.load(r, "scores.json"):
                continue
            t, info = ticket(r, detail=True); games, first = ab.test_games(r)
            res = ab.score_ticket(t, games, first); o, p = ab.outcome(res); pts += p; wins += o == "win"; net6 += res[6]["net"]
            for k, x in res.items():
                sizes[k] += rp.big(x)
            print(f"R{r:03d} {t}  trees {info['trees']}  {o:8s} {pts}  K6 {res[6]['end']:14s} " + " ".join(f"K{k}" for k, x in res.items() if rp.big(x)), flush=True)
        print(f"rounds won {wins}, points {pts}, wins by size {sizes}, K6 net {net6:+d}")
    else:
        t, info = ticket(sys.argv[1], detail=True)
        print(t, info)
