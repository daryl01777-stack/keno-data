#!/usr/bin/env python3
"""Boosted trees build (Daryl, 13 Nov): a machine-learning ticket run as one of the AI builds.

At a freeze it learns only from games up to the round's last training game (all of draws.csv before
that, not just the 100-game window). For every earlier game it knows 33 facts about each number (how
often it came up over the last 3 to 300 games, whether it was drawn in each of the last 12 games, its
gap, its draw position, neighbours and board position, pairs with the last game's numbers, the hour)
and how many times that number was then drawn over a 40-game window placed like the test window.
Gradient-boosted trees learn that link; the build is the 10 numbers forecast to be drawn most over
the test window, in order, nested as K4-K10 like any other build.

Needs scikit-learn (`pip install scikit-learn`). The ticket is written into aibuilds.json at the
freeze, so scoring never has to rebuild it. Replayed walk-forward over R018-R071 (53 rounds) the same
recipe won in 6 rounds (K4 2, K8 3, K9 1, K10 3) for 107 points (`python lab/boosted.py replay 18 71`);
it joins the builds on the group's median points like the other late builds, the first time it is scored.
Its forecasts sit very close to level, so a one-game change in the gap can swap two or three numbers.

usage: python lab/boosted.py R0NN            print the ticket for a frozen round
       python lab/boosted.py replay A B      replay rounds A-B (no files written)
"""
import csv, json, os, sys
from datetime import datetime, timedelta
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOW, FIRST, MIN_POINTS = 40, 300, 400      # test window; games skipped at the start; least history to train on
_cache = {}


def draws():
    """All games in close order: rows, and POS (games x 80) = draw position 1-20, 0 if not drawn."""
    if "d" not in _cache:
        seen, R = set(), []
        for r in sorted(csv.DictReader(open(os.path.join(ROOT, "draws.csv"))), key=lambda r: r["closed_utc"]):
            if r["closed_utc"] not in seen:
                seen.add(r["closed_utc"]); R.append(r)
        POS = np.zeros((len(R), 80), np.int8)
        for i, r in enumerate(R):
            for k in range(1, 21):
                POS[i, int(r[f"n{k}"]) - 1] = k
        hour = np.array([int(r["closed_brisbane"][11:13]) for r in R])
        _cache["d"] = (R, POS, hour, {r["closed_utc"]: i for i, r in enumerate(R)})
    return _cache["d"]


def features(POS, hour):
    """games x 80 x 33; the row for game t uses only games before t."""
    X = POS > 0; N = len(X); t = np.arange(N)
    cs = np.vstack([np.zeros((1, 80)), np.cumsum(X, 0)])
    F = []
    for w in (3, 5, 10, 20, 50, 100, 300):                                   # rate over the last w games
        F.append((cs[t] - cs[np.maximum(t - w, 0)]) / np.minimum(np.maximum(t, 1), w)[:, None])
    F.append(cs[t] / np.maximum(t, 1)[:, None])                              # all-time rate
    for lag in range(1, 13):                                                 # drawn exactly `lag` games ago
        F.append(np.vstack([np.zeros((lag, 80)), X[:-lag]]).astype(float))
    gap = np.zeros((N, 80)); g = np.full(80, 4.0)
    for i in range(N):
        gap[i] = g; g = np.where(X[i], 0, g + 1)
    F.append(np.minimum(gap, 60))                                            # games since last drawn
    F.append(np.vstack([np.zeros((1, 80)), POS[:-1]]).astype(float))         # draw position in the last game
    prev = np.vstack([np.zeros((1, 80), bool), X[:-1]])
    nb = np.zeros((N, 80)); nb[:, 1:] += prev[:, :-1]; nb[:, :-1] += prev[:, 1:]
    F.append(nb)                                                             # neighbours drawn last game
    board = prev.reshape(N, 8, 10)
    F.append(np.repeat(board.sum(2), 10, 1).astype(float))                   # its board row, last game
    F.append(np.tile(board.sum(1), (1, 8)).astype(float))                    # its board column, last game
    pair = np.zeros((N, 80)); Xf = X.astype(np.float32)
    for i in range(2, N):                                                    # draws alongside last game's numbers, 100 games
        a = Xf[max(0, i - 101):i - 1]
        pair[i] = (a.T @ (a @ Xf[i - 1])) / len(a)
    F.append(pair)
    pcs = np.vstack([np.zeros((1, 80)), np.cumsum(POS, 0)]); lo = np.maximum(t - 100, 0)
    cnt = cs[t] - cs[lo]
    F.append(np.where(cnt > 0, (pcs[t] - pcs[lo]) / np.maximum(cnt, 1), 10.5))   # average draw position, 100 games
    F.append(np.repeat(hour[:, None], 80, 1).astype(float))                  # Brisbane hour
    num = np.tile(np.arange(80), (N, 1))
    F += [num.astype(float), (num % 10).astype(float), (num // 10).astype(float)]
    return np.stack(F, 2).astype(np.float32)


def forecast(L, gap, hour_next, seed=0):
    """Forecast draws per number over games L+1+gap .. L+40+gap, from games 0..L only.
    Returns (ranked top 10, number of trees)."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        raise SystemExit("The Boosted trees build needs scikit-learn: run `pip install scikit-learn` and build again.")
    _, POS, hour, _ = draws()
    POS = np.vstack([POS[:L + 1], np.zeros((1, 80), np.int8)])               # nothing after the last training game
    hour = np.append(hour[:L + 1], hour_next)
    Fa = features(POS, hour)
    cs = np.vstack([np.zeros((1, 80)), np.cumsum(POS[:L + 1] > 0, 0)])
    t = np.arange(FIRST, L + 1 - gap - (WINDOW - 1))                         # points whose window ends by game L
    if len(t) < MIN_POINTS:
        raise SystemExit(f"Boosted trees needs more history: {L + 1} games before the freeze is too few.")
    y = cs[t + gap + WINDOW] - cs[t + gap]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=400,
                                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                      n_iter_no_change=20, random_state=seed)
    m.fit(Fa[t].reshape(-1, Fa.shape[2]), y.ravel())
    pred = m.predict(Fa[L + 1])
    return [int(j) + 1 for j in np.argsort(-pred, kind="stable")[:10]], int(m.n_iter_)


def ticket(r):
    """Ranked top 10 for frozen round r (int or 'R0NN')."""
    rid = r if isinstance(r, str) else f"R{r:03d}"
    f = json.load(open(os.path.join(ROOT, "rounds", rid, "frozen.json")))
    R, _, _, at = draws()
    last = f["training_closed_utc"][-1]
    if last not in at:
        raise SystemExit(f"{rid}: the last training game ({last}) is not in draws.csv.")
    L = at[last]
    utc = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    first = utc(f["test_expected_close_utc"][0])
    gap = int(min(10, max(0, round((first - utc(last)).total_seconds() / 160) - 1)))   # games between training and test
    return forecast(L, gap, (first + timedelta(hours=10)).hour)[0]


if __name__ == "__main__":
    if sys.argv[1] == "replay":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import aibuilds as ab, aipick, replay as rp
        a, b = (int(x.lstrip("R")) for x in sys.argv[2:4])
        pts, wins, sizes = ab.START, 0, {k: 0 for k in range(4, 11)}
        for r in range(a, b + 1):
            if not aipick.load(r, "scores.json"):
                continue
            t = ticket(r); games, first = ab.test_games(r)
            res = ab.score_ticket(t, games, first); o, p = ab.outcome(res); pts += p; wins += o == "win"
            for k, x in res.items():
                sizes[k] += rp.big(x)
            print(f"R{r:03d} {t}  {o:8s} {pts}  " + " ".join(f"K{k}" for k, x in res.items() if rp.big(x)))
        print(f"rounds won {wins}, points {pts}, wins by size {sizes}")
    else:
        print(ticket(sys.argv[1]))
