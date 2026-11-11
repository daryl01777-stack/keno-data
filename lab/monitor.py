#!/usr/bin/env python3
"""Randomness monitor (Daryl, 11 Nov): look for anomalies in the draw itself over the whole of draws.csv.

Every check is one number computed from the real history and from SIMS random histories of the same shape
(same game count, same gaps, same clock times, 20 of 80 drawn in a random order). Its p-value is where the
real value falls among the random ones (two-sided), so no check relies on a formula. With about 30 checks,
one or two will flag at p < 0.05 by luck; a flag only counts once it repeats on newer games
(`python lab/monitor.py since 2026-11-11` runs the same checks on games from a date on).

usage: python lab/monitor.py [run]          all games -> ledger/monitor.json and a printed table
       python lab/monitor.py since DATE     only games closed on or after DATE (UTC), printed only
"""
import csv, datetime as dt, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIMS = 400
LAGS = (1, 2, 3, 5, 10, 20)


def load(since=None):
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "draws.csv"))) if not since or r["closed_utc"] >= since]
    D = np.array([[int(r[f"n{i}"]) - 1 for i in range(1, 21)] for r in rows])            # draw order, 0-79
    g = np.array([int(r["game"]) for r in rows])
    t = [dt.datetime.fromisoformat(r["closed_utc"].replace("Z", "+00:00")) + dt.timedelta(hours=10) for r in rows]
    hour = np.array([x.hour for x in t]); mins = np.array([x.hour * 60 + x.minute for x in t])
    meta = {"game": g, "part": hour // 6,                                              # four 6-hour parts of the Brisbane day
            "after_pause": (mins >= 240) & (mins < 240 + 14),                          # first 5 games after the 3:20-4:00 am pause
            "before_pause": (mins >= 200 - 14) & (mins <= 200),                        # last 5 games before it
            "adj": {k: np.flatnonzero((g[k:] - g[:-k]) % 1000 == k) for k in LAGS}}    # i where game i+k is exactly k games later
    return rows, D, meta


def chi(obs, exp):
    return float(((obs - exp) ** 2 / exp).sum())


def stats(D, m):
    n = len(D); X = np.zeros((n, 80), int); np.put_along_axis(X, D, 1, 1)
    pos = np.zeros((n, 80)); np.put_along_axis(pos, D, np.arange(1, 21)[None, :].repeat(n, 0), 1)
    f = X.sum(0); s = {}
    # frequency
    s["Number frequency: spread across the 80 numbers"] = chi(f, n / 4)
    s["Number frequency: most drawn number"] = float(f.max())
    s["Number frequency: least drawn number"] = float(f.min())
    s["Low half (1-40) per game: average"] = float(X[:, :40].sum(1).mean())
    s["Low half (1-40) per game: spread"] = float(X[:, :40].sum(1).std())
    s["Odd numbers per game: average"] = float(X[:, ::2].sum(1).mean())
    s["Board rows (tens): spread of row totals"] = chi(f.reshape(8, 10).sum(1), n * 2.5)
    s["Board columns (last digit): spread of column totals"] = chi(f.reshape(8, 10).sum(0), n * 2.0)
    # draw order
    mp = pos.sum(0) / f
    s["Draw order: spread of each number's average position"] = float((((mp - 10.5) ** 2) * f).sum())
    s["Draw order: most extreme average position"] = float((np.abs(mp - 10.5) * np.sqrt(f)).max())
    s["Draw order: first ball spread across numbers"] = chi(np.bincount(D[:, 0], minlength=80), n / 80)
    s["Draw order: last ball spread across numbers"] = chi(np.bincount(D[:, 19], minlength=80), n / 80)
    s["Draw order: do low numbers come out early (correlation)"] = float(np.corrcoef(D.ravel(), np.tile(np.arange(20), n))[0, 1])
    s["Draw order: neighbours drawn back to back (count)"] = float((np.abs(np.diff(D, axis=1)) == 1).sum())
    s["Draw order: same board row back to back (count)"] = float((np.diff(D // 10, axis=1) == 0).sum())
    # time
    parts = np.array([X[m["part"] == p].sum(0) for p in range(4)]); pn = np.array([(m["part"] == p).sum() for p in range(4)])
    s["Time of day: number frequency differs by part of day"] = chi(parts, pn[:, None] / 4 * np.ones(80))
    for key, name in (("after_pause", "first 5 games after the 4 am restart"), ("before_pause", "last 5 games before the 3:20 am pause")):
        k = int(m[key].sum())
        s[f"Daily pause: number spread in the {name}"] = chi(X[m[key]].sum(0), k / 4) if k else 0.0
    cyc = np.array([X[(m["game"] // 100) == b].sum(0) for b in range(10)]); cn = np.array([((m["game"] // 100) == b).sum() for b in range(10)])
    s["Game number cycle (0-999): number frequency differs by hundred"] = chi(cyc, np.maximum(cn, 1)[:, None] / 4 * np.ones(80))
    # togetherness
    C = X.T @ X; iu = np.triu_indices(80, 1); pc = C[iu]
    s["Pairs: spread of how often each pair comes together"] = float(pc.var())
    s["Pairs: most frequent pair"] = float(pc.max())
    s["Pairs: least frequent pair"] = float(pc.min())
    tmax, tmin = 0, 10 ** 9
    for k in range(78):
        S = X[X[:, k] == 1][:, k + 1:]; T = np.triu(S.T @ S, 1)
        tmax = max(tmax, int(T.max())); tmin = min(tmin, int(T[np.triu_indices(len(T), 1)].min()))
    s["Triples: most frequent triple"] = float(tmax)
    s["Triples: least frequent triple"] = float(tmin)
    # memory
    for k in LAGS:
        i = m["adj"][k]
        s[f"Memory: numbers repeated from {k} game{'s' if k > 1 else ''} back (average)"] = float((X[i] & X[i + k]).sum(1).mean()) if len(i) else 5.0
    i = m["adj"][1]; both = (X[i] & X[i + 1]).sum(0); was = X[i].sum(0)
    s["Memory: some numbers repeat more than others (spread)"] = chi(both, was / 4)
    nb = (X[i][:, :-1] & X[i + 1][:, 1:]).sum() + (X[i][:, 1:] & X[i + 1][:, :-1]).sum()
    s["Memory: neighbours of last game's numbers drawn (count)"] = float(nb)
    s["Memory: same draw position as last game (count)"] = float((D[i] == D[i + 1]).sum())
    run = np.zeros(80, int); best = 0; gap = np.zeros(80, int); worst = 0; g = m["game"]
    for j in range(n):
        if j and (g[j] - g[j - 1]) % 1000 != 1:
            run[:] = 0; gap[:] = 0
        run = (run + 1) * X[j]; gap = (gap + 1) * (1 - X[j])
        best = max(best, int(run.max())); worst = max(worst, int(gap.max()))
    s["Streaks: longest run of one number in consecutive games"] = float(best)
    s["Streaks: longest absence of one number"] = float(worst)
    return s


def run(since=None):
    rows, D, meta = load(since)
    real = stats(D, meta); rng = np.random.default_rng(20261111); n = len(D)
    sims = {k: [] for k in real}
    for _ in range(SIMS):
        S = np.argsort(rng.random((n, 80)), axis=1)[:, :20]
        for k, v in stats(S, meta).items():
            sims[k].append(v)
    out = []
    for k, v in real.items():
        a = np.array(sims[k]); lo = (np.sum(a <= v) + 1) / (SIMS + 1); hi = (np.sum(a >= v) + 1) / (SIMS + 1)
        p = min(1.0, 2 * min(lo, hi))
        out.append({"check": k, "value": round(v, 4), "random_typical": round(float(np.median(a)), 4),
                    "random_low": round(float(np.percentile(a, 2.5)), 4), "random_high": round(float(np.percentile(a, 97.5)), 4),
                    "p": round(float(p), 4), "flag": bool(p < 0.05)})
    return {"games": n, "from": rows[0]["closed_utc"], "to": rows[-1]["closed_utc"], "sims": SIMS, "checks": out,
            "flags": sum(x["flag"] for x in out), "flags_expected_by_luck": round(0.05 * len(out), 1)}


if __name__ == "__main__":
    since = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "since" else None
    r = run(since)
    if not since:
        json.dump(r, open(os.path.join(ROOT, "ledger", "monitor.json"), "w"), indent=1)
    print(f"{r['games']} games, {r['from']} to {r['to']}; {len(r['checks'])} checks, {r['flags']} flagged (about {r['flags_expected_by_luck']} expected by luck)")
    for x in r["checks"]:
        print(f"{'FLAG' if x['flag'] else '    '} p {x['p']:.3f}  {x['check']:62s} {x['value']:>12.4f}  (random {x['random_low']:.4f} to {x['random_high']:.4f})")
