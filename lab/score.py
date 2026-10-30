#!/usr/bin/env python3
"""Score a frozen Keno round and apply the GRPO weight update.

usage: python lab/score.py R001
Reads rounds/<R>/frozen.json, draws.csv, keno_qld_paytable.json, ledger/weights.json.
Writes rounds/<R>/scores.json, updates ledger/weights.json and ledger/sizes.json, appends
ledger/leaderboard.csv. The reward is 1 if any K4-K10 ticket reached the take-win cutoff, else 0 (see replay.py).
"""
import csv, json, math, os, sys
from datetime import datetime, timedelta, timezone

from replay import WINDOW, paytable, replay, reward, round_replays, scored_rounds, size_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = range(4, 11)

OUTSIDE = ("Weighted blend", "Agreement")    # built from the weights: scored, but outside GRPO
RULE_POINTS = ("points since R018: start 100, +5 for a round with a cutoff win, +1 if no win but a ticket timed out, "
               "-1 if every ticket busted; weight = points / total")

def path(*p):
    return os.path.join(ROOT, *p)


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


POINTS_FROM, POINTS_START = "R038", "R018"   # points weights (Daryl, 28 Oct, applied after the R037 review)
START, WIN, SURVIVE, BUST = 100, 5, 1, -1


def round_points(reps):
    """+5 if any K4-K10 replay ticket reached the take-win cutoff, +1 if none did but one timed out
    (did not bust), -1 if every ticket busted."""
    if any(r["end"].startswith("big win") for r in reps.values()):
        return WIN
    return SURVIVE if any(r["end"] == "timed out" for r in reps.values()) else BUST


def points_weights(upto, current=None):
    """Every policy starts on 100 points at POINTS_START and adds round_points for each scored round up to
    `upto` (plus `current`, this round's {policy: replays}, when given); weight = points / total.
    Returns (weights, points)."""
    pay = paytable()
    pts = {}
    for r in scored_rounds():
        if POINTS_START <= r <= upto and not (current and r == upto):
            for p, v in round_replays(json.load(open(path("rounds", r, "scores.json"))), pay).items():
                if p not in OUTSIDE:
                    pts[p] = pts.get(p, START) + round_points(v)
    for p, v in (current or {}).items():
        if p not in OUTSIDE:
            pts[p] = pts.get(p, START) + round_points(v)
    total = sum(pts.values())
    return {p: round(x / total, 6) for p, x in pts.items()}, pts


def averaged_weights(rid, current):
    """Softmax of each policy's mean advantage over the last WINDOW scored rounds (this one
    included). Every round's advantages are recomputed from its replays under the current reward, so the whole
    window is judged the same way. A policy added later counts the rounds it missed as zero advantage (the
    group average), so one lucky first round can't carry its whole weight."""
    pay = paytable()
    rounds = []
    for r in scored_rounds()[-(WINDOW - 1):]:
        if r == rid:
            continue
        reps = round_replays(json.load(open(path("rounds", r, "scores.json"))), pay)
        rew = {p: reward(v) for p, v in reps.items() if p not in OUTSIDE}
        m = sum(rew.values()) / len(rew)
        sd = math.sqrt(sum((x - m) ** 2 for x in rew.values()) / len(rew))
        rounds.append({p: (x - m) / sd if sd > 0 else 0.0 for p, x in rew.items()})
    rounds = rounds[-(WINDOW - 1):] + [current]
    avg = {n: sum(r.get(n, 0.0) for r in rounds) / len(rounds) for n in current}
    e = {n: math.exp(a) for n, a in avg.items()}
    total = sum(e.values())
    return {n: round(v / total, 6) for n, v in e.items()}


def main(rid):
    frozen = json.load(open(path("rounds", rid, "frozen.json")))
    pay = {int(k): {int(h): v for h, v in t.items()}
           for k, t in json.load(open(path("keno_qld_paytable.json")))["payouts"].items()}
    first, last = frozen["test_games"]
    n_test = (last - first) % 1000 + 1
    want = [(first + i) % 1000 for i in range(n_test)]
    frozen_at = parse(frozen["frozen_utc"])
    games = {}
    with open(path("draws.csv"), newline="") as f:
        for r in csv.DictReader(f):
            t = parse(r["closed_utc"])
            if frozen_at < t <= frozen_at + timedelta(hours=6) and int(r["game"]) in want:
                games[int(r["game"])] = {"closed": r["closed_utc"],
                                         "draw": {int(r[f"n{i}"]) for i in range(1, 21)}}
    missing = [g for g in want if g not in games]
    if missing:
        sys.exit(f"Missing test games: {missing[:10]} ... run keno first.")

    results = {}
    for name, ranking in frozen["policies"].items():
        res = {"sizes": {}}
        for k in SIZES:
            ticket = set(ranking[:k])
            hits = [len(ticket & games[g]["draw"]) for g in want]
            won = sum(pay[k].get(h, 0) for h in hits)
            complete = [g for g, h in zip(want, hits) if h == k]
            pays = [pay[k].get(h, 0) for h in hits]
            res["sizes"][k] = {
                "hits": sum(hits), "hit_rate": round(sum(hits) / (k * n_test), 4),
                "returned": won, "return_per_dollar": round(won / n_test, 4),
                "roi_pct": round((won / n_test - 1) * 100, 1),
                "complete_hits": len(complete), "first_complete": complete[0] if complete else None,
                "best_catch": max(hits), "per_game": hits,
                "replay": replay(k, hits, pays, first),
            }
        res["k10_hit_rate"] = res["sizes"][10]["hit_rate"]
        res["in_luck_band"] = 0.21 <= res["k10_hit_rate"] <= 0.29
        res["reward"] = round(reward({k: v["replay"] for k, v in res["sizes"].items()}), 3)
        results[name] = res

    w = json.load(open(path("ledger", "weights.json")))
    group = [n for n in results if n not in OUTSIDE]       # GRPO group: every policy but the blend / Agreement
    rewards = {n: results[n]["reward"] for n in group}
    mean = sum(rewards.values()) / len(rewards)
    sd = math.sqrt(sum((x - mean) ** 2 for x in rewards.values()) / len(rewards))
    for n in group:
        results[n]["advantage"] = round((rewards[n] - mean) / sd if sd > 0 else 0.0, 3)
    if rid >= POINTS_FROM:
        new, pts = points_weights(rid, {n: {k: v["replay"] for k, v in results[n]["sizes"].items()} for n in group})
        for n in group:
            results[n]["points"] = pts[n]
    else:
        new = averaged_weights(rid, {n: results[n]["advantage"] for n in group})
    for n in new:
        results[n]["weight_before"] = w["weights"].get(n, "")
        results[n]["weight_after"] = new[n]
    # ties on weight go to the higher K10 hit rate
    leader = max(new, key=lambda n: (new[n], results[n]['k10_hit_rate']))
    w.update({"after_round": rid, "weights": new, "on_a_roll": leader,
              "rule": RULE_POINTS if rid >= POINTS_FROM else "softmax of mean advantage over the last 10 scored rounds (rounds a policy missed count as 0)",
              "reward": "rounds with a cutoff win: 1 if any K4-K10 replay ticket got a single prize over $20 in the round, else 0"})
    json.dump(w, open(path("ledger", "weights.json"), "w"), indent=1)

    with open(path("ledger", "leaderboard.csv"), "a", newline="") as f:
        cw = csv.writer(f)
        for n, r in results.items():
            r.setdefault("advantage", ""); r.setdefault("weight_after", "")
            k10 = r["sizes"][10]
            cw.writerow([rid, n, k10["hits"], k10["hit_rate"], r["reward"], "",
                         sum(s["complete_hits"] for s in r["sizes"].values()),
                         min([s["first_complete"] for s in r["sizes"].values() if s["first_complete"] is not None],
                             default=""), r["advantage"], r["weight_after"]])

    ht = None
    if "heads_tails" in frozen:
        ptab = json.load(open(path("keno_qld_paytable.json")))["heads_tails"]
        outcome = []
        for g in want:
            lo = sum(1 for n in games[g]["draw"] if n <= 40)
            outcome.append("heads" if lo > 10 else "tails" if lo < 10 else "evens")
        ht = {"outcomes": outcome, "counts": {k: outcome.count(k) for k in ("heads", "tails", "evens")},
              "bets": {}}
        for name, side in frozen["heads_tails"]["bets"].items():
            wins = [g for g, o in zip(want, outcome) if o == side]
            ret = len(wins) * ptab[side]
            ht["bets"][name] = {"side": side, "wins": len(wins), "win_games": wins,
                                "returned": ret, "net": ret - n_test}

    out = {"round": rid, "heads_tails": ht, "scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "test_games": [first, last], "n_test": n_test, "group_mean_reward": round(mean, 4),
           "group_sd_reward": round(sd, 4), "on_a_roll": leader, "results": results}
    json.dump(out, open(path("rounds", rid, "scores.json"), "w"), indent=1)
    sizes = size_table(scored_rounds(rid)[-WINDOW:], pay)
    json.dump({"after_round": rid, "window": WINDOW, "policies": sizes},
              open(path("ledger", "sizes.json"), "w"), indent=1)

    if ht:
        print("  Heads/Tails:", ht["counts"], {n: (b["side"], b["net"]) for n, b in ht["bets"].items()})
    print(f"{rid}: games {first}-{last}, reward = 1 if any K4-K10 ticket hit the cutoff, "
          f"group mean {mean:.2f}, sd {sd:.2f}, on a roll: {leader}")
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["reward"]):
        pk = frozen["primary_ticket_size"]
        s6, s10 = r["sizes"][pk], r["sizes"][10]
        print(f"  {n:24s} cutoff round {r['reward']}  replay avg ${sum(v['replay']['net'] for v in r['sizes'].values())/len(r['sizes']):+6.2f}  K{pk} ${s6['returned']:>5} back ({s6['roi_pct']:+.0f}%)  "
              f"K10 {s10['hits']}/{10*n_test} = {s10['hit_rate']:.1%}  "
              f"complete {[ (k, s['complete_hits']) for k, s in r['sizes'].items() if s['complete_hits']]}  "
              + (f"best K{pk} catch {s6['best_catch']}  adv {r['advantage']:+.2f}  w {r['weight_before'] or 0:.3f}->{r['weight_after']:.3f}"
                 if n in new else f"best K{pk} catch {s6['best_catch']}  (outside GRPO)"))


if __name__ == "__main__":
    main(sys.argv[1])
