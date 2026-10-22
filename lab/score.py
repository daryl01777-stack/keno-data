#!/usr/bin/env python3
"""Score a frozen Keno round and apply the GRPO weight update.

usage: python lab/score.py R001
Reads rounds/<R>/frozen.json, draws.csv, keno_qld_paytable.json, ledger/weights.json.
Writes rounds/<R>/scores.json, updates ledger/weights.json, appends ledger/leaderboard.csv.
"""
import csv, json, math, os, sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = range(4, 11)


def path(*p):
    return os.path.join(ROOT, *p)


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


WINDOW = 10   # weights come from each policy's average advantage over the last 10 scored rounds


def averaged_weights(rid, current):
    """Softmax of each policy's mean advantage over the last WINDOW rounds (this one included)."""
    hist = {}
    with open(path("ledger", "leaderboard.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r["advantage"] and r["round"] != rid and r["policy"] in current:
                hist.setdefault(r["round"], {})[r["policy"]] = float(r["advantage"])
    rounds = [hist[k] for k in sorted(hist)][-(WINDOW - 1):] + [current]
    avg = {n: sum(r[n] for r in rounds) / len(rounds) for n in current}
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
            res["sizes"][k] = {
                "hits": sum(hits), "hit_rate": round(sum(hits) / (k * n_test), 4),
                "returned": won, "return_per_dollar": round(won / n_test, 4),
                "roi_pct": round((won / n_test - 1) * 100, 1),
                "complete_hits": len(complete), "first_complete": complete[0] if complete else None,
                "best_catch": max(hits), "per_game": hits,
            }
        res["k10_hit_rate"] = res["sizes"][10]["hit_rate"]
        res["in_luck_band"] = 0.21 <= res["k10_hit_rate"] <= 0.29
        res["reward"] = res["sizes"][frozen["primary_ticket_size"]]["return_per_dollar"]
        results[name] = res

    w = json.load(open(path("ledger", "weights.json")))
    rewards = {n: results[n]["reward"] for n in w["weights"]}  # GRPO group only
    mean = sum(rewards.values()) / len(rewards)
    sd = math.sqrt(sum((x - mean) ** 2 for x in rewards.values()) / len(rewards))
    for n in w["weights"]:
        results[n]["advantage"] = round((rewards[n] - mean) / sd if sd > 0 else 0.0, 3)
    new = averaged_weights(rid, {n: results[n]["advantage"] for n in w["weights"]})
    for n in new:
        results[n]["weight_before"] = w["weights"][n]
        results[n]["weight_after"] = new[n]
    # ties on weight go to the higher K10 hit rate
    leader = max(new, key=lambda n: (new[n], results[n]['k10_hit_rate']))
    w.update({"after_round": rid, "weights": new, "on_a_roll": leader})
    json.dump(w, open(path("ledger", "weights.json"), "w"), indent=1)

    with open(path("ledger", "leaderboard.csv"), "a", newline="") as f:
        cw = csv.writer(f)
        for n, r in results.items():
            r.setdefault("advantage", ""); r.setdefault("weight_after", "")
            k6, k10 = r["sizes"][6], r["sizes"][10]
            cw.writerow([rid, n, k10["hits"], k10["hit_rate"], k6["return_per_dollar"], k6["roi_pct"],
                         sum(s["complete_hits"] for s in r["sizes"].values()),
                         min([s["first_complete"] for s in r["sizes"].values() if s["first_complete"] is not None],
                             default=""), r["advantage"], r["weight_after"]])

    out = {"round": rid, "scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "test_games": [first, last], "n_test": n_test, "group_mean_reward": round(mean, 4),
           "group_sd_reward": round(sd, 4), "on_a_roll": leader, "results": results}
    json.dump(out, open(path("rounds", rid, "scores.json"), "w"), indent=1)

    print(f"{rid}: games {first}-{last}, group mean K6 return {mean:.3f}, sd {sd:.3f}, on a roll: {leader}")
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["reward"]):
        s6, s10 = r["sizes"][6], r["sizes"][10]
        print(f"  {n:24s} K6 ${s6['returned']:>5} back ({s6['roi_pct']:+.0f}%)  "
              f"K10 {s10['hits']}/{10*n_test} = {s10['hit_rate']:.1%}  "
              f"complete {[ (k, s['complete_hits']) for k, s in r['sizes'].items() if s['complete_hits']]}  "
              + (f"best K6 catch {s6['best_catch']}  adv {r['advantage']:+.2f}  w {r['weight_before']:.3f}->{r['weight_after']:.3f}"
                 if n in new else f"best K6 catch {s6['best_catch']}  (outside GRPO)"))


if __name__ == "__main__":
    main(sys.argv[1])
