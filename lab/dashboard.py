#!/usr/bin/env python3
"""Build the round dashboard page from the frozen record (and scores, once scored).

usage: python lab/dashboard.py R003 OUT_HTML
"""
import csv, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def replay(k, hits, won, first, buy=20, target=40):
    """A $20 ticket for the first 20 test games, then its winnings are replayed at $1 a game.
    It stops when the bank runs out (bust), the ticket catches all k numbers, the bank reaches
    $40 (twice the cost), or the test window ends."""
    bank, played = sum(won[:buy]), buy
    full = [i for i in range(buy) if hits[i] == k]
    if full:
        end = "caught all"
    elif bank >= target:
        end = "reached $40"
    elif bank < 1:
        end = "bust"
    else:
        end = "window ended"
        for i in range(buy, len(won)):
            bank += won[i] - 1
            played += 1
            if hits[i] == k:
                end, full = "caught all", [i]
                break
            if bank >= target:
                end = "reached $40"
                break
            if bank < 1:
                end = "bust"
                break
    return {"bank": bank, "net": bank - buy, "games": played, "end": end,
            "last_game": (first + played - 1) % 1000}


def round_data(rid):
    rdir = os.path.join(ROOT, "rounds", rid)
    d = json.load(open(os.path.join(rdir, "frozen.json")))
    prev_id = f"R{int(rid[1:]) - 1:03d}"
    prev = {r["policy"]: {"k6": float(r["reward"]), "k10": float(r["k10_hit_rate"])}
            for r in csv.DictReader(open(os.path.join(ROOT, "ledger", "leaderboard.csv")))
            if r["round"] == prev_id}
    data = {"round": rid, "frozen": d["frozen_utc"], "train": d["training_games"],
            "test": d["test_games"], "close": d["test_expected_close_utc"],
            "policies": d["policies"], "primary": d["primary_ticket_size"],
            "weights": d.get("weights_at_freeze") or {p: round(1 / 7, 6) for p in d["policies"]},
            "roll": d.get("on_a_roll") or "none yet (first round)",
            "prev": prev, "scan": d["window_vs_chance"],
            "sha": open(os.path.join(rdir, "frozen.sha256")).read().strip(), "scores": None,
            "ht": d.get("heads_tails"), "ht_scores": None}
    sp = os.path.join(rdir, "scores.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        pay = {int(k): {int(h): v for h, v in t.items()}
               for k, t in json.load(open(os.path.join(ROOT, "keno_qld_paytable.json")))["payouts"].items()}
        first, n = s["test_games"][0], s["n_test"]

        def ticket(k, x):
            won = [pay[k].get(h, 0) for h in x["per_game"]]
            return {"net20": sum(won[:20]) - 20, "net40": sum(won) - n,
                    "replay": replay(k, x["per_game"], won, first),
                    "pays": [[(first + i) % 1000, h, w] for i, (h, w) in enumerate(zip(x["per_game"], won)) if w]}
        data["scores"] = {p: {"k6": r["reward"], "k10": r["k10_hit_rate"], "band": r["in_luck_band"],
                              "adv": r["advantage"], "w": r["weight_after"],
                              "k10hits": r["sizes"]["10"]["hits"],
                              "complete": [[int(k), x["first_complete"]] for k, x in r["sizes"].items()
                                           if x["complete_hits"]],
                              "sizes": {k: ticket(int(k), x) for k, x in r["sizes"].items()}}
                          for p, r in s["results"].items()}
        data["next_roll"], data["mean"] = s["on_a_roll"], s["group_mean_reward"]
        data["ht_scores"] = s.get("heads_tails")
        # catch table: how often each catch landed across every ticket this round, per size
        data["catches"] = {k: [0] * (int(k) + 1) for k in s["results"][next(iter(s["results"]))]["sizes"]}
        for r in s["results"].values():
            for k, x in r["sizes"].items():
                for h in x["per_game"]:
                    data["catches"][k][h] += 1
    return data


def main(rid, out_path):
    """Build the page with every round up to rid; it opens on rid."""
    last = int(rid[1:])
    rounds = {r: round_data(r) for r in sorted(os.listdir(os.path.join(ROOT, "rounds")))
              if r.startswith("R") and int(r[1:]) <= last}
    # chance of catching h of a k-number ticket in one game: C(20,h) C(60,k-h) / C(80,k)
    odds = {k: [math.comb(20, h) * math.comb(60, k - h) / math.comb(80, k) for h in range(k + 1)]
            for k in range(4, 11)}
    pay = json.load(open(os.path.join(ROOT, "keno_qld_paytable.json")))["payouts"]
    tpl = open(os.path.join(ROOT, "lab", "dashboard.tpl.html")).read()
    open(out_path, "w").write(tpl.replace("__DATA__", json.dumps(rounds))
                              .replace("__ODDS__", json.dumps({"odds": odds, "pay": pay})))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
