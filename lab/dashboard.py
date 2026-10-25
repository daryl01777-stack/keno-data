#!/usr/bin/env python3
"""Build the round dashboard page from the frozen record (and scores, once scored).

usage: python lab/dashboard.py R003 OUT_HTML
"""
import csv, json, math, os, sys

from replay import chance, replay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW = "Retro pick"
AGREE = "Agreement"
OUTSIDE = ("Weighted blend", SHADOW, AGREE)     # scored and shown, but not in the group averages


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
            "ideal": d.get("ideal_sizes", {})}
    shp = os.path.join(rdir, "shadow.json")       # Retro pick: shadow ticket outside the weights and blend
    sh = json.load(open(shp)) if os.path.exists(shp) else d.get("shadow")
    if sh:
        data["policies"] = dict(d["policies"], **sh["policies"])
        data["ideal"] = dict(data["ideal"], **{SHADOW: {"size": sh["ideal_size"], "rule": sh["rule"]}})
        data["shadow"] = {"rule": sh["rule"], "late": bool(sh.get("added_after_freeze")),
                          "frozen": sh.get("frozen_utc", d["frozen_utc"]),
                          "sha": open(os.path.join(rdir, "shadow.sha256")).read().strip() if os.path.exists(shp) else None}
    ag = d.get("agreement")                      # Agreement: shadow ticket outside the weights and blend (from R018)
    if ag:
        data["policies"] = dict(data["policies"], **ag["policies"])
        data["ideal"] = dict(data["ideal"], **{AGREE: {"size": ag["ideal_size"], "own": ag["ideal_rounds"]}})
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
        ssp = os.path.join(rdir, "shadow_scores.json")
        if sh and os.path.exists(ssp):
            x = json.load(open(ssp))
            data["scores"][SHADOW] = {"k10": x["k10_hit_rate"], "adv": "", "w": "", "k10hits": x["k10_hits"],
                                      "complete": [], "sizes": {k: ticket(int(k), v) for k, v in x["sizes"].items()}}
        asp = os.path.join(rdir, "agreement_scores.json")
        if ag and os.path.exists(asp):
            x = json.load(open(asp))
            data["scores"][AGREE] = {"k10": x["k10_hit_rate"], "adv": "", "w": "", "k10hits": x["k10_hits"],
                                     "complete": [], "sizes": {k: ticket(int(k), v) for k, v in x["sizes"].items()}}
        data["next_roll"], data["mean"] = s["on_a_roll"], s["group_mean_reward"]
        for v in data["scores"].values():      # average replay result across K4-K10 (the reward)
            v["rr"] = sum(x["replay"]["net"] for x in v["sizes"].values()) / len(v["sizes"])
            v["big"] = sum(x["replay"]["end"].startswith("big win") for x in v["sizes"].values())
        group = [v["rr"] for p, v in data["scores"].items() if p not in OUTSIDE]
        data["rr_mean"] = sum(group) / len(group)
        bg = [v["big"] for p, v in data["scores"].items() if p not in OUTSIDE]
        data["big_mean"] = sum(bg) / len(bg)
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
    sp = os.path.join(ROOT, "ledger", "sizes.json")
    sizes = json.load(open(sp)) if os.path.exists(sp) else None
    for rid_, d in rounds.items():           # previous round's result, judged the same way
        before = rounds.get(f"R{int(rid_[1:]) - 1:03d}")
        d["prev"] = ({p: {"rr": v["rr"], "k10": v["k10"]} for p, v in before["scores"].items()}
                     if before and before["scores"] else {})
    tpl = open(os.path.join(ROOT, "lab", "dashboard.tpl.html")).read()
    open(out_path, "w").write(tpl.replace("__DATA__", json.dumps(rounds))
                              .replace("__ODDS__", json.dumps({"odds": odds, "pay": pay, "sizes": sizes, "chance": {k: round(v, 4) for k, v in chance().items()}})))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
