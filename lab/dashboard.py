#!/usr/bin/env python3
"""Build the round dashboard page from the frozen record (and scores, once scored).

usage: python lab/dashboard.py R003 OUT_HTML
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(rid, out_path):
    rdir = os.path.join(ROOT, "rounds", rid)
    d = json.load(open(os.path.join(rdir, "frozen.json")))
    prev_id = f"R{int(rid[1:]) - 1:03d}"
    prev = {r["policy"]: {"k6": float(r["k6_return"]), "k10": float(r["k10_hit_rate"])}
            for r in csv.DictReader(open(os.path.join(ROOT, "ledger", "leaderboard.csv")))
            if r["round"] == prev_id}
    data = {"round": rid, "frozen": d["frozen_utc"], "train": d["training_games"],
            "test": d["test_games"], "close": d["test_expected_close_utc"],
            "policies": d["policies"], "weights": d["weights_at_freeze"], "roll": d["on_a_roll"],
            "prev": prev, "scan": d["window_vs_chance"],
            "sha": open(os.path.join(rdir, "frozen.sha256")).read().strip(), "scores": None}
    sp = os.path.join(rdir, "scores.json")
    if os.path.exists(sp):
        s = json.load(open(sp))
        data["scores"] = {p: {"k6": r["reward"], "k10": r["k10_hit_rate"], "band": r["in_luck_band"],
                              "adv": r["advantage"], "w": r["weight_after"],
                              "k10hits": r["sizes"]["10"]["hits"],
                              "complete": [[int(k), x["first_complete"]] for k, x in r["sizes"].items()
                                           if x["complete_hits"]]}
                          for p, r in s["results"].items()}
        data["next_roll"], data["mean"] = s["on_a_roll"], s["group_mean_reward"]
    tpl = open(os.path.join(ROOT, "lab", "dashboard.tpl.html")).read()
    open(out_path, "w").write(tpl.replace("__DATA__", json.dumps(data)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
