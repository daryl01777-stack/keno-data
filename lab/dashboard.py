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
            "ideal": d.get("ideal_sizes", {}), "seats": d.get("agreement_seats", {})}
    shp = os.path.join(rdir, "shadow.json")       # Retro pick: shadow ticket outside the weights and blend
    sh = json.load(open(shp)) if os.path.exists(shp) else d.get("shadow")
    if sh:
        data["policies"] = dict(d["policies"], **sh["policies"])
        data["ideal"] = dict(data["ideal"], **{SHADOW: {"size": sh["ideal_size"], "rule": sh["rule"]}})
        data["shadow"] = {"rule": sh["rule"], "late": bool(sh.get("added_after_freeze")),
                          "frozen": sh.get("frozen_utc", d["frozen_utc"]),
                          "sha": open(os.path.join(rdir, "shadow.sha256")).read().strip() if os.path.exists(shp) else None}
    ap = os.path.join(rdir, "aipick.json")         # AI pick: Claude's own K4 each round from R040 (Daryl, 30 Oct)
    if os.path.exists(ap):
        import aipick
        a = json.load(open(ap))
        data["ai"] = {"ticket": a["ticket"], "why": a.get("why", ""),
                      "result": aipick.result(int(rid[1:]), a["ticket"]) if os.path.exists(os.path.join(rdir, "scores.json")) else None}
    bp = os.path.join(rdir, "aibuilds.json")       # AI builds: the 21 AI pick ideas as fixed rules (Daryl, 5 Nov)
    if os.path.exists(bp):
        bl = os.path.join(ROOT, "ledger", "aibuilds.csv")
        res = {r["build"]: r for r in csv.DictReader(open(bl)) if r["round"] == rid} if os.path.exists(bl) else {}
        data["builds"] = {n: {"t": t, "res": res.get(n)} for n, t in json.load(open(bp))["builds"].items()}
        if os.path.exists(os.path.join(rdir, "scores.json")) and res:   # per-size replay for the AI heatmap detail
            import aipick, replay as rpl
            pay_ = rpl.paytable(); first_ = json.load(open(os.path.join(rdir, "scores.json")))["test_games"][0]
            for n, b in data["builds"].items():
                b["rp"] = {}
                for k in range(4, 11):
                    h, _ = aipick.hits(int(rid[1:]), b["t"][:k])
                    x = rpl.replay(k, h, [pay_[k].get(c, 0) for c in h], first_)
                    b["rp"][k] = {"end": x["end"], "net": x["net"]}
    vp = os.path.join(rdir, "void.json")           # void round: results never became available (Daryl, 9 Nov)
    if os.path.exists(vp):
        data["void"] = json.load(open(vp)).get("reason", "")
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
    lp = os.path.join(ROOT, "ledger", "aibuilds.json")
    if os.path.exists(lp):                    # standings after each scored round, for the builds panel
        led = json.load(open(lp)); bc = os.path.join(ROOT, "ledger", "aibuilds.csv")
        rows = list(csv.DictReader(open(bc))) if os.path.exists(bc) else []
        after = {}
        for r in rows:
            after.setdefault(r["round"], {})[r["build"]] = int(r["points"])
        # carried hindsight points, before any live round
        base = {b: led["points"][b] - sum({"win": 5, "survive": 1, "bust": -1}[r["outcome"]] for r in rows if r["build"] == b) for b in led["points"]}
        for rid_, d in rounds.items():
            if "builds" in d:
                prev = [r for r in sorted(after) if r < rid_]
                d["build_pts"] = after[prev[-1]] if prev else base
                d["build_pts_after"] = after.get(rid_)
                # live-only standings (Daryl, 11 Nov): 100 at the first live round, same +5/+1/-1, no hindsight points
                upto = rid_ if rid_ in after else (prev[-1] if prev else "")
                d["build_live"] = {b: 100 + sum({"win": 5, "survive": 1, "bust": -1}[r["outcome"]] for r in rows if r["build"] == b and r["round"] <= upto) for b in d["builds"]}
                d["build_live_n"] = len({r["round"] for r in rows if r["round"] <= upto})
    tpl = open(os.path.join(ROOT, "lab", "dashboard.tpl.html")).read()
    open(out_path, "w").write(tpl.replace("__DATA__", json.dumps(rounds))
                              .replace("__ODDS__", json.dumps({"odds": odds, "pay": pay, "sizes": sizes, "chance": {k: round(v, 4) for k, v in chance().items()}})))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
