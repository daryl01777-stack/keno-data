"""The headline ticket result and the history it feeds into the weights.

A ticket is bought for the first 20 test games ($20). Its winnings are then replayed at $1 a
game until the bank runs out (bust), a single game pays more than $20 (a big win: 4/4, 5/5,
5/6+, 6/7+, 6/8+, 7/9+, 7/10+; a $20 prize would only buy the rest of the window, so it does
not count), or the 40-game test window ends (time out). A big win in the first 20 games ends
the ticket there.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUY, BIG, WINDOW = 20, 20, 10      # a big win pays more than BIG dollars in one game


def paytable():
    return {int(k): {int(h): v for h, v in t.items()}
            for k, t in json.load(open(os.path.join(ROOT, "keno_qld_paytable.json")))["payouts"].items()}


def replay(k, hits, won, first):
    bank, played = sum(won[:BUY]), BUY
    big = [i for i in range(BUY) if won[i] > BIG]
    if big:
        end = f"big win {hits[big[0]]}/{k}"
    elif bank < 1:
        end = "bust"
    else:
        end = "timed out"
        for i in range(BUY, len(won)):
            bank += won[i] - 1
            played += 1
            if won[i] > BIG:
                end = f"big win {hits[i]}/{k}"
                break
            if bank < 1:
                end = "bust"
                break
    return {"bank": bank, "net": bank - BUY, "games": played, "end": end,
            "last_game": (first + played - 1) % 1000}


def round_replays(scores, pay=None):
    """{policy: {k: replay}} for a scores.json record, worked out from its per-game catches."""
    pay = pay or paytable()
    first = scores["test_games"][0]
    out = {}
    for p, r in scores["results"].items():
        out[p] = {}
        for k, x in r["sizes"].items():
            k = int(k)
            won = [pay[k].get(h, 0) for h in x["per_game"]]
            out[p][k] = replay(k, x["per_game"], won, first)
    return out


def reward(reps):
    """Policy reward: average replay net across K4-K10, in dollars per $20 ticket."""
    return sum(r["net"] for r in reps.values()) / len(reps)


def scored_rounds(upto=None):
    """Scored rounds in order (optionally only up to and including `upto`)."""
    rdir = os.path.join(ROOT, "rounds")
    ids = sorted(r for r in os.listdir(rdir) if os.path.exists(os.path.join(rdir, r, "scores.json")))
    return [r for r in ids if upto is None or r <= upto]


def size_table(rounds, pay=None):
    """For each policy and size, the average replay net over the given rounds, and the best size."""
    pay = pay or paytable()
    acc = {}
    for rid in rounds:
        reps = round_replays(json.load(open(os.path.join(ROOT, "rounds", rid, "scores.json"))), pay)
        for p, by in reps.items():
            for k, r in by.items():
                acc.setdefault(p, {}).setdefault(k, []).append(r["net"])
    table = {}
    for p, by in acc.items():
        avg = {k: round(sum(v) / len(v), 2) for k, v in by.items()}
        best = max(sorted(avg), key=lambda k: avg[k])      # ties go to the smaller ticket
        table[p] = {"avg_net": avg, "rounds": len(next(iter(by.values()))), "ideal": best}
    return table
