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


def big(r):
    return r["end"].startswith("big win")


def reward(reps):
    """Policy reward: 1 if any of its K4-K10 tickets reached the take-win cutoff (a single prize
    over $20) this round, else 0 (from R018 scoring, Daryl, 25 Oct). The sizes are nested, so one hot
    draw used to count up to 7 times; scoring the round once rewards steady rounds over one lucky one.
    (R008-R017 scored the count of tickets, 0-7.)"""
    return int(any(big(r) for r in reps.values()))


def chance(pay=None):
    """{k: probability that a random K-k replay ticket reaches the take-win cutoff}, worked out
    exactly from the hypergeometric catch odds (bank is whole dollars, so a DP over the bank)."""
    from math import comb
    pay = pay or paytable()
    out = {}
    for k, t in pay.items():
        p = {}
        for h in range(k + 1):
            w = t.get(h, 0)
            p[w] = p.get(w, 0) + comb(20, h) * comb(60, k - h) / comb(80, k)
        pbig = sum(v for w, v in p.items() if w > BIG)
        small = {w: v for w, v in p.items() if w <= BIG}
        bank, won = {0: 1.0}, 0.0
        for _ in range(BUY):                          # the $20 ticket
            nb = {}
            for b, q in bank.items():
                won += q * pbig
                for w, v in small.items():
                    nb[b + w] = nb.get(b + w, 0) + q * v
            bank = nb
        bank = {b: q for b, q in bank.items() if b >= 1}
        for _ in range(BUY, 2 * BUY):                 # winnings replayed at $1 a game
            nb = {}
            for b, q in bank.items():
                won += q * pbig
                for w, v in small.items():
                    if b + w - 1 >= 1:
                        nb[b + w - 1] = nb.get(b + w - 1, 0) + q * v
            bank = nb
        out[k] = won
    return out


def scored_rounds(upto=None):
    """Scored rounds in order (optionally only up to and including `upto`)."""
    rdir = os.path.join(ROOT, "rounds")
    ids = sorted(r for r in os.listdir(rdir) if os.path.exists(os.path.join(rdir, r, "scores.json")))
    return [r for r in ids if upto is None or r <= upto]


def size_table(rounds, pay=None):
    """For each policy and size over the given rounds: cutoff wins, how that compares with chance
    (x_chance = observed rate / chance rate), the average replay net, and the best ("ideal") size:
    the size that reached the cutoff most often (ties: higher average net, then the smaller ticket).
    Raw counts, not x_chance, so one lucky hit on a rarely-winning size can't make it the ideal."""
    pay = pay or paytable()
    odds = chance(pay)
    acc = {}
    for rid in rounds:
        reps = round_replays(json.load(open(os.path.join(ROOT, "rounds", rid, "scores.json"))), pay)
        for p, by in reps.items():
            for k, r in by.items():
                acc.setdefault(p, {}).setdefault(k, []).append(r)
    table = {}
    for p, by in acc.items():
        n = len(next(iter(by.values())))
        avg = {k: round(sum(r["net"] for r in v) / len(v), 2) for k, v in by.items()}
        hits = {k: sum(big(r) for r in v) for k, v in by.items()}
        x = {k: round(hits[k] / n / odds[k], 2) for k in by}
        best = max(sorted(by), key=lambda k: (hits[k], avg[k], -k))
        table[p] = {"big": hits, "x_chance": x, "avg_net": avg, "rounds": n, "ideal": best}
    return table
