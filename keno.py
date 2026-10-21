#!/usr/bin/env python3
"""
keno.py - collect Queensland Keno draw results. Runs in Termux on your phone.

Main use - leave this running and it keeps draws.csv in your GitHub repo up to date:
  python keno.py collect         backfill the last 100 games, then add each new game

Other commands:
  python keno.py now             show the latest completed game
  python keno.py last 100        save the last 100 completed games to a file
  python keno.py range 237 276   save a run of games by number (handles 999 -> 0)
  python keno.py watch 40        save the next 40 games to a file as they're drawn
  python keno.py probe           test the history request and print the raw reply
"""
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

BASE = "https://api-info-qld.keno.com.au"
JUR = "QLD"
BNE = timezone(timedelta(hours=10))   # Queensland has no daylight saving
CYCLE = 1000                          # game numbers run 0-999, then wrap to 0
WINDOW = timedelta(hours=30)          # a game number comes round again ~44 h later
CHUNK = 50                            # games per history request
POLL = 20                             # seconds between live checks
BACKFILL = 100                        # games collect fills in when it starts
PUSH_EVERY = 3                        # push to GitHub after this many new games...
PUSH_MAX_WAIT = 600                   # ...or this many seconds, whichever comes first

_here = os.path.dirname(os.path.abspath(__file__))
REPO = _here if os.path.isdir(os.path.join(_here, ".git")) else os.path.expanduser("~/keno-data")
DATA = os.path.join(REPO, "draws.csv")


# ---------- talking to the feed ----------

def fetch(path, params):
    """GET a feed URL with curl (proven to work from the phone) and return JSON."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE}{path}?{query}"
    r = subprocess.run(
        ["curl", "-sS", "--max-time", "20", "-w", "\n%{http_code}", url],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"curl failed: {r.stderr.strip()}\n  {url}")
    body, _, code = r.stdout.rpartition("\n")
    if code.strip() != "200":
        raise RuntimeError(f"HTTP {code.strip()} from {url}\n  {body[:400]}")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError(f"reply wasn't JSON from {url}\n  {body[:400]}")


def parse_time(s):
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def to_game(d):
    """Turn one game object into {game, closed, draw}, or None if it isn't a full draw."""
    if not isinstance(d, dict) or "game-number" not in d:
        return None
    try:
        nums = [int(n) for n in d.get("draw") or []]
        number = int(d["game-number"])
    except (TypeError, ValueError):
        return None
    if len(nums) != 20 or len(set(nums)) != 20 or not all(1 <= n <= 80 for n in nums):
        return None
    closed = parse_time(d.get("closed"))
    if closed is None:
        return None
    return {"game": number, "closed": closed, "draw": nums}


def find_games(obj, out=None):
    """Collect every complete game anywhere in a reply, whatever its layout."""
    if out is None:
        out = []
    if isinstance(obj, dict):
        g = to_game(obj)
        if g:
            out.append(g)
        else:
            for v in obj.values():
                find_games(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_games(v, out)
    return out


def latest(tries=6):
    """The most recent completed game. Waits out a draw that's still in progress."""
    for attempt in range(tries):
        data = fetch("/v2/games/kds", {"jurisdiction": JUR})
        g = to_game(data.get("current"))
        if g:
            return g
        if attempt < tries - 1:
            time.sleep(POLL)
    raise RuntimeError("couldn't get a completed game from the live feed")


# ---------- game-number arithmetic ----------

def back_from(last, n):
    """n game numbers ending at `last`, oldest first."""
    return [(last - i) % CYCLE for i in range(n - 1, -1, -1)]


def span(start, end):
    """Game numbers from start to end inclusive, across the 999 -> 0 wrap."""
    count = (end - start) % CYCLE + 1
    return [(start + i) % CYCLE for i in range(count)]


def runs(numbers):
    """Split a sequence into runs that don't cross the wrap, each at most CHUNK long."""
    out, cur = [], [numbers[0]]
    for a, b in zip(numbers, numbers[1:]):
        if b == a + 1 and len(cur) < CHUNK:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


# ---------- history ----------

def history(numbers, ref_time, verbose=True):
    """Fetch these game numbers, keeping only games that closed in the 30 h before ref_time.
    Tries the Brisbane date of ref_time first, then the day before."""
    want = set(numbers)
    got = {}
    today = ref_time.astimezone(BNE).date()
    dates = [today, today - timedelta(days=1)]
    for run in runs(numbers):
        for d in dates:
            missing = [n for n in run if n not in got]
            if not missing:
                break
            start, count = missing[0], len(span(missing[0], missing[-1]))
            for page in range(1, 6):
                params = {
                    "jurisdiction": JUR,
                    "starting_game_number": start,
                    "number_of_games": count,
                    "date": d.isoformat(),
                    "page_size": count,
                    "page_number": page,
                }
                try:
                    data = fetch("/v2/info/history", params)
                except RuntimeError as e:
                    if verbose:
                        print(f"  note: {e}")
                    break
                new = 0
                for g in find_games(data):
                    ok_time = ref_time - WINDOW <= g["closed"] <= ref_time + timedelta(minutes=5)
                    if g["game"] in want and g["game"] not in got and ok_time:
                        got[g["game"]] = g
                        new += 1
                if new == 0 or all(n in got for n in run):
                    break
                time.sleep(1)
            time.sleep(1)
    return got


# ---------- saving ----------

def out_dir():
    if os.path.isdir(os.path.join(REPO, ".git")):
        path = os.path.join(REPO, "exports")
    else:
        downloads = os.path.expanduser("~/storage/downloads")
        base = downloads if os.path.isdir(downloads) else os.path.expanduser("~")
        path = os.path.join(base, "keno")
    os.makedirs(path, exist_ok=True)
    return path


def save(games, label):
    stamp = datetime.now(BNE).strftime("%Y%m%d-%H%M")
    name = f"keno_{label}_{games[0]['game']}-{games[-1]['game']}_{stamp}.csv"
    path = os.path.join(out_dir(), name)
    write_csv(path, games)
    return path


def write_csv(path, games):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "closed_utc", "closed_brisbane"] + [f"n{i}" for i in range(1, 21)])
        for g in games:
            w.writerow([
                g["game"],
                g["closed"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                g["closed"].astimezone(BNE).strftime("%Y-%m-%d %H:%M:%S"),
            ] + g["draw"])


def report(numbers, got, label):
    games = [got[n] for n in numbers if n in got]
    missing = [n for n in numbers if n not in got]
    if not games:
        print("No games came back. Run:  python keno.py probe  and paste the output to Claude.")
        return None
    path = save(games, label)
    t0 = games[0]["closed"].astimezone(BNE).strftime("%d %b %H:%M")
    t1 = games[-1]["closed"].astimezone(BNE).strftime("%d %b %H:%M")
    print(f"Saved {len(games)} of {len(numbers)} games: {games[0]['game']} to {games[-1]['game']}"
          f" ({t0} to {t1} Brisbane)")
    if missing:
        shown = ", ".join(map(str, missing[:15])) + (" ..." if len(missing) > 15 else "")
        print(f"Missing {len(missing)}: {shown}")
    print(f"File: {path}")
    return path


# ---------- commands ----------

def cmd_now():
    g = latest()
    t = g["closed"].astimezone(BNE).strftime("%d %b %H:%M:%S")
    print(f"Game {g['game']} closed {t} Brisbane")
    print(" ".join(str(n) for n in sorted(g["draw"])))


def cmd_last(n):
    g = latest()
    numbers = back_from(g["game"], n)
    print(f"Latest completed game is {g['game']}. Fetching {numbers[0]} to {numbers[-1]} ...")
    got = history(numbers[:-1], g["closed"]) if n > 1 else {}
    got[g["game"]] = g
    report(numbers, got, f"last{n}")


def cmd_range(start, end):
    numbers = span(start, end)
    newest = latest()   # anchor to the feed's clock, not the phone's
    print(f"Fetching games {start} to {end} ({len(numbers)} games); latest drawn is {newest['game']} ...")
    got = history(numbers, newest["closed"])
    if newest["game"] in numbers:
        got.setdefault(newest["game"], newest)
    path = report(numbers, got, "range")
    if path and len(got) < len(numbers):
        print("If the missing games haven't been drawn yet, wait and run this again.")


def cmd_watch(n):
    first = latest()
    print(f"Latest completed game is {first['game']}. Collecting the next {n} as they're drawn.")
    print("Keep Termux open (run termux-wake-lock first so the phone doesn't stop it).")
    collected, last_num, path = [], first["game"], None
    while len(collected) < n:
        time.sleep(POLL)
        try:
            data = fetch("/v2/games/kds", {"jurisdiction": JUR})
        except RuntimeError as e:
            print(f"  note: {e}")
            continue
        g = to_game(data.get("current"))
        if not g or g["game"] == last_num:
            continue
        expected = (last_num + 1) % CYCLE
        if g["game"] != expected:
            gap = span(expected, (g["game"] - 1) % CYCLE)
            print(f"  missed {len(gap)} game(s) while away, fetching them from history ...")
            filled = history(gap, g["closed"])
            for num in gap:
                if num in filled:
                    collected.append(filled[num])
                else:
                    print(f"  couldn't recover game {num}")
        collected.append(g)
        last_num = g["game"]
        collected = collected[:n]
        if path is None:
            path = os.path.join(out_dir(), f"keno_watch_from{expected}_"
                                f"{datetime.now(BNE).strftime('%Y%m%d-%H%M')}.csv")
        write_csv(path, collected)
        print(f"  game {g['game']} saved ({len(collected)}/{n})")
    print(f"Done. File: {path}")


# ---------- collector: keeps draws.csv in the GitHub repo current ----------

def stamp(g):
    return g["closed"].strftime("%Y-%m-%dT%H:%M:%SZ")


def load_store():
    store = {}
    if os.path.exists(DATA):
        with open(DATA, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    g = {"game": int(row["game"]), "closed": parse_time(row["closed_utc"]),
                         "draw": [int(row[f"n{i}"]) for i in range(1, 21)]}
                except (KeyError, TypeError, ValueError):
                    continue
                if g["closed"]:
                    store[stamp(g)] = g
    return store


def save_store(store):
    games = sorted(store.values(), key=lambda g: g["closed"])
    tmp = DATA + ".tmp"
    write_csv(tmp, games)
    os.replace(tmp, DATA)


def git(*args):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def publish(message):
    """Commit draws.csv and push. Pulls first so anything Claude pushed is kept."""
    git("add", "draws.csv")
    ok, out = git("commit", "-q", "-m", message)
    if not ok and "nothing to commit" not in out:
        print(f"  note: commit failed: {out[:200]}")
    ok, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    branch = branch if ok and branch else "main"
    ok, out = git("ls-remote", "--exit-code", "--heads", "origin", branch)
    if ok:  # remote branch exists, so bring in any changes before pushing
        ok, out = git("pull", "-q", "--rebase", "origin", branch)
        if not ok:
            git("rebase", "--abort")
            print(f"  note: pull failed, will retry: {out[:200]}")
            return False
    ok, out = git("push", "-q", "origin", f"HEAD:{branch}")
    if not ok:
        print(f"  note: push failed, will retry: {out[:200]}")
    return ok


def recent_numbers(store, ref_time):
    return {g["game"] for g in store.values() if ref_time - WINDOW <= g["closed"] <= ref_time}


def cmd_collect():
    if not os.path.isdir(os.path.join(REPO, ".git")):
        sys.exit(f"No GitHub repo at {REPO}. Do the one-time setup first.")
    store = load_store()
    first = latest()
    store[stamp(first)] = first
    want = back_from(first["game"], BACKFILL)
    have = recent_numbers(store, first["closed"])
    missing = [n for n in want if n not in have]
    if missing:
        print(f"Backfilling {len(missing)} recent games ...")
        got = history(missing, first["closed"], verbose=False)
        for g in got.values():
            store[stamp(g)] = g
        if got:
            print(f"Backfilled {len(got)} of {len(missing)}.")
        else:
            print("History request didn't work, so collecting forward only "
                  f"({BACKFILL} games takes about 4.5 hours). Tell Claude.")
    save_store(store)
    publish(f"draws to game {first['game']}")
    print(f"Collecting from game {first['game']}. {len(store)} games on file. "
          "Leave this running (termux-wake-lock keeps the phone from stopping it).")

    last, pending, last_push = first, 0, time.time()
    while True:
        time.sleep(POLL)
        try:
            data = fetch("/v2/games/kds", {"jurisdiction": JUR})
        except RuntimeError as e:
            print(f"  note: {str(e).splitlines()[0]}")
            continue
        g = to_game(data.get("current"))
        if g and g["game"] != last["game"] and g["closed"] > last["closed"]:
            expected = (last["game"] + 1) % CYCLE
            if g["game"] != expected:
                gap = span(expected, (g["game"] - 1) % CYCLE)
                filled = history(gap, g["closed"], verbose=False) if len(gap) <= 500 else {}
                for x in filled.values():
                    store[stamp(x)] = x
                print(f"  missed {len(gap)} game(s) while away, recovered {len(filled)}")
            store[stamp(g)] = g
            save_store(store)
            pending += 1
            print(f"  {g['closed'].astimezone(BNE).strftime('%H:%M')}  game {g['game']}")
            last = g
        if pending and (pending >= PUSH_EVERY or time.time() - last_push >= PUSH_MAX_WAIT):
            if publish(f"draws to game {last['game']}"):
                pending, last_push = 0, time.time()


def cmd_probe():
    g = latest()
    print(f"Live feed OK. Latest completed game: {g['game']}")
    start = (g["game"] - 4) % CYCLE
    d = g["closed"].astimezone(BNE).date().isoformat()
    params = {"jurisdiction": JUR, "starting_game_number": start, "number_of_games": 5,
              "date": d, "page_size": 5, "page_number": 1}
    print(f"Asking history for 5 games from {start} on {d} ...")
    try:
        data = fetch("/v2/info/history", params)
    except RuntimeError as e:
        print(f"History request failed:\n{e}")
        return
    games = find_games(data)
    print(f"Parsed {len(games)} game(s): {[x['game'] for x in games]}")
    print("Raw reply (first 1500 characters):")
    print(json.dumps(data)[:1500])


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, args = argv[0], argv[1:]
    try:
        if cmd == "collect":
            cmd_collect()
        elif cmd == "now":
            cmd_now()
        elif cmd == "last":
            n = int(args[0]) if args else 100
            if not 1 <= n <= 500:
                sys.exit("Pick between 1 and 500 games.")
            cmd_last(n)
        elif cmd == "range" and len(args) == 2:
            cmd_range(int(args[0]) % CYCLE, int(args[1]) % CYCLE)
        elif cmd == "watch":
            cmd_watch(int(args[0]) if args else 40)
        elif cmd == "probe":
            cmd_probe()
        else:
            print(__doc__)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")
    except KeyboardInterrupt:
        sys.exit("\nStopped.")


if __name__ == "__main__":
    main(sys.argv[1:])
