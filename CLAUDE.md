# Keno lab: Queensland Keno GRPO rounds

Daryl's experiment, run for fun. Queensland Keno draws are certified random, so no
policy is expected to beat chance. Report results honestly; never imply an edge.

## Data
- To fetch draws, Claude commits a new timestamp to `requests/fetch.txt` and pushes.
  The GitHub Actions workflow `keno fetch` then runs `python3 keno.py sync 200` on
  GitHub's servers, updates `draws.csv`, and writes the outcome to
  `requests/last_result.txt` (usually within 30 seconds). Keno allows GitHub's servers.
- Backup: Daryl can type `keno` in Termux. It is an alias for
  `cd ~/keno-data && git pull -q --rebase origin main && python keno.py sync`.
- `draws.csv` is the master file. `exports/` holds older one-off files that sync folds in.
- Merge every CSV by `closed_utc`. Game numbers run 0-999 then wrap; a game closes every 160 s.
- The Keno site blocks Anthropic's servers (Akamai "Access Denied"), so never fetch
  results directly from a Claude session. Always go through the workflow or the phone.
- `keno_qld_paytable.json` gives $1 payouts for K4-K10. Average return: K4-K6 about 74-75c,
  K7-K10 about 64-65c per $1. K7-K10 top prizes are growing jackpots, so the table is a floor.

## Round protocol
1. Training window: the latest 100 completed games.
2. `python lab/round.py <merged csv> <last training game> R00N out.json` runs the anomaly
   scan (10 checks against 2,000 random 100-game windows) and builds 7 policies:
   Anomaly composite (primary), Hot, Surge, Pair cluster, Overdue, Random A, Random B.
   Each policy keeps a ranked top 10, nested as tickets K4-K10. K6 is the primary size.
3. Freeze: add `frozen_utc`, `test_games` and the expected close times, write canonical
   JSON (`sort_keys`, compact separators), and post its sha256 in chat before the test
   window opens. Commit it to `rounds/R00N/frozen.json` and `frozen.sha256`.
   The test window is the first 40 games closing at least 4 minutes after the freeze.
4. Scoring, after a fetch once the test games are drawn. For each policy:
   K10 hits and hit rate, K6 return (the reward), return for K4-K10, complete hits, and
   the first game with a complete hit. No random-ticket simulation: use the exact
   luck band instead. For K10 over 40 games, 95% of tickets land between 21% and 29%.
   That band is a label only, never a gate.
5. GRPO update. Reward is the K6 return. Advantage = (reward - group mean) / group sd.
   Each weight is multiplied by exp(0.5 x advantage), then all weights are renormalised.
   Update `ledger/weights.json` and append to `ledger/leaderboard.csv`. The policy with
   the highest weight is flagged "On a roll" for the next round.
6. Commit the scores to `rounds/R00N/scores.json`.

## Reporting
Use the in-chat dashboard layout from round 1:
- header with the round number and status pill
- stat tiles: training games, ticket variants, test games, scored games
- primary prediction card with numbered tiles and a copy button
- category chips (All / Anomaly / Signal / Baseline) and a K4-K10 size selector
- one card per policy
- test status checklist and the fingerprint

Scored rounds add results to each card. Daryl wants honest opinions, never what he
wants to hear.
