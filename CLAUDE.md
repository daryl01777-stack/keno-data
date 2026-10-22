# Keno lab: Queensland Keno GRPO rounds

Daryl's experiment, run for fun. Queensland Keno draws are certified random, so no
policy is expected to beat chance. Report results honestly; never imply an edge.

## Data
- Main route: Daryl types `keno` in Termux on his phone. It is an alias for
  `cd ~/keno-data && git pull -q --rebase origin main && python keno.py sync`, which adds
  any of the last 200 games to `draws.csv` and pushes. Ask him to run it when draws are needed.
- The GitHub Actions workflow `keno fetch` (triggered by committing a new timestamp to
  `requests/fetch.txt`; outcome in `requests/last_result.txt`) worked once on 21 Oct but
  was then blocked by Keno's Akamai protection twice in a row. Treat it as unreliable:
  try it at most once, never retry repeatedly to get past the block, and fall back to the phone.
- `draws.csv` is the master file. `exports/` holds older one-off files that sync folds in.
- Merge every CSV by `closed_utc`. Game numbers run 0-999 then wrap; a game closes every 160 s.
- The Keno site blocks Anthropic's servers (Akamai "Access Denied"), so never fetch
  results directly from a Claude session.
- `keno_qld_paytable.json` gives $1 payouts for K4-K10. Average return: K4-K6 about 74-75c,
  K7-K10 about 64-65c per $1. K7-K10 top prizes are growing jackpots, so the table is a floor.

## Round protocol
1. Training window: the latest 100 completed games.
2. `python lab/round.py <merged csv> <last training game> R00N out.json` runs the anomaly
   scan (10 checks against 2,000 random 100-game windows) and builds 7 policies:
   Anomaly composite (primary), Hot, Surge, Pair cluster, Overdue, Random A, Random B.
   Each policy keeps a ranked top 10, nested as tickets K4-K10. K6 is the primary size.
   From R005 it also builds a Weighted blend: each of the 7 policies gives its top 10 numbers
   10 points down to 1, multiplied by its current weight in `ledger/weights.json`; the 10 numbers
   with the most points form the blend's tickets. The blend is scored and shown, but stays out of
   the GRPO update because it is built from the weights.
   From R006 it also sets three Heads or Tails bets ($1 a game for all 40 test games):
   H/T trend (whichever of heads or tails won more training games), H/T contrarian (the other
   side) and Always evens. Payouts: heads or tails $2, evens $4 per $1 (`keno_qld_paytable.json`).
   They are scored and shown in their own panel, outside the GRPO update.
3. Freeze: add `frozen_utc`, `test_games` and the expected close times, write canonical
   JSON (`sort_keys`, compact separators), and post its sha256 in chat before the test
   window opens. Commit it to `rounds/R00N/frozen.json` and `frozen.sha256`.
   The test window is the first 40 games closing at least 4 minutes after the freeze.
4. Scoring with `python lab/score.py R00N`, once the test games are in draws.csv. For each policy:
   K10 hits and hit rate, K6 return (the reward), return for K4-K10, complete hits, and
   the first game with a complete hit. No random-ticket simulation: use the exact
   luck band instead. For K10 over 40 games, 95% of tickets land between 21% and 29%.
   That band is a label only, never a gate.
5. GRPO update. Reward is the K6 return. Advantage = (reward - group mean) / group sd.
   Weights (from R004 scoring on): each policy's weight is exp(its mean advantage over the last
   10 scored rounds, this one included), renormalised to sum to 1. So chance results fade instead
   of compounding. (R001-R003 used a multiplicative update; weights.json was recomputed from their
   advantages under this rule on 22 Oct.)
   Update `ledger/weights.json` and append to `ledger/leaderboard.csv`. The policy with
   the highest weight is flagged "On a roll" for the next round (ties go to the higher
   K10 hit rate). `lab/score.py` does steps 4-6 apart from the commit.
6. Commit the scores to `rounds/R00N/scores.json`.

## Reporting
Use the in-chat dashboard layout from round 1:
- header with the round number and status pill
- stat tiles: training games, ticket variants, test games, scored games
- primary prediction card with numbered tiles and a copy button
- category chips (All / Anomaly / Signal / Baseline) and a K4-K10 size selector
- one card per policy
- test status checklist and the fingerprint

Build the page with `python lab/dashboard.py R00N out.html` (works before and after scoring; it
includes every round up to R00N with a round picker and a results-so-far table)
and republish it to the same artifact: https://claude.ai/artifact/CHcWvqAfQfKLw4Eoqu2MEg
Show times in Brisbane time only (AEST, UTC+10, no daylight saving), in the dashboard and in chat.
Scored rounds add results to each card. Daryl wants honest opinions, never what he
wants to hear.
