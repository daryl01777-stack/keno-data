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
Rules frozen for R018-R027 (Daryl, 25 Oct, after the R017 review): no changes to the reward, weights
rule, policies, blend, test window, replay rule or Retro pick until R027 is scored. Bug fixes and
dashboard display changes are fine; anything else waits for the review after R027. (R008-R017 was the
first frozen block; its review found no policy beat chance and changed only the reward and Retro
pick's windows.)

1. Training window: the latest 100 completed games.
2. `python lab/round.py <merged csv> <last training game> R00N out.json` runs the anomaly
   scan (10 checks against 2,000 random 100-game windows) and builds 7 policies:
   Anomaly composite (primary), Hot, Surge, Pair cluster, Overdue, Random A, Random B, and from R008
   Persistence (numbers drawn 7+ times in each 25-game quarter, i.e. above average every quarter;
   steadiest first, then most drawn; it shares about half its top 10 with Hot).
   Each policy keeps a ranked top 10, nested as tickets K4-K10. K4 is the primary size from R007 (K6 up to R006).
   It also builds a Weighted blend (from R005). From R008 the blend is a seat allocation: Hot,
   Surge, Pair cluster, Overdue and Persistence once it has a weight (not Anomaly composite, which already mixes the other signals,
   and not Random A/B, which are controls) take turns by
   Sainte-Lague (weight / 1, 3, 5...), using their weights in `ledger/weights.json`; on its turn a
   policy adds its best-ranked number not already taken, so overlap is not additive and each policy
   is independent. Pick order is the ticket ranking, so K4 is the first 4 picks. (R005-R007 used an
   additive vote over all 7 policies.) The blend is scored and shown, but stays out of the GRPO
   update because it is built from the weights.
   R006-R007 also set three Heads or Tails bets (dropped from R008 at Daryl's request; score.py
   still scores them for rounds that have them) ($1 a game for all 40 test games):
   H/T trend (whichever of heads or tails won more training games), H/T contrarian (the other
   side) and Always evens. Payouts: heads or tails $2, evens $4 per $1 (`keno_qld_paytable.json`).
   They were outside the GRPO update and are no longer shown on the dashboard.
3. Freeze: add `frozen_utc`, `test_games` and the expected close times, write canonical
   JSON (`sort_keys`, compact separators), and post its sha256 in chat before the test
   window opens. Commit it to `rounds/R00N/frozen.json` and `frozen.sha256`.
   The test window is the first 40 games closing at least 4 minutes after the freeze.
   After each freeze, message the project coordinator with the last test game's expected close
   (Brisbane time); the coordinator schedules the phone alert telling Daryl the round is ready to
   score (run `keno`, then say "score it"). Don't also schedule one from the round's thread.
4. Scoring with `python lab/score.py R00N`, once the test games are in draws.csv. For each policy:
   K10 hits and hit rate, the replay result for every size (see below), the reward, return for
   K4-K10, complete hits, and
   the first game with a complete hit. Every policy except the blend is in the GRPO group; a policy
   added later counts the window rounds it missed as zero advantage (from 23 Oct, after R008
   scoring; before that it was averaged over only the rounds it had played). No random-ticket simulation: use the exact
   luck band instead. For K10 over 40 games, 95% of tickets land between 21% and 29%.
   That band is a label only, never a gate.
   Replay result (`lab/replay.py`): a $20 ticket for the first 20 test games, then its winnings
   replayed at $1 a game until it busts, a single game pays more than $20 (4/4, 5/5, 5/6+, 6/7+,
   6/8+, 7/9+, 7/10+), or times out at the end
   of the 40-game window. Net = final bank - $20.
5. GRPO update. Reward (from R018, Daryl's call on 25 Oct) is 1 if any of the policy's K4-K10 replay
   tickets reached the take-win cutoff (a single prize over $20) in the round, else 0: rounds with a
   win, so one hot draw lifting several nested sizes counts once. Weights after R017 were recomputed
   under it. (R008-R017 scored the count of tickets, 0-7, and the leaderboard keeps those values.) (22 Oct to R007 it was the average replay net in dollars.) Advantage = (reward - group mean) / group sd. When weights are computed, every round in
   the 10-round window is re-judged under this reward from its stored per-game catches.
   (Leaderboard `reward` up to R006 holds the old per-$1 K6 return; from R007 it is the replay reward.
   `reward_roi_pct` holds ROI up to R007 and is left blank from R008, since a count has no ROI.)
   `ledger/sizes.json` holds, per policy and size over the last 10 rounds, cutoff wins (`big`), the
   rate against chance (`x_chance`; chance from `replay.chance()`, exact: K4 8.8%, K5 1.8%, K6 8.4%,
   K7 2.4%, K8 6.7%, K9 1.8%, K10 4.9% a round) and the average replay net, plus the best ("ideal")
   size: most cutoff wins, ties to the higher average net then the smaller ticket (R008 froze the
   older dollar-based ideal sizes); round.py freezes those as `ideal_sizes`, and the dashboard shows the
   weights' pick: the on-a-roll policy's ticket at its ideal size.
   Weights (from R004 scoring on): each policy's weight is exp(its mean advantage over the last
   10 scored rounds, this one included), renormalised to sum to 1. So chance results fade instead
   of compounding. (R001-R003 used a multiplicative update; weights.json was recomputed from their
   advantages under this rule on 22 Oct.)
   Update `ledger/weights.json` and append to `ledger/leaderboard.csv`. The policy with
   the highest weight is flagged "On a roll" for the next round (ties go to the higher
   K10 hit rate). `lab/score.py` does steps 4-6 apart from the commit.
6. Commit the scores to `rounds/R00N/scores.json`.

## Retro pick shadow (from R009, Daryl, 23 Oct)
A shadow policy that automates the retrospective: at freeze, `lab/shadow.py` runs 12 simple rules
over every completed 100+40 game window before the round and takes the ticket of the rule with the
most windows with a cutoff hit (the round reward; ties: higher K10 catches, then library order), plus
that rule's ideal size (most cutoff hits by size). From R018 the windows are anchored at the oldest game
in draws.csv and step forward 40 games, so past windows stay fixed as history grows (before, they were
re-cut from the latest game every freeze and the chosen rule kept flipping). It is outside the GRPO
group, the weights and the blend.
From R010 round.py writes it under `shadow` in frozen.json. R009 was added after its freeze (built
from games up to 308 only, before any test result was seen) in `rounds/R009/shadow.json` and
`shadow.sha256`. Score with `python lab/shadow.py R00N` after score.py; it writes
`rounds/R00N/shadow_scores.json` and appends `ledger/shadow.csv` with Random A/B for comparison.
The dashboard shows it as a "Shadow" card (own chip, no weight, last in order, its own fingerprint
under the main one) and scores it from shadow_scores.json; it is left out of the group averages.

## Agreement shadow (from R018, Daryl, 25 Oct)
A second shadow ticket, outside the GRPO group, the weights and the blend (added alongside the
R018-R027 freeze, which it does not touch). At freeze `lab/agreement.py` (called by round.py, stored
under `agreement` in frozen.json) gives each non-random policy's top 10 rank points (10 for #1 down to
1 for #10), multiplies them by the policy's weight renormalised over Anomaly composite, Hot, Surge,
Pair cluster, Overdue and Persistence, and ranks numbers by the sum (ties to the lower number), nested
K4-K10. Agreement only, no strength term. Ideal size: from its own last 10 scored rounds (most cutoff
wins, then average net, then smaller), K4 until it has one. Score with `python lab/agreement.py R0NN`
after score.py; it writes `rounds/R0NN/agreement_scores.json` and appends `ledger/agreement.csv`.
The dashboard shows it as a second Shadow card. Daryl does not want per-number scores shown.
Transition and late-spike shadows were floated but not approved.

## Reporting
Use the in-chat dashboard layout from round 1:
- header with the round number and status pill
- stat tiles: training games, ticket variants, test games, scored games
- primary prediction card with numbered tiles and a copy button (from R008, Daryl's call on 23 Oct: the Weighted blend at its frozen ideal
  size; rounds without ideal_sizes, R001-R007, show Anomaly composite;
  the ideal size is underlined and the selected size is drawn slightly larger)
- category chips (All / Anomaly / Signal / Baseline / Blend); a K4-K10 size selector on the primary
  card and inside each policy card, coloured by replay outcome (green big win, red bust, yellow timed out)
- one card per policy, ordered by current weight (weight after scoring on scored rounds); unweighted cards (blend, new policies) last
- cutoff-wins heatmap (replaced the catch table on 23 Oct): policies x K4-K10, how many times each
  ticket reached the take-win cutoff over scored rounds, a chance-per-round row, best size underlined,
  tap a cell for per-round results
- test status checklist and the fingerprint

Build the page with `python lab/dashboard.py R00N out.html` (works before and after scoring; it
includes every round up to R00N with a round picker; the results-so-far, best-size and
training-scan panels were removed at Daryl's request on 23 Oct)
and republish it to the same artifact: https://claude.ai/artifact/CHcWvqAfQfKLw4Eoqu2MEg
Show times in Brisbane time only (AEST, UTC+10, no daylight saving), in the dashboard and in chat.
Scored rounds add results to each card. The headline result per ticket (from 22 Oct, applied to
all rounds by `lab/dashboard.py`) is a $20 ticket for the first 20 test games, then its winnings
replayed at $1 a game until it busts, a single game pays more than $20, or the 40-game window ends. Daryl wants honest opinions, never what he
wants to hear.
