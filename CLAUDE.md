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
Rules frozen for R028-R037 (Daryl, 27 Oct, after the R027 review, unchanged from R018-R027): no
changes to the reward, weights rule, policies, Agreement ticket, test window or replay rule until R037
is scored. (R018-R027 review: no policy beat chance; Agreement K4 won 2 of 10 rounds, +$66.) Bug fixes and
dashboard display changes are fine; anything else waits for the review after R037. (R008-R017 was the
first frozen block; its review found no policy beat chance and changed only the reward and Retro
pick's windows.) The same day Daryl approved a restructure from R018: the policies became the 12 Retro
rules plus Pair cluster (with Random A/B), the Agreement ticket became the primary pick, and Anomaly
composite, the Sainte-Lague blend and the Retro pick shadow were dropped. The setup before that is kept
on the branch `pre-r018-restructure` (commit 0d807e0) so it can be restored.

1. Training window: the latest 100 completed games.
2. `python lab/round.py <merged csv> <last training game> R00N out.json` runs the anomaly
   scan (10 checks against 2,000 random 100-game windows; information only from R018) and builds the
   policies. From R018 (Daryl, 25 Oct): 13 weighted policies plus two random controls.
   - Hot (most drawn), Overdue (longest gap), Surge (last 20 games), Persistence (numbers drawn 7+
     times in each 25-game quarter; steadiest first, then most drawn) and Pair cluster keep their own
     builders and weight history.
   - Eight more rules come from `shadow.rules()`: Cold (100), Hot last 10, Not in last game, Hot last 3,
     Rising (2nd half), Just drawn (last game), Pairs with last game, Neighbours of last game. They
     count R008-R017 as zero advantage under the newcomer rule.
   - Random A and Random B are the luck controls and keep their weight as the benchmark.
   Each policy keeps a ranked top 10, nested as tickets K4-K10. K4 is the primary size from R007.
   It also builds the Agreement ticket (`lab/agreement.py`), the primary pick from R018: for every
   non-random policy, each number in its top 10 gets a signal strength, -log10 of the chance that a
   random 100-game history gives that policy's raw score or more (`lab/strength.py`; chance table in
   `ledger/strength_null.json`, 20,000 seeded random histories, rebuild with `python lab/strength.py
   build`), floored at 0; times the policy's weight in `ledger/weights.json` renormalised over the 13,
   summed per number; ticket in descending score (ties to the lower number), nested K4-K10. Daryl chose
   strength over rank points before the R018 freeze. Per-number scores are not shown (Daryl).
   From R019 (Daryl, 25 Oct) look-alike balancing: each policy's weight is divided by 1 + its
   above-chance top-10 overlap with every other policy (`ledger/agreement_overlap.json`, built once
   from 106 windows to game 472 with `python lab/agreement.py overlap 472`), then renormalised, so the
   hot-number family (Hot, Pairs with last game, Pair cluster, Not in last game, Persistence, Surge)
   no longer outvotes the rest (its share fell from about 42% to 25%). R018's frozen ticket is unchanged. It is
   scored and shown, but stays out of the GRPO update because it is built from the weights. Its ideal
   size comes from `ledger/sizes.json` like the others, K4 until it has scored rounds.
   Up to R017: Anomaly composite (z-score mix of the signals), and from R005 a Weighted blend (from R008
   a Sainte-Lague seat allocation over Hot, Surge, Pair cluster, Overdue and Persistence; R005-R007 an
   additive vote). R006-R007 also set three Heads or Tails bets (H/T trend, H/T contrarian, Always
   evens; score.py still scores them for rounds that have them).
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
   the first game with a complete hit. Every policy except the blend (Agreement from R018) is in the GRPO group; a policy
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

## Retro pick shadow (R009-R017)
A shadow ticket from whichever of the 12 rules did best over past 100+40 windows (`lab/shadow.py`),
outside the weights. From R018 its rules became live policies, so it is no longer built. Past rounds
keep it: R009 in `rounds/R009/shadow.json`, R010-R017 under `shadow` in frozen.json, scores in
`shadow_scores.json` and `ledger/shadow.csv`. The dashboard still shows it on those rounds.

## Reporting
Use the in-chat dashboard layout from round 1:
- header with the round number and status pill
- stat tiles: training games, ticket variants, test games, scored games
- primary prediction card with numbered tiles and a copy button (from R018 the Agreement ticket at its
  frozen ideal size; R008-R017 the Weighted blend at its ideal size; R001-R007 Anomaly composite;
  the ideal size is underlined and the selected size is drawn slightly larger)
- category chips (All / Signal / Baseline / Agreement). Anomaly composite, Weighted blend and Retro pick are
  hidden on every round from 25 Oct (Daryl; data kept in the repo), except that older rounds' primary card
  still shows the ticket that was the pick then; a K4-K10 size selector on the primary
  card and inside each policy card, coloured by replay outcome (green big win, red bust, yellow timed out)
- one card per policy, ordered by current weight (weight after scoring on scored rounds); unweighted cards (Agreement or blend, shadows) last
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
