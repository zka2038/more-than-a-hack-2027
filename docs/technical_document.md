# Possession from automatic tracking: technical document (draft)

> Draft. Final numbers are filled in at freeze; rename to `<team_name>.md` for submission.

## 1. Approach

We label every tactical-video frame `H`, `A` or `D` with four stages that share fixed file
contracts (`PLAN.md`, `solution/pipeline/contracts.py`):

1. **S0** reads the ATD (positions in metres, one team per track slot), the homography and Smart
   Tagging into one cache.
2. **S1** decides whether the ball is in play, using learned evidence, a joint decoder and a kick-off
   rule.
3. **S2** gives a ball track, linearly interpolated through gaps of up to 6 s inside play.
4. **S3** gives per-frame team evidence from the learned model.
5. **S4** turns the evidence into possessions with Viterbi decoding, tuned on spellF1.

The core is one **learned per-frame evidence model**. It reads the tracking around each frame and
outputs P(H), P(A), P(D) and P(possession changes here). An **exact Viterbi decoder** over {H, A, D}
then turns those probabilities into a small number of well-placed possessions. A change of state
costs a fixed penalty, so the decoder only switches team when the evidence summed over frames pays
for it. This matches the ranking metric: spellF1 scores whole possessions, and one invented flicker
splits a possession in two.

**Why this design.**
- Possession changes only on *controlled* possession, and whether a touch became control shows in
  the seconds *after* it. A bidirectional model sees those seconds. Because decoding is offline,
  each change is also placed at its start, not 1–2 s late.
- An oracle analysis on the external benchmark (§4) showed the loss is almost all in team
  hand-overs, not in stoppages. So most of the modelling effort went there.

### Features (`solution/model/features.py`)

Each frame gets team-level statistics only, never slot identity, because the 46 tracklets swap and
fragment:
- **Ball:** observed or missing, gap length, time since and until the ball was seen, speed, change in
  velocity, signed distance past each line.
- **Each team:** distances of its 3 nearest players to the ball, counts within 2, 5 and 10 m, the
  nearest player's speed and speed relative to the ball, centroid position and forward velocity,
  spread, defensive-line height, share of players goal-side of the ball, ball velocity along the
  team's attacking direction, and opponent pressure.

x is expressed in each team's attacking direction, found from where the teams stand and never from
labels. That makes the features invariant to pitch flips. The two team blocks are interchangeable,
and training swaps them at random (with H↔A) so the model cannot learn a home-team bias.

### Model (`solution/model/net.py`, `train.py`)

- **Network:** a linear layer and a temporal convolution, then a 2-layer bidirectional GRU with
  128 units, then two heads: 3-way state, and "possession changes within ±5 frames".
- **Training:** about 40-s chunks from external matches, each under several random degradations.
- **Ensemble:** the shipped checkpoint averages the probabilities of several seed members.
- **Inference:** on CPU, deterministic, with probabilities rounded before decoding.

### Rules on top (`solution/model/rules.py`)

**Kick-off detection.** The second half is graded from before its kick-off, and those frames are all
`D`. The kick-off is found from tracking alone: the ball on the centre spot with each team in its own
half, then the ball leaving the spot. It is accepted only if the model already rates the frames
before it as mostly dead. No frame number is hard-coded.

## 2. Assets used and dropped

| asset | use |
|---|---|
| tactical ATD | the only input to the model |
| homography | read into S0 (`cam_xy`, `reproj_err`); not used by the model, because external matches have no equivalent to train on |
| Smart Tagging | read into S0; not used yet, for the same reason. Candidate restart prior |
| tactical video | not used |
| broadcast video and ATD | not used: no alignment to the tactical timeline |

## 3. Training data (none from the provided game)

| dataset | what | labels | source and licence |
|---|---|---|---|
| DFL open data (Bassek et al. 2025) | 7 Bundesliga matches, 25 fps, corrected tracking | BallStatus / BallPossession per frame, from DFL's tracking provider | figshare 28196177, CC BY 4.0 |
| Metrica Sports sample games 1–3 | 3 anonymised matches, 25 fps | derived by our code from the published human event data | github.com/metrica-sports/sample-data (acknowledge source) |

**Label convention.** Possession changes only on control, so challenges never flip it.
- **Metrica:** labels come from the events. Possession is the team of the latest pass, recovery,
  carry, set piece, shot or ball lost. Dead ball runs from ball out, foul, goal or offside to the
  next set piece.
- **DFL:** DFL flips possession on the first touch. Opposing spells shorter than 1 s that return to
  the same team (or to a stoppage) are folded back. The 1 s threshold makes DFL's possession counts
  match Metrica's; it was chosen by comparing external matches only.

**Degradation (`solution/bench/degrade.py`).** Clean tracking is made to look like the ATD, using
only facts published in `DATA.md`:
- about 41% of the ball missing, in bursts that are likelier in crowds, in the air and off the pitch;
- ground-plane projection of airborne balls;
- correlated jitter;
- player dropouts;
- wrong-team ghost tracklets.

Training randomises these settings.

**What was never used.** The BAR-ATH ground truth, public or private, was used for no training,
tuning, calibration or model selection. A guard test (`tests/test_rules.py`) checks this. The public
window serves only as a final correctness check.

## 4. Validation

- **Protocol:** 4-fold grouped cross-validation over 8 external matches. Two further DFL matches
  were held out and scored once.
- **Metric:** the official scorer (`evaluation/possession_metrics.py`).

| method | dev spellF1 (8 matches, degraded) | held-out spellF1 (2 DFL) |
|---|---|---|
| stub pipeline (≈ published baseline) | 0.330 | — |
| single model + Viterbi | 0.730 | — |
| 3-model ensemble + Viterbi | 0.755 | 0.733 |

Oracle ladder (3-model run):
- true dead/alive state: +0.03;
- true team inside our predicted play: 0.95.
- Remaining loss: short possessions under 4 s, recovered 17–59% of the time against over 90% for
  possessions over 10 s.

These are scores on *simulated* noise and other matches' label conventions. They rank methods; they
do not predict the BAR-ATH score.

## 5. Possession rules implemented

- `D` comes from the decoded dead/alive state. `X` is written only on ungraded frames.
- A restart gives possession to whichever team the evidence favours after it.
- Possession changes when the summed evidence for the other team outweighs the switch cost. A single
  touch, a deflection or a duel the other team does not control is too short to pay that cost.
- A loose or unseen ball stays with the last possessor (the decoder's switch cost).
- Changes are placed at their start because decoding is offline and bidirectional.
- Frames before a detected kick-off are `D`.

## 6. Limitations

- **Domain gap.** Simulated degradation is not the real ATD. A script compares input statistics
  (`solution/bench/domain_check.py`), but real-noise performance is unknown.
- **Short possessions** under 2 s are mostly missed, which causes under-segmentation.
- **Restart team, Smart Tagging, homography** and video re-detection of the ball are not exploited.

## 7. Reproduction

```bash
python -m pip install -r requirements-solution.txt
python -m solution.pipeline.run_all --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

To retrain (optional; the committed weights reproduce the file):

```bash
export MTAH_CACHE=/path/to/cache            # default outputs/cache
git clone https://github.com/metrica-sports/sample-data $MTAH_CACHE/metrica
python -m solution.bench.external --download --parse
python -m solution.model.train --cv --run cv --members 3 --set steps=6000
python -m solution.model.tune --run cv --out decoder.json
python -m solution.model.train --final --with-holdout --members 5 --set steps=6000 --decoder decoder.json
```
