# PLAN — possession pipeline for BAR-ATH

Goal: one `H` / `A` / `D` label per tactical video frame, ranked on **spellF1** (macroF1 is the
tie-break). Deadline **17 Nov 2026**. Floor to beat: `baseline_smooth50`, spellF1 0.2433.

The pipeline is four stages with fixed file contracts between them, so each stage can be built on
its own branch by a different person (and their agent) without waiting for the others.

```
ATD + homography + Smart Tagging (+ video)
        │
   [S0] tracking cache ───────────────────────────────┐
        │                                             │
   [S1] in play / dead ──► s1_inplay.npz, s1_stoppages.csv
        │                                             │
   [S2] ball track (inside alive segments) ──► s2_ball.npz
        │                                             │
   [S3] touches / last-touch team ──► s3_touch.npz, s3_touches.csv
        │                                             │
   [S4] possession decode ──► <team_name>.csv  (the submission)
```

---

## Hard rules — read before writing any code

A rule breach disqualifies the submission, and no spellF1 gain is worth that. These hold on every
branch, for every person and every agent. Sources: `RULES.md`, `FAQ.md`, `SUBMISSION.md`, and the
LLM clause in `OVERVIEW.md`. If an idea sits near one of these lines, open an issue and ask the
organisers before building on it.

**1. BAR-ATH ground truth never trains, tunes or selects anything.** That covers the public 5
minutes (`evaluation/gt/BAR-ATH_possession_gt_public.csv` and its `_meta.json`) and the private
full match. It is never used for:
- training labels, pseudo-label seeds, or calibrating any probability;
- choosing a hyperparameter, threshold, window, switch cost, minimum duration or lag;
- early stopping, model selection, or picking between two branches or versions;
- a number read off it and baked into the code (an offset, a duration, a count, a class share).

It **is** for one correctness check per frozen candidate: frames aligned, `H`/`A` not swapped, file
well-formed, score clearly above the baseline. If that check exposes a bug, fix the bug; never
nudge a parameter until the public score looks better. Its qualitative conventions may be read once
by a human and written down as words, not numbers (§6).

**2. Everything learned is learned from other games.** Every model is trained, every parameter
tuned and every probability calibrated on external matches with their own ground truth (DFL open
data, Metrica sample games; §6). Self-supervision on BAR-ATH is allowed only where no person decides
any target, for example hiding observed ball positions and scoring the fill (§3). Before using an
external dataset, confirm it does not contain this match (Barcelona v Athletic, LaLiga 2024-25,
matchday 2) and record its source, licence and who labelled it, for the technical document.

**3. No hand labels or hand corrections on BAR-ATH, in any amount.** That means:
- no labelling frames;
- no fixing ball positions or tracklet identities by hand;
- no editing our own output;
- no rules or constants tied to frames someone watched (`if 52000 < f < 52300: ...`).

Watching the video or the frame viewer to understand the data and design features is fine.
Turning what you saw into frame-specific labels, corrections or constants is not. The only frame
constants allowed are the boundaries the organisers publish (the graded halves, via
`games.half_video_blocks`). The published second-half kickoff frame checks S1's detection and is
never an input.

**4. No human-produced data about this match** from any source: no event data (Opta, StatsBomb,
Wyscout, LaLiga or anyone else), corrected tracking, possession statistics or annotations. Smart
Tagging is fine: it is automatic and supplied for this purpose.

**5. Match data never reaches a cloud-hosted LLM.** The terms forbid making "any videos, datasets
… or any extracts thereof" available to one, and that includes the coding agent writing this
pipeline. In practice:
- Agents never open `data/`, `outputs/`, `evaluation/gt/`, or any frame-level file produced from
  BAR-ATH: stage `.npz`/`.csv` files, submissions, frame-viewer pages.
- Agents never run a command that reads those files, because tool output goes back to the model.
- A human runs real-data commands in a separate terminal, not through the agent's shell and not
  with an in-chat `!` command.
- What goes back to an agent is pass/fail and coarse aggregates only: validator OK, scores on the
  external benchmark, at most a BAR-ATH possession count. Never rows, arrays, plots, screenshots,
  video frames, or tracebacks that print data values; scrub those first. When in doubt, share nothing.
- Keep `data/` and `outputs/` out of cloud-synced folders (OneDrive, Dropbox, Google Drive), which
  may run their own AI indexing over files. If the repo lives in one, set `MTAH_DATA_DIR` and
  `--stages` to a local path.

**6. Nothing match-derived is committed or published.** `data/`, `outputs/` and
`evaluation/results/*.csv` stay gitignored. No model or parameter file fitted on BAR-ATH is
committed: anything self-supervised on BAR-ATH is refitted by the pipeline at run time. Weights
trained on external data may be committed if that data's licence allows it.

**7. The repository regenerates the submitted file exactly.** That needs:
- pinned dependencies with exact versions, including the GPU stack;
- fixed seeds;
- external-trained weights committed;
- deterministic inference (on CPU, or deterministic GPU kernels);
- one command: `python -m solution.pipeline.run_all --out evaluation/results/<team_name>.csv`.

Before submitting, run that command twice from a clean clone and check the two files are
byte-identical.

**Guard in code.** `tests/test_rules.py` fails in either case:
- anything under `solution/` references `evaluation/gt`, `possession_gt` or `--gt`;
- git tracks a data-like file (`.npz`, `.npy`, `.pkl`, `.parquet`, `.csv`) that is not an
  organiser-committed file and not on an explicit allow-list of external-trained weights.

Run it before every merge to `main`.

---

## 0. What the metric and the ground truth force on us

Read this before designing anything; every stage has a consequence here.

**How spellF1 works** (`evaluation/possession_metrics.py`): a possession is a maximal run of `H` or
of `A`. `D` is never a possession but it **breaks** one. Predicted spells are matched one-to-one to
true spells of the same team by temporal IoU; F1 is averaged over IoU 0.3–0.7.

| our mistake | what it costs |
|---|---|
| a false `D` flicker inside a possession | splits one spell into two: +1 spurious, and the IoU of the real one drops |
| a missed `D` between two spells of the **same team** (throw-in, free kick to the side that had it) | merges two true spells: at least one missed, the other with diluted IoU |
| a missed `D` between spells of **different teams** | cheap: only the boundaries move |
| a false short `A` inside an `H` spell (a deflection read as a hand-over) | splits `H` in two and adds a spurious `A`: up to −1 matched, +2 spurious |
| a missed real short spell | 1 missed, and its two neighbours merge into one |
| boundary 1 frame off | free |
| boundary 8 frames late everywhere | −0.11 |
| boundary 1 s late everywhere | −0.35 |
| `X` on a graded frame | forward-filled with our previous label, so never emit it in graded frames; fill ourselves |

So: **segment count and boundary timing matter, per-frame accuracy barely does.** Every stage must
output stable intervals and well-timed boundaries (target: within ±8 frames), not smooth per-frame
scores.

**What the ground truth is.** Derived mechanically from *corrected* tracking: `D` where that
tracking marks the ball dead, otherwise the team it names as owner. Possession changes only on
**controlled** possession: a touch, a deflection or an aerial duel won without control does not
flip it; a loose ball stays with the last team. This is exactly the S3 → S4 split below.

**Published reference numbers** are for spotting bugs and are never targets. `DATA.md` says the
public 5 minutes hold 17 possessions with a mean of 14.2 s, and about 20% of frames are `D`. Do not
scale these up to a full-match count and steer toward it. The organisers withhold full-match counts
precisely so nobody tunes to them (`EVALUATION.md`). The only count reference we use comes from
external matches: Metrica sample games have 300–356 possessions per 90 min (§6). On BAR-ATH a
count only flags gross breakage. 800 possessions means something upstream is broken; 340 versus 300
tells us nothing.

**Free facts from the docs:**
- Graded frames are 11,185–90,651 and 110,808–190,197. Everything else is written `X`.
- The second half is graded from 110,808 but kicks off at ~113,742: those frames are all `D`.
  Detect the kickoff by algorithm (S1) and use the published anchor only to check it.
- The ball is missing on 41% of graded frames, mostly where it matters.
- Track slots are tracklets, not players (46 slots, identity swaps, occasional wrong-team ghosts).

**Rules that shape the work:** see *Hard rules* above; they override anything below. Two of them
shape the design most:
- All training, tuning and calibration happens on **other matches** (§6). BAR-ATH ground truth,
  public or private, is never a training or tuning signal.
- **Agents never see match data.** They write code against the contracts, synthetic fixtures and
  external matches, and a human runs it on the real data.

**What we already know** (branch `no-training-approach`, `solution/decoder/RESULTS.md`): a learned
per-frame evidence model plus a sticky Viterbi decoder scores 0.53–0.56 spellF1 on held-out
external matches with simulated tracking noise. On that stand-in:
- substituting the true dead/alive state adds about +0.07;
- substituting the true team during play lifts the score to 0.86;
- recovering every missing ball position adds about +0.06.

So the largest loss is **hand-over detection (S3/S4)**, not stoppages and not ball coverage. These
were measured on simulated noise and DFL's first-touch convention, so treat them as a staffing
hint, not a fact about BAR-ATH. That decoder is our end-to-end v0: every stage below must beat it
when swapped in, on the same benchmark.

---

## 1. Shared conventions (all stages)

| item | convention |
|---|---|
| frame index | tactical video frame `0 … 199318`; arrays have length `N = 199319`; ATD `frame_id − 1` |
| time | 25 fps; Smart Tagging seconds → `round(s * 25)` |
| coordinates | **metres**, `x = atd_x * 105.3`, `y = atd_y * 68.0`, ATD orientation kept (no per-half flip) |
| teams | `0 = H` (Barcelona), `1 = A` (Athletic), `-1 = unknown` |
| missing | `NaN` for floats, `-1` for ints |
| half | `int8 (N,)`: `1`, `2`, or `0` for ungraded frames |
| intervals | `[start_frame, end_frame)`, end exclusive |
| probabilities | `float32`, calibrated **on external matches only** (never on BAR-ATH ground truth); every stage ships soft scores **and** a hard decision |
| storage | `outputs/stages/` (gitignored; never commit, it contains match-derived data) |
| format | `.npz` for per-frame arrays, `.csv` for event tables; each `.npz` carries a `meta` JSON string (stage, version, git sha, params) |

Each stage is a module `solution/pipeline/sN_<name>/` with one entry point:

```bash
python -m solution.pipeline.s1_inplay.run --stages outputs/stages
```

`solution/pipeline/contracts.py` owns the schemas: `load_s1()`, `save_s1()`, … each validating
shape, dtype, value range and NaN policy. `tests/test_contracts.py` runs on synthetic fixtures with
no match data. **Nobody changes a contract on a stage branch**; contract changes go through a PR to
`main` that all four owners see.

**Stubs first.** Day one, each stage ships a trivial implementation that satisfies its contract, so
downstream owners are never blocked:

| stage | stub |
|---|---|
| S1 | dead where the raw ball is missing for > 3 s or outside the pitch |
| S2 | raw ATD ball, linear interpolation through gaps ≤ 6 s inside alive segments |
| S3 | nearest player within 1.5 m of the ball = touch |
| S4 | last-touch team + majority vote ±50 frames (= the published baseline, re-expressed) |

### S0 — tracking cache (shared input, built once)

Owner: whoever takes the contracts PR. Reuse `solution/decoder/atd.py` from `no-training-approach`.

`s0_tracking.npz`

| array | dtype, shape | meaning |
|---|---|---|
| `ball_xy` | f32 `(N,2)` | raw ATD ball, metres, NaN if missing |
| `player_xy` | f32 `(N,46,2)` | track slots, metres |
| `player_speed` | f32 `(N,46)` | m/s as given |
| `player_team` | i8 `(46,)` | team per slot. The reader maps every frame's groups through its `DataFormatSpecification` back to `<Player>` list order, so a slot's team is constant even where the raw column order changes (mapping by list order alone scrambled teams on a Metrica sample game) |
| `half` | i8 `(N,)` | see above |
| `cam_xy` | f32 `(N,2)` | pitch point under the image centre, from the inverse homography |
| `reproj_err` | f32 `(N,)` | homography reprojection error |

`s0_tags.csv`: `start_frame,end_frame,code,team,type,half,side` (team `-1` for `N/A`).

---

## 2. Stage 1 — ball in play vs out of play

**Branch** `stage1-inplay`. **In:** S0 (later, optionally, S2 for a second pass).

**Out:** `s1_inplay.npz`

| array | dtype, shape | meaning |
|---|---|---|
| `p_alive` | f32 `(N,)` | P(ball in play); NaN on ungraded frames |
| `alive` | i8 `(N,)` | `1` alive, `0` dead, `-1` ungraded |

and `s1_stoppages.csv`, one row per dead interval:

`stop_id, start_frame, end_frame, cause, restart_type, restart_team, restart_x, restart_y, conf`

- `cause`: `out_touchline | out_goalline | foul | goal | offside | period | unknown`
- `restart_type`: `throw_in | corner | goal_kick | free_kick | kick_off | penalty | unknown`
- `restart_team`: `0 | 1 | -1`. **This is valuable to S4**: every restart has a clear possessor.
- `end_frame` is the frame the ball is put back in play (first alive frame).

**What "accurate" has to mean here** (from §0): few false stoppages, sharp start and end, and above
all catching stoppages where the **same team** restarts. A stoppage shorter than about 2 s is
almost never real; enforce a minimum duration in the decoder rather than by smoothing.

**Evidence to build, strongest first:**

1. **Ball leaves the pitch.** Ball observed beyond a line by a margin, or last seen moving toward
   a line and extrapolated to cross it before it vanishes. Start of `D` = crossing frame.
2. **Restart geometry at the end of a stoppage**, which also gives `restart_type` and the end time:
   - *kick-off:* ball still on the centre spot, each team in its own half;
   - *corner:* ball still within ~1 m of a corner, crowd in the box;
   - *goal kick:* ball still on the six-yard line, defending team spread, attackers outside the box;
   - *free kick:* ball still for > 2 s in open play areas, nearest opponent ≥ ~9 m away;
   - *throw-in:* a player at or beyond the touchline where the ball left, ball reappearing from there;
   - *penalty:* ball on the spot, box empty except taker and keeper.
   The restart frame is the first frame the ball accelerates from rest.
3. **Collective player kinematics.** Mean and upper-quantile player speed, the share of players
   walking, team centroid speed. On external data, stoppages averaged ~1.0 m/s versus ~2.25 in play.
4. **Ball stillness** and **long ball gaps**, especially gaps that begin near a line.
5. **Smart Tagging `SET PIECES`** (59 instances with a `Type`): high-precision anchors for a
   restart near their start time. Recall is low (26 throw-ins tagged in a whole match) and every
   throw-in has team `N/A`, so use them as evidence, not as the segmentation. `GOALS` (4) each
   imply a long stoppage followed by a `Kick Off (After Goal)`.
6. **Camera behaviour** from the homography: pan speed of `cam_xy`, and `reproj_err` rising when
   the camera leaves the pitch. Worth an hour to test; drop it if the tactical camera is static.
7. **Referee whistle from audio**, if the mp4 has an audio track: a narrow-band energy detector
   around 3.5–4.5 kHz. This is the only signal that sees fouls and offsides directly, which geometry
   cannot. Stretch goal; check for an audio track first.

**Model.** Per-frame features in windows (±0.5 / 2 / 5 s, plus one-sided before/after windows) →
gradient-boosted classifier for `p_alive`, trained on external matches → a two-state
**semi-Markov decoder** with explicit minimum durations for both states. Then snap each boundary to
the nearest physical event (line crossing for the start, ball acceleration for the end).

**Stage metrics** (external benchmark): alive/dead F1, number of predicted vs true stoppages,
median absolute boundary error in frames for starts and for ends, and recall on same-team restarts
specifically.

**Note on ordering.** S1 needs the ball and S2 wants S1's segments. Resolve it with two passes
under the same contract: S1 pass A on the raw ball → S2 → S1 pass B on the S2 ball (optional, only
if it measurably helps).

---

## 3. Stage 2 — ball tracking inside in-play segments

**Branch** `stage2-ball`. **In:** S0, S1.

**Out:** `s2_ball.npz`

| array | dtype, shape | meaning |
|---|---|---|
| `ball_xy` | f32 `(N,2)` | best ball position, metres; NaN where we decline to estimate |
| `ball_vel` | f32 `(N,2)` | m/s, from the smoothed track |
| `ball_src` | u8 `(N,)` | `0` none · `1` observed · `2` observed but corrected · `3` interpolated · `4` carried by a player · `5` re-detected from video |
| `ball_conf` | f32 `(N,)` | 0–1; downstream stages weight evidence by this |
| `in_air` | f32 `(N,)` | P(ball off the ground); NaN if not estimated |

**Guarantees:** no fill ever crosses a stoppage; inside dead intervals only `ball_src ∈ {0,1}`;
anything with `ball_src ≥ 3` has `ball_conf < 1`.

**Work, in order of value per hour:**

1. **Clean what is observed.** Remove false detections: teleports (implied speed > ~40 m/s),
   single-frame blips, a "ball" sitting on a static object. Mark as `ball_src = 2` or drop.
2. **Segment-wise smoothing.** Split each alive segment at velocity change-points, fit constant
   velocity with drag per piece (RTS/Kalman smoother). The change-points are S3's touch candidates,
   so export them cleanly through `ball_vel`.
3. **Gap filling by cause**, not one rule:
   - *short gap, ball in flight between two observations:* straight line with drag (`src 3`);
   - *ball vanished next to a player and reappears next to the same tracklet or a teammate nearby:*
     attach it to that player (`src 4`), the dribble/occlusion case;
   - *gap anchored at a restart:* use S1's `restart_x, restart_y` as the start point;
   - *long gap with no consistent story:* leave NaN. S4 can hold the previous possessor, which is
     free on the ranking metric; a wrong fill is not.
4. **Aerial balls.** The homography assumes the ground plane, so a ball in the air is projected to
   the wrong pitch position and appears to curve. Flag `in_air` from apparent curvature and speed;
   S3 must not attribute touches while `in_air` is high.
5. **Camera prior.** If the tactical camera follows play, `cam_xy` bounds where an unseen ball is.
6. **Re-detection from video** (stretch): crop around the predicted position via the homography
   and run a small-object detector trained on other games. Expensive; only after 1–4 are done.
7. **Broadcast ATD as a second ball source** (probably skip): needs a tactical↔broadcast time
   alignment we do not have, and earlier estimates put the gain near +0.02.

**Stage metrics.** Self-supervised on BAR-ATH itself, which the rules allow: hide observed ball
runs using the real gap-length distribution, fill them, report position error by gap length and by
`ball_src`. Also report coverage inside alive segments and the downstream spellF1 change on the
external benchmark.

**Honest expectation:** earlier experiments suggest perfect ball recovery is worth about +0.06 on
its own. S2 pays off mainly by giving S3 clean change-points and trustworthy confidence, so
prioritise items 1–3.

---

## 4. Stage 3 — who touched the ball last

**Branch** `stage3-touch`. **In:** S0, S1, S2.

**Out:** `s3_touches.csv`, one row per detected touch:

`touch_id, frame, team, slot, x, y, kind, conf, p_home`

- `kind`: `restart | receive | release | carry | deflection | duel | unknown`
- `p_home`: P(the touching player is home), so an ambiguous duel can be 0.5 instead of a wrong hard call
- `conf`: P(a touch happened at all)

and `s3_touch.npz`

| array | dtype, shape | meaning |
|---|---|---|
| `last_touch_team` | i8 `(N,)` | team of the latest touch with `conf ≥ 0.5`; `-1` before the first touch of an alive segment |
| `frames_since_touch` | i32 `(N,)` | age of that touch |
| `touch_ll` | f32 `(N,2)` | per-frame log-evidence that H / A is on the ball; `0,0` when uninformative |
| `control` | f32 `(N,2)` | per-frame evidence that H / A has the ball **under control** (see below) |

**Touch detection:**

1. **Kinematic change-points.** A touch is where the ball's velocity changes (direction or speed)
   beyond what drag explains, with a player within reach (~1–1.5 m, scaled by `ball_conf`).
2. **Carries.** Ball within ~1.5 m of one player and moving with that player's velocity for ≥ 0.4 s:
   a run of touches by the same player, `kind = carry`. This is the strongest control signal.
3. **Touches inside ball gaps.** When the ball reappears with a new velocity, back-project its path:
   the player nearest to the ray origin at the right time is the likely toucher. Lower `conf`.
4. **Restarts.** The first touch of each alive segment comes from S1's `restart_team` when known.

**Team attribution:**

- Use the team of the nearest tracklet, but vote over a short window around the touch so a
  one-frame wrong-team ghost cannot flip it.
- When players of both teams are within reach (tackle, aerial duel), emit `kind = duel` with
  `p_home` near 0.5 and let what happens next decide. Do not guess.
- Check before trusting it: are referees or keepers in the track slots, and how often does a
  tracklet's team label disagree with its neighbours in time?

**Control evidence** (what separates a possession change from a touch, per the challenge rules):
same-team consecutive touches, a carry lasting more than ~1 s, a completed pass (touch by player A,
flight, touch by a teammate), ball speed dropping to the player's speed after a reception.

**Model.** Rules first, then a classifier per candidate touch (features: distance and relative
velocity of the two nearest players of each team, change-point size, `ball_conf`, `in_air`, time
since last touch), trained on external matches that have event data (Metrica sample games carry
PASS / RECOVERY / CHALLENGE events with frames).

**Stage metrics** (external benchmark): touch detection precision/recall within ±5 frames, team
accuracy on detected touches, and the fraction of true hand-overs that have a new-team touch within
±12 frames. That last number was only 50–56% in earlier work and is the main thing to raise.

---

## 5. Stage 4 — from touch sequence to possession

**Branch** `stage4-possession`. **In:** S1, S3 (and S2's `ball_conf`).

**Out:** `evaluation/results/<team_name>.csv` in submission format (`frame,label`, 199,319 rows),
and `s4_possession.npz` with `prob` f32 `(N,3)` over `[H, A, D]` and `label` u8 `(N,)`.

**Rules to implement:**

1. `D` comes from S1. Ungraded frames are `X`. No `X` on graded frames.
2. At each restart the possessor is the restarting team: S1's `restart_team`, else the first
   confident touch of the segment.
3. Possession **changes only on control** by the other team: two consecutive touches by that
   team, or a carry over ~1 s, or a touch followed by a completed pass.
4. A single opposing touch, a deflection, or a duel followed by the original team keeping the ball
   changes **nothing** (the "brief touch that comes back" case).
5. A loose or unseen ball stays with the last possessor.
6. **Backdate.** We decode offline, so once control is confirmed, move the change back to the first
   touch of the controlling sequence, not to the moment we became sure. Otherwise every hand-over
   is 1–2 s late, and 1 s late costs 0.35.
7. Where the ball goes dead straight after an uncontrolled touch, the previous possessor keeps the
   frames up to `D`.

**Decoder.** A semi-Markov / Viterbi decode over `{H, A, D}` per half: emissions from
`touch_ll`, `control` and `p_alive`; a switch cost for `H↔A`; minimum durations per state; `D`
boundaries pinned to S1's snapped boundaries. Start from `solution/decoder/viterbi.py`
(`no-training-approach`) and add the control gate and backdating. Keep a pure rule-based state
machine alongside as a cross-check; if the two disagree wildly, something upstream is broken.

**Optimising for spellF1 directly:**

- Tune the switch cost, minimum durations and control thresholds on **spellF1 on external
  matches**, never on per-frame loss and never on the public window.
- Watch the possession count and mean duration next to the score **on external matches, against
  their own ground truth**. That is where over- or under-segmentation is measured and corrected. On
  BAR-ATH the count is only an alarm: a few hundred is plausible, 800 is a bug. Never adjust a
  parameter to move BAR-ATH's count toward a number.
- Check for systematic lag with the scorer's transition metrics at τ = 12 vs τ = 25 frames. Good
  at 25 but bad at 12 means a constant delay: fix it with backdating, not smoothing.
- Short spells: emit one only when its probability is above ~0.5. Both errors (inventing one,
  missing one) cost about the same, so do not bias toward either.
- Stretch: minimum-Bayes-risk decoding, i.e. sample segmentations from the model and pick the one
  with the best expected spellF1 against the others.

**Final checks before upload:**
- `python evaluation/validate_submission.py <file>`.
- Score on the public window as a correctness check only: frames aligned, labels the right way
  round, score well above the baseline. A human does this once per frozen candidate and never uses
  it to choose between candidates (Hard rule 1). If it reveals a bug, fix the bug, not a parameter.
- Second-half pre-kickoff frames are `D`.

---

## 6. Validation: how we know a change helped

There is no leaderboard and the public window cannot rank anything (one possession ≈ 6 points
there). So:

- **External benchmark** = Metrica sample games 1–3 (same vendor, closest possession convention:
  300–356 possessions per 90 min) plus DFL open matches, with tracking degraded to resemble ours
  (~42% ball missing in bursts, player dropouts, jitter, wrong-team ghosts). Loaders, degradation
  and scoring already exist in `solution/decoder/{metrica,dfl,degrade,evaluate}.py` on
  `no-training-approach`; caches are in `outputs/cache/`. Port these to `main` with the contracts.
- **Protocol:** leave-one-match-out; score on a degradation style not used in training; a change
  under +0.01 spellF1 is noise (the full-match noise band is ~0.04).
- **Oracle ladder**, re-run whenever a stage changes, to see where the remaining loss is:

  | run | tells us |
  |---|---|
  | all stages ours | the real number |
  | true S1 + our S2–S4 | what dead-ball errors cost |
  | true ball + our S1, S3, S4 | what ball gaps cost |
  | true touches + our S4 | what the possession rules cost |

- **Convention check on the public window.** This is allowed: `CHALLENGE.md` says to use the
  window to understand what is expected. Three questions:
  - Does the GT flip at the first touch of the new team, or later?
  - Does `D` start at the line crossing or at the whistle?
  - Who owns the frames between a duel and the next controlled touch?

  A human reads these off once; no agent opens the GT file. Write the answers into this file as
  words ("flips at the first controlled touch", "`D` starts at the line crossing"), never as
  measured offsets, durations or counts. Build to the words, and set every number from external
  matches.

---

## 7. Branches, ownership, order of work

| branch | owner | first deliverable |
|---|---|---|
| `pipeline-contracts` → `main` | 1 person, first | `contracts.py`, S0 cache, four stubs, `run_all`, ported external benchmark |
| `stage1-inplay` | | `p_alive` + stoppage table with restart type and team |
| `stage2-ball` | | cleaned + gap-classified ball with confidence |
| `stage3-touch` | | touch table + control evidence |
| `stage4-possession` | | control-gated decoder with backdating, tuned on spellF1 |

Suggested people split if we are short: S3 and S4 carry the most score, S1 next, S2 last.

**Order of work:**

1. Merge `pipeline-contracts` to `main`: contracts, S0 cache, four stubs, `run_all`, tests, and
   `tests/test_rules.py` (the guard in *Hard rules*). The rest is scaffolded in `solution/pipeline/`;
   the guard is not written yet.
2. Run the stub pipeline on the real data once and confirm it writes a valid submission.
3. Port the external benchmark (Metrica + DFL loaders, degradation, scoring) to `main` so it
   emits the same stage files. Nothing can be compared before this exists.
4. Settle the open questions in §8, including the convention check on the public window.
5. Build S1–S4 in parallel on their branches, each against the stubs of the others.
6. Merge a stage to `main` only when it beats the current version of that stage on the benchmark.
7. Re-run the oracle ladder (§6) after each merge and move effort to the stage that costs most.
8. Tune S4 on the integrated pipeline, on spellF1 on external matches.
9. Freeze. Reproduce the submission from a clean clone, twice, and confirm the files are
   byte-identical (Hard rule 7).
10. Write the technical document, validate the exact file, submit.

**Integration rule:** `main` always produces a valid submission with one command:

```bash
python -m solution.pipeline.run_all --out evaluation/results/<team_name>.csv
```

**Deliverables checklist** (`SUBMISSION.md`):
- `<team_name>.csv`.
- A public repo with pinned dependencies and the regenerating command. It must contain no match
  data and nothing fitted on BAR-ATH; `tests/test_rules.py` passes.
- A short technical document: approach, assets used and dropped, possession rules implemented,
  limitations, reproduction.
- In that document, every external dataset: its source, licence, who labelled it and how much, a
  statement that it does not include this match, and a statement that BAR-ATH ground truth was used
  only for the correctness check.

---

## 8. Open questions to settle early

1. Does the tactical mp4 have audio (whistle detector for S1)?
2. Does the tactical camera pan with play (camera prior for S1 and S2)?
3. Are referees or goalkeepers present in the 46 track slots, and how reliable is the per-slot team?
4. The three convention questions in §6, read off the public window.
5. Does the existing decoder on `no-training-approach` already run on BAR-ATH, and what does it
   score on the public window as a correctness check? That is the number the staged pipeline
   starts from.
