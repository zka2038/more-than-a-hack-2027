# Decoder results log

The iteration metric is spellF1 on DFL open matches (CC BY 4.0), using tracking degraded to look like the challenge's automatic tracking (`degrade.py`: ~42% ball missing in bursts, player dropouts, jitter, brief wrong-team ghosts). The reference labels are DFL's own per-frame BallStatus/BallPossession.
- **Dev matches:** J03WMX, J03WN1, J03WOH, J03WOY, J03WPY.
- **Held back** and scored once at the end: J03WR9, J03WQQ.
- **LOMO** means each dev match is scored with weights tuned on the other four.

This is a stand-in. DFL's possession flags are not Metrica's, and DFL changes possession on first touch, which gives about 500 possessions per match. Use it to compare versions, not to predict the board score.

**Ceilings measured on match J03WMX, with stoppages taken from the reference labels:**

| team rule | clean tracking | degraded tracking |
|---|---|---|
| last touch, no decoder | 0.57 | – |
| touch evidence + Viterbi | 0.69 | 0.50 |

| # | change | dev mean spellF1 | dev mean macroF1 | kept |
|---|---|---|---|---|
| 0 | starter baseline (nearest player + majority vote, ±2 s) on the same degraded data | 0.205 | 0.383 | reference |
| 1 | proximity evidence + out-of-pitch / low-speed D + sticky Viterbi (2 matches, untuned) | 0.078 | 0.433 | yes (scaffold) |
| 2 | + still-ball, long-gap and gap-at-line D features; coordinate search (4 matches, train fit) | 0.351 | – | yes |
| 3 | + touch evidence (proximity plus velocity change or slow ball) and a carried last-touch team | 0.422 (defaults) | 0.725 | yes |
| 3b | same as 3, tuned (LOMO, coordinate search) | **0.436** | 0.785 | yes |
| 4 | + straight-line ball interpolation through gaps of up to 6 s, for team evidence only (D features still see the raw gaps). LOMO at 0/50/150/300 frames/unlimited: .436/.447/**.471**/.462/.460 | **0.471** | 0.832 | yes |
| 5 | + control evidence (a player within ~1 m moving with the ball). Alone it catches only 20–30% of hand-overs, with few false alarms. LOMO 0.469 / 0.465 from two starting points | 0.469 | 0.812 | **reverted** (no gain) |
