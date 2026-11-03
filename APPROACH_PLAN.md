**Working plan — More than a Hack 2027**

You registered on 3 November and are waiting for the match footage. The current stage is preparation. We can get the basic tools ready now, then make modeling decisions after inspecting the supplied tracking and tags. The delivery date of the assets is not yet known.

**The task is to recover complete possessions.**

For every video frame, our system must say home possession, away possession or dead ball. It can also abstain with X. The main score rewards correctly recovered possession intervals, so a prediction that constantly switches teams can perform poorly even when many individual frames are correct.

There are four core problems:

1. **The ball often disappears from tracking.** The repository says it is missing on about 41% of graded frames. We need to carry useful information through gaps without assuming the game has stopped.

2. **Being near the ball does not mean controlling it.** A pass, deflection or tackle can put an opponent closest to the ball temporarily. We need evidence of actual control before changing possession.

3. **Stoppages and restarts matter.** We must distinguish live play from dead ball. If the same team has possession before and after a stoppage, those are still two separate possessions.

4. **Predictions need continuity and accurate boundaries.** Too little smoothing invents turnovers. Too much smoothing removes real short possessions. The system needs to remember its current state while still responding to convincing changes.

**While we wait, prepare the tools and validation.**

- The existing scorer has already been checked against the supplied baseline files.
- The submission pipeline is ready in `solution/`: it writes a full-length all-X placeholder and runs the official validator. The generated file passed validation, and four automated contract checks passed. Usage is in `solution/README.md`.
- Create artificial examples of missing balls, passes, turnovers and same-team restarts to check the logic without match footage.
- Identify suitable data from other matches for testing improvements. The public five-minute labels in this repository may check pipeline correctness, but cannot be used to train or tune the model.

Success at this stage means the tools work and we have a defensible way to test ideas. It does not mean we have measured possession accuracy on the competition match.

**Once the data arrives, start with tracking and automatic tags.**

1. **Verify the files and timing.** Check frame counts, home/away mapping, coordinate units and alignment. The final output uses the tactical video's frame index.

2. **Regenerate the supplied baseline.** Confirm that our environment can reproduce the reference predictions from tracking.

3. **Inspect observation quality.** Measure how long ball gaps last, how reliable player tracks are, and where automatic tags overlap or conflict. Do not manually label or correct the match.

4. **Build a simple first model.** Estimate which team controls the ball and whether play is live, then combine those signals with memory of the previous state.

5. **Test one improvement at a time.** Use other-match validation to decide whether better gap handling, restart detection or temporal modeling helps.

The initial approach is:

`tracking + automatic tags → control and live/dead evidence → continuous possession sequence → validated CSV`

Smart Tagging should provide supporting evidence. Its intervals are approximate, and a defensive phase names the defending team rather than necessarily the team with possession.

**Keep the first version focused.**

Use the supplied tactical tracking first. Full video analysis, replacing the tracker and synchronizing the broadcast feed can wait until we identify a specific limitation they would solve. A basic version can use tracking and tags without decoding the footage.

We do not yet have evidence that a large neural model is necessary. Begin with transparent rules and state memory; consider a more structured temporal decoder once there is a working reference and a way to compare results.

**Deliver in stages, rather than assuming when the footage will arrive.**

- **Before access:** working tools, synthetic checks and a validation plan.
- **After access:** verified data, regenerated baseline and a simple complete model.
- **Then:** independently tested improvements, followed by a method freeze.
- **Before submission:** reproduce the exact CSV, validate it, and finish the public code repository and short technical report.

The repository lists 17 November 2026 as the deadline; the cutoff time and timezone still need checking in the organizer's communications. If access arrives late, reduce optional experiments and protect time for reproduction and submission.

Competition assets must remain out of the public repository and hosted/cloud LLMs. The detailed code findings, full 35-item problem inventory and implementation options are preserved in `APPROACH_DETAILS.md` for when they become relevant.
