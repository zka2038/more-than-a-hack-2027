**Team handoff for a SoccerTrack v2 test model**

Prepared 3 November 2026. The project owner has SoccerTrack v2 access. Teammates downloading from Hugging Face need their own account access. Build and run a small, reproducible visual-model experiment using those external matches. The eventual application is the More than a Hack possession challenge, but its match assets have not arrived.

Start with **one person detector fine-tuned on verified MOT clips**, comparing it with the same pretrained detector before fine-tuning. This exercises the footage-to-model pipeline with existing box annotations. Ball detection and possession prediction require separate labels and evaluation; they are follow-on work.

Work from the root of your clone of `more-than-a-hack-2027`. Use `experiments/soccertrack_v2/` for this pilot; its README and config are already prepared. This handoff is for a teammate or their coding agent. Inspect existing changes before editing and preserve the submission scaffold.

1. **Read the project constraints and preserve the working submission pipeline.**

   Read [RULES.md](RULES.md), [FAQ.md](FAQ.md), [solution/README.md](solution/README.md) and the [SoccerTrack terms review](SOCCERTRACK_DATA_TERMS.md).

   Other-match training and manual annotations are permitted when organizers can access them and the method is documented. Never manually label or correct the challenge fixture, use human-produced data about that fixture, or train/tune on its public five-minute labels. Keep competition-provided restricted assets and extracts out of a hosted coding assistant's context, including images and tool outputs. Git ignore rules do not control what a coding agent reads.

   The existing writer produces exactly 199,319 `frame,label` rows using H/A/D/X and invokes the official validator. Person boxes are not possession labels: do not feed detector output to this writer or fabricate a competition prediction. Preserve official evaluation code.

2. **Set up a small, versioned data subset.**

   Inspect available disk space, Python, FFmpeg and CPU/GPU support. Use a separate experiment environment and lock its dependencies so the working evaluation environment remains reproducible. Use the runner's own Hugging Face access and local authentication; never print, share or commit credentials. Follow the download workflow in [the experiment README](experiments/soccertrack_v2/README.md). Locate existing downloads before fetching more.

   Pin SoccerTrack release `v1.2` and record its resolved revision. Download only the paired four-minute MOT videos, annotations and metadata needed for the pilot. List candidate files and sizes first. The authors provide a [download helper with match and revision selection](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/setup.md); verify its selected paths instead of assuming match selection excludes full-length videos.

   Use this subset of the [official split](https://huggingface.co/datasets/atomscott/soccertrack-v2), provided matching MOT assets exist:

   | Purpose | Match IDs | Use |
   |---|---|---|
   | Training | 117092, 118575 | Fit weights |
   | Validation | 117093 | Select settings and checkpoint |
   | Test | 128057 | Evaluate once after settings are frozen |

   If an asset is absent, choose another match within the same official split and record the substitution. Never split neighboring frames, halves, crops or augmentations from one match across partitions. Start by sampling about two annotated frames per second for detection. Tracking evaluation, if added later, needs contiguous sequences.

3. **Audit the actual files before trusting a loader.**

   Produce a manifest with dataset revision, source path and checksum, match, sequence/half, clip offset when applicable, original frame index, timestamp, dimensions, annotation provenance and split. Preserve source IDs separately from internal zero-based frame indices.

   Probe decoded video geometry and frame rate. Verify that each annotation belongs to its paired clip, rather than assuming its indices refer to a full half. Check frame bounds, positive box sizes, duplicate rows, missing labels and ignore flags. Render box overlays at the beginning, middle and end of every selected sequence. Missing annotations mean unknown unless annotation coverage establishes a genuinely empty frame.

   The [MOT format](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-mot.md) describes one-based frames, pixel `x,y,width,height` boxes and a person class including players, goalkeepers and referees. Verify these against the files. Do not infer teams, roles or ball classes from track IDs. Preserve ignore semantics in evaluation; do not train ignored areas as ordinary background. Every resize, crop or tile must transform boxes consistently and preserve the inverse mapping.

   Upstream code needs particular care:

   - The [YOLO converter](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/data_utils/create_yolo_dataset.py) randomly splits frames and infers class from track-ID ranges. Do not reuse these choices. Check its frame offset and box scaling if borrowing code.
   - The [MOT baseline config](https://github.com/AtomScott/SoccerTrack-v2/blob/main/baselines/mot/config.yaml) does not match the current official split. Create our own explicit manifest/config.
   - The [GSR format notice](https://github.com/AtomScott/SoccerTrack-v2/blob/main/docs/format-gsr.md) reports schema and dimension discrepancies. The [loader](https://github.com/AtomScott/SoccerTrack-v2/blob/main/src/data_utils/soccertrack_v2.py) assumes an older schema and timestamp convention. If GSR/BAS is needed later, inspect the pinned files and write a validated adapter. Current v1.2 events have explicit frame alignment; do not automatically subtract 45 minutes from second-half timestamps. [Release card](https://huggingface.co/datasets/atomscott/soccertrack-v2).

4. **Train one modest baseline and measure it honestly.**

   Choose one lightweight pretrained detector compatible with available hardware. Record the architecture, weight source/checksum, software version and both code and weight licences. Prefer a small existing implementation over training from scratch. Use one `person` class initially.

   Evaluate the pretrained detector's person predictions first. Then verify one training batch, followed by a bounded fine-tuning run of at most ten epochs initially. Save the seed, resolved config, checkpoint and optimizer state. Inspect object size after preprocessing: aggressive full-panorama shrinking may erase useful detail. If using tiles, keep all tiles from a source frame together, transform boxes correctly, and merge predictions back into original-image coordinates before scoring.

   Compare before/after results on exactly the same validation frames, preprocessing and evaluation settings, with an explicit mapping between pretrained and fine-tuned person classes. Report AP50, AP50:95, precision/recall with the confidence threshold stated, frame/image counts, per-match results, inference speed, resolution, hardware and runtime. Retain representative success and failure overlays. After configuration freeze, evaluate both the pretrained baseline and selected checkpoint on identical held-out test frames in one final comparison; do not use that result to select another checkpoint.

   A successful pilot may show no accuracy gain. Report that result rather than extending experiments until a favorable number appears. If only a tiny CPU run is practical, label it a pipeline smoke test and state what larger run remains necessary.

5. **Keep ball and possession experiments separate.**

   SoccerTrack's `ball/` trajectories are interpolated between event anchors, not measured framewise ball positions. Do not turn them into supposed ground-truth image boxes or use them as independent ball-accuracy evidence. BAS supplies sparse action events rather than continuous possession labels. [Dataset description](https://huggingface.co/datasets/atomscott/soccertrack-v2).

   If the downloaded assets contain no verified ball boxes, finish the person baseline and prepare a small annotation specification for selected external clips: visible ball boxes or centers, visibility/occlusion, annotator and source frame. Leave occluded positions unknown. Do not silently invent labels or make manual annotation a prerequisite for finishing this first model.

   A later possession model needs continuous external-match labels following the challenge's control and dead-ball definitions. Carrying the last event's team forward creates heuristic pseudo-labels. Training and testing against those same heuristics cannot establish possession accuracy. Detector AP also does not establish a challenge-score improvement.

6. **Deliver something another person can rerun.**

   Prepared layout, relative to the repository root:

   | Path | Deliverable |
   |---|---|
   | `experiments/soccertrack_v2/src/` and `experiments/soccertrack_v2/scripts/` | Implement data audit/conversion, training, evaluation and inference |
   | `experiments/soccertrack_v2/config.json` | Initial matches, seed, sampling and training limits |
   | `data/external/soccertrack_v2/` | Local source files and prepared data; ignored by Git |
   | `outputs/soccertrack_v2/` | Checkpoints, manifests, metrics and overlays; ignored by Git |
   | `experiments/soccertrack_v2/README.md` and `reports/` | Setup/run commands, measured results, limitations and attribution |

   Keep a small, sanitized provenance summary and dependency lock in version control. Document where organizers can obtain the exact data subset and trained checkpoint. Data redistribution is permitted under its licence, but large videos do not belong in this Git checkout.

   Add meaningful checks for match leakage, frame/box conversion and ignored/missing annotation handling. Run the existing submission tests after integration changes. Demonstrate loading the saved checkpoint in a fresh process and producing repeatable predictions on the same input with a documented numerical tolerance.

   Finish with the commands executed, measured baseline/fine-tuned results, saved checkpoint location, data/weight licences and any remaining blocker. Do not claim training or evaluation ran when only code was written. Stop after this bounded pilot; leave a clear next experiment based on its results.
