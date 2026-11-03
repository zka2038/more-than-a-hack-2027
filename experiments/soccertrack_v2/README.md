**SoccerTrack v2 model experiment**

This workspace is ready for a teammate or coding agent to implement the pilot in [the team handoff](../../MODEL_TRAINING_HANDOFF.md). Start with a person detector, compare pretrained and fine-tuned predictions on separate matches, and save a reproducible result. Read [the data terms](../../SOCCERTRACK_DATA_TERMS.md) before using or sharing the footage.

The directory and initial configuration are prepared. Training code, data downloads and measured results are not present yet. Run commands from the repository root; paths in `config.json` are relative to that root.

| Location | Purpose |
|---|---|
| `config.json` | Dataset release, match splits, sampling, seed and epoch limit |
| `src/` | Model and data-adapter implementation |
| `scripts/` | Commands for auditing, preparation, training, evaluation and inference |
| `tests/` | Checks for alignment, box transforms, missing labels and split leakage |
| `reports/` | Small, shareable result summaries and provenance records |
| `data/external/soccertrack_v2/` at repo root | Local downloaded/prepared data; Git ignored |
| `outputs/soccertrack_v2/` at repo root | Checkpoints, predictions and previews; Git ignored |

**A teammate or agent can download the training data after authentication.** Hugging Face grants gated access to individuals. Each direct downloader needs their own accepted access to [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2). An agent can use that person's saved local login or an injected `HF_TOKEN`; sharing the project owner's token is unnecessary. [Access documentation](https://huggingface.co/docs/hub/datasets-gated).

Use Python 3.12+ and an isolated environment. These setup commands are for macOS/Linux:

```bash
python3 -m venv experiments/soccertrack_v2/.venv
source experiments/soccertrack_v2/.venv/bin/activate
python -m pip install huggingface_hub
hf auth login
hf auth whoami
```

Authenticate interactively on the download machine. For an unattended runner, provision a token with read access through its secret manager. Never paste credentials into chat, commit them, or print them in logs. Record installed versions in the experiment's eventual dependency lock. [CLI authentication](https://huggingface.co/docs/huggingface_hub/guides/cli).

Before downloading media, have the agent resolve `v1.2` to a commit SHA and inspect the repository file listing and sizes using the Hugging Face API. Select only verified paired MOT clips, labels and metadata for the configured matches. Record those exact paths and the resolved revision in `config.json`; its file list starts empty deliberately. Check available disk space against the selected download size and space needed for extracted frames.

This is a command template: replace both uppercase placeholders with verified values. Add further explicit filenames before the options when needed.

```bash
hf download atomscott/soccertrack-v2 "VERIFIED_REPOSITORY_FILE_PATH" \
  --repo-type dataset --revision "RESOLVED_COMMIT_SHA" \
  --local-dir data/external/soccertrack_v2 --dry-run
```

Inspect the preview, then run the same command without `--dry-run`. Use exact filenames or narrowly checked include patterns; omitting a selection can download the whole dataset. The Python equivalents are `hf_hub_download` for one file or `snapshot_download` with explicit `allow_patterns`. [Download documentation](https://huggingface.co/docs/huggingface_hub/guides/download), [CLI options](https://huggingface.co/docs/huggingface_hub/guides/cli).

Complete the data/annotation audit before implementing training. Follow the handoff's six steps, then add the actual runnable commands here. Keep the detector's software and weight licences in the provenance record. Put large artifacts in the ignored output directory and add a small results report with their locations and checksums. Preserve the existing submission pipeline; this pilot does not generate challenge possession predictions.
