# Flow-GRPO training dataset

This dataset was built from the meeting archives in data/.

- manifest.json: directly accepted by scripts/training_scripts/flow_grpo.py
- lr/: 120x120 conditioning images
- preferred/: aligned 480x480 positively rated or expert-accepted references
- metadata.json and metadata.jsonl: provenance and selection rationale
- eaction_annotations.jsonl: all source likes and dislikes, including excluded images
- contact_sheet.jpg: visual quality-control overview of selected references

Only images marked like in reaction CSVs and four unambiguous selections from the
14 August expert protocol are included. Dislikes and ambiguous recommendations were
excluded because the current Flow-GRPO loader has no negative-image field.

Non-square source artwork is center-cropped with the same fill-and-crop behavior used
by ImageOps.fit in the trainer. Each LR image is downsampled from its exact preferred
reference, so every pair is spatially aligned and contains no synthetic letterbox bars.

Records: 20
Reaction annotations: 16 likes and 36 dislikes