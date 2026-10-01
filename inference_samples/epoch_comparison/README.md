# Epoch checkpoint comparison

This directory contains the user-provided low-quality input image and deterministic
Flow-GRPO upscaling results generated with the same prompt and seed at checkpoints
closest to the ends of training epochs.

Epoch boundaries were steps 126, 252, 378, 504, and 630. The available checkpoints
used for comparison are steps 120, 250, 380, 500, and the final step 630.

- Resolution: 128 input conditioning, 512 output
- Inference steps: 4
- Guidance scale: 1.0
- LoRA scale: 1.0
- Seed: 42
- Prompt: `A restored vintage circular lacquer miniature depicting a graceful woman dancing, wearing a flowing orange skirt and pale blouse, dark background, faithful composition, preserved historical painted details`

Generated outputs:

- `epoch_01_step_120.png`
- `epoch_02_step_250.png`
- `epoch_03_step_380.png`
- `epoch_04_step_500.png`
- `epoch_05_step_630.png`
