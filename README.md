# Flow-GRPO for image upscaling

This repository trains a LoRA adapter for
`stabilityai/stable-diffusion-x4-upscaler` with grouped relative policy
optimization (Flow-GRPO). The committed checkout contains the code and both
training datasets; model weights and Python packages are downloaded on the
target machine.

## Included datasets

- `flow_grpo_dataset/manifest.json`: 20 curated LR/preferred-image pairs.
- `flow_grpo_dataset/upscaling_dataset/manifest.json`: 21 HR images with six
  degradation pipelines. The loader expands this fetched dataset to 126
  trainable LR/HR pairs and it is the default used by the launch scripts.

Generated checkpoints, caches, virtual environments, raw source archives and
inference results are intentionally excluded from Git.

## VM requirements

- Linux or Windows with an NVIDIA CUDA-capable GPU.
- A recent NVIDIA driver.
- Python 3.8 through 3.11 with `venv` support (3.10 or 3.11 recommended).
- At least 20 GB of free disk space for the environment, model cache and
  checkpoints. More space is useful for long runs.
- Internet access on first setup to download packages and Hugging Face models.

Training currently uses one CUDA GPU per process. A GPU with more VRAM permits
larger groups and is more useful than adding GPUs to the same process.

## Linux setup and smoke test

From the repository root:

```bash
bash scripts/setup_vm.sh
bash scripts/run_smoke_training.sh
```

The setup script installs the CUDA 12.4 build of PyTorch 2.4.1 by default. To
use another official PyTorch wheel index, set `TORCH_INDEX_URL` before running
it:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_vm.sh
```

The smoke run trains one sample with two generated candidates and writes
`flow_grpo_smoke_output/final/flow_grpo_lora.pt`.

## Full Linux training

Train on all 126 fetched LR/HR pairs:

```bash
bash scripts/run_training.sh
```

Useful environment overrides:

```bash
FLOW_GRPO_GROUP_SIZE=4 \
FLOW_GRPO_MIXED_PRECISION=bf16 \
FLOW_GRPO_OUTPUT_DIR=flow_grpo_output_h100 \
bash scripts/run_training.sh
```

Train only on one degradation pipeline:

```bash
FLOW_GRPO_LR_PIPELINE=LR_06_realistic bash scripts/run_training.sh
```

Use the smaller curated preference dataset:

```bash
FLOW_GRPO_MANIFEST=flow_grpo_dataset/manifest.json bash scripts/run_training.sh
```

Any extra arguments are forwarded to `flow_grpo.py`, so command-line values can
override launcher defaults. For example:

```bash
bash scripts/run_training.sh --epochs 3 --learning-rate 5e-6 --save-every 10
```

## Windows PowerShell

```powershell
.\scripts\setup_vm.ps1
.\scripts\run_smoke_training.ps1
.\scripts\run_training.ps1 -Epochs 3 -GroupSize 4
```

For an H100/A100-class GPU, pass `-MixedPrecision bf16`. Use `-MaxSamples` for
a short run and `-LrPipeline LR_06_realistic` to select one degradation type.

## Validate an existing environment

```bash
.venv/bin/python scripts/validate_setup.py
```

On a CPU-only machine, dataset and import validation can still be run with
`--allow-no-cuda`.

## Inference with a trained checkpoint

```bash
.venv/bin/python scripts/flow_grpo_inference.py \
  --image flow_grpo_dataset/upscaling_dataset/LR_06_realistic/image_0001.png \
  --prompt "An eighteenth-century decorative hunting scene" \
  --checkpoint flow_grpo_output/final/flow_grpo_lora.pt \
  --output inference_output/upscaled.png
```

The first training or inference run downloads the base upscaler and CLIP model
into `.cache/`. Copying only the committed repository is therefore sufficient;
there is no need to copy the local `.venv` or `.cache` directories.
