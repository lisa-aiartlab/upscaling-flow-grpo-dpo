# Flow-GRPO for image upscaling

This repository trains a LoRA adapter for
`black-forest-labs/FLUX.2-klein-4B` with grouped relative policy optimization
(Flow-GRPO). Low-resolution images are supplied as FLUX image references and
the generated target is four times larger on each side. The committed checkout
contains the code and both training datasets; model weights and Python packages
are downloaded on the target machine.

## Included datasets

- `flow_grpo_dataset/manifest.json`: 20 curated LR/preferred-image pairs.
- `flow_grpo_dataset/upscaling_dataset/manifest.json`: 21 HR images with six
  degradation pipelines. The loader expands this fetched dataset to 126
  trainable LR/HR pairs and it is the default used by the launch scripts.

Generated checkpoints, caches, virtual environments, raw source archives and
inference results are intentionally excluded from Git.

## VM requirements

- Linux or Windows with an NVIDIA CUDA-capable GPU. The default profile targets
  NVIDIA V100/Volta (`sm_70`) with 48 GB VRAM.
- A recent NVIDIA driver compatible with the selected PyTorch CUDA wheel.
- Python 3.10, 3.11 or 3.12 with `venv` support.
- At least 35 GB of free disk space for the environment, model cache and
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

The setup script installs pinned, mutually compatible dependencies: PyTorch
2.6.0, Diffusers 0.40.0, Transformers 5.15.1 and PEFT 0.20.0. The default
PyTorch wheel uses CUDA 11.8 for broad V100/driver compatibility. To use
another official PyTorch 2.6 wheel index, set `TORCH_INDEX_URL` before setup:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_vm.sh
```

The default FLUX.2 Klein model revision is also pinned. Pass
`--model-revision <commit-or-tag>` explicitly only when intentionally changing
the base model; checkpoints record it and reject an incompatible resume.

V100 does not support bf16, so fp16 is the default for setup, smoke, training
and inference. `GradScaler` remains enabled to protect fp16 updates from
underflow. The following is therefore optional but shows the explicit setting:

```bash
FLOW_GRPO_MIXED_PRECISION=fp16 bash scripts/setup_vm.sh
FLOW_GRPO_MIXED_PRECISION=fp16 bash scripts/run_smoke_training.sh
```

The validator also checks that the installed PyTorch wheel contains `sm_70`
kernels. Selecting `bf16` on a V100 fails immediately with a clear error before
the model is downloaded or training begins.

The smoke run exercises model loading, LoRA injection, image conditioning, the
real CLIP/metric reward path, backward, optimizer step and checkpoint writing
on one sample with two generated candidates. It writes
`flow_grpo_smoke_output/final/flow_grpo_lora.pt`.

## Full Linux training

Train on all 126 fetched LR/HR pairs. The conservative default group size is
2; raise it only after the smoke test succeeds with enough free VRAM:

```bash
bash scripts/run_training.sh
```

Useful environment overrides:

```bash
FLOW_GRPO_GROUP_SIZE=4 \
FLOW_GRPO_MIXED_PRECISION=fp16 \
FLOW_GRPO_OUTPUT_DIR=flow_grpo_output_v100 \
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
bash scripts/run_training.sh \
  --epochs 3 \
  --grpo-epochs 1 \
  --group-size 2 \
  --resolution 128 \
  --learning-rate 5e-6 \
  --max-grad-norm 1.0 \
  --lora-rank 8 \
  --save-every 10
```

## Resume and ACPI shutdown checkpoints

Every checkpoint contains the LoRA weights, optimizer and GradScaler state,
random-number-generator states, and the next epoch/dataset position. Resume
from either a checkpoint directory or its `flow_grpo_lora.pt` file:

```bash
bash scripts/run_training.sh \
  --epochs 3 \
  --resolution 128 \
  --learning-rate 5e-6 \
  --resume-from-checkpoint flow_grpo_output/checkpoint-25
```

The Linux launcher uses `exec`, so systemd's `SIGTERM` reaches the Python
trainer directly. On `SIGTERM` (ACPI shutdown) or `SIGINT`, training stops at
the nearest safe point and atomically writes:

```text
flow_grpo_output/shutdown/flow_grpo_lora.pt
```

Restart it with the same training parameters and the shutdown checkpoint:

```bash
bash scripts/run_training.sh \
  --epochs 3 \
  --resolution 128 \
  --learning-rate 5e-6 \
  --resume-from-checkpoint flow_grpo_output/shutdown
```

For a VM platform that allows 30 seconds after ACPI shutdown, configure
systemd not to wait longer than that before stopping services:

```ini
# /etc/systemd/system.conf
DefaultTimeoutStopSec=30s
```

Apply the systemd setting before starting the training process. Checkpoints are
written through a temporary file and renamed atomically, so an existing
checkpoint is not corrupted if the VM is forcibly powered off during a write.
The handler cannot save while a long-running CUDA kernel is still executing;
use periodic checkpoints as an additional safeguard.

## Windows PowerShell

```powershell
.\scripts\setup_vm.ps1
.\scripts\run_smoke_training.ps1
.\scripts\run_training.ps1 -Epochs 3 -GroupSize 4
.\scripts\run_training.ps1 -Epochs 3 -ResumeFromCheckpoint flow_grpo_output\shutdown
```

V100 uses `-MixedPrecision fp16`, which is already the default. For an
H100/A100-class GPU, bf16 remains available explicitly. Use `-MaxSamples` for a
short run and `-LrPipeline LR_06_realistic` to select one degradation type.

## Validate an existing environment

```bash
.venv/bin/python scripts/validate_setup.py
```

The validator checks exact dependency versions, required FLUX.2 APIs, free
disk, every dataset image, CUDA, VRAM and bf16 support. On a CPU-only machine,
dataset and import validation can still be run with `--allow-no-cuda`.

## Inference with a trained checkpoint

```bash
.venv/bin/python scripts/flow_grpo_inference.py \
  --image flow_grpo_dataset/upscaling_dataset/LR_06_realistic/image_0001.png \
  --prompt "An eighteenth-century decorative hunting scene" \
  --checkpoint flow_grpo_output/final/flow_grpo_lora.pt \
  --output inference_output/upscaled.png
```

The first training or inference run downloads FLUX.2 Klein 4B and the CLIP model
into `.cache/`. Copying only the committed repository is therefore sufficient;
there is no need to copy the local `.venv` or `.cache` directories.
