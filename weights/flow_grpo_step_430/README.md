# Flow-GRPO LoRA weights — step 430

`flow_grpo_lora_step_430.pt` contains the inference-ready transformer LoRA
weights saved after global training step 430. It was exported from the complete
periodic checkpoint without optimizer, GradScaler, or RNG state, so it is not a
resume checkpoint.

Base model: `black-forest-labs/FLUX.2-klein-4B`  
Base revision: `e7b7dc27f91deacad38e78976d1f2b499d76a294`  
LoRA rank: `8`  
SHA-256: `34d5bd0fd39d5d14acfcf1dcc45cc408b0445ad3357e624348f08b051e55cd5c`

Run inference from the repository root:

```bash
.venv/bin/python scripts/flow_grpo_inference.py \
  --image flow_grpo_dataset/upscaling_dataset/LR_06_realistic/image_0001.png \
  --prompt "An eighteenth-century decorative hunting scene" \
  --checkpoint weights/flow_grpo_step_430/flow_grpo_lora_step_430.pt \
  --output inference_output/upscaled-step-430.png
```

See `training_metadata_step_430.json` for the complete training configuration,
progress position, source-checkpoint hash, and exported-file hash.
