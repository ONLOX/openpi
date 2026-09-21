# Tactile experiments

The current baseline directly reuses openpi's `TrainConfig`, `Pi0Config`,
`PI0Pytorch`, data loader, policy, and trainer. It reads the five embedded
fingertip normal forces, normalizes them with the dataset statistics, and
concatenates them to the 27-d robot state before π0.5 tokenization.

## Commands

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/tactile/train.py pi05_kaihand_usb_concat

python scripts/tactile/eval.py \
  --config-name pi05_kaihand_usb_concat
```
