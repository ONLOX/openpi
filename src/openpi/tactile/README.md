# Tactile experiments

The current baseline directly reuses openpi's `TrainConfig`, `Pi0Config`,
`PI0Pytorch`, data loader, policy, and trainer. Tactile code only aligns the
sidecar and concatenates five normalized fingertip forces to the 27-d robot
state before π0.5 tokenization.

N₀-VTLA and T-Rex should get dedicated model/policy code when implemented,
because they change the model graph and inference schedule.

## Commands

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/tactile/train.py pi05_kaihand_card_concat

python scripts/tactile/eval.py \
  --config-name pi05_kaihand_card_concat
```
