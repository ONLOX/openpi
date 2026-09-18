export WANDB_ENTITY='yeqianyu'
source /oss/yeqianyu/miniconda3/bin/activate openpi
python scripts/tactile/train.py pi05_kaihand_card_concat \
  --exp-name card-concat \
  --overwrite
