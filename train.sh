export WANDB_DIR='/oss/yeqianyu'
export WANDB_ENTITY='yeqianyu'

ROOT_DIR="/cpfs_infra/user/yeqianyu/repos/openpi"
CONDA_ENV="/cpfs_infra/user/yeqianyu/conda_envs/openpi"
TORCHRUN="${CONDA_ENV}/bin/torchrun"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

"${TORCHRUN}" --standalone --nproc-per-node=8 \
  scripts/tactile/train.py pi05_kaihand_bulb_concat \
  --overwrite
