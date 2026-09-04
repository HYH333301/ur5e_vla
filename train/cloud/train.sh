#!/bin/bash
# Launch openpi training on the paratera container, detached from SSH.
# Run ON the cloud:  bash train.sh [config_name] [batch_size] [train_log_name]
#
# Hard-won memory settings for 2x RTX 4090 24G (see train/README.md):
#   - XLA default reserves only 75% of VRAM (~18.3GiB) but the compiled step
#     peaks at ~18.8GiB -> every run OOMs. XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 is REQUIRED.
#   - batch 8 on a single GPU fits (23.7GiB). FSDP 2-GPU also OOMs (fragmentation).
CONFIG=${1:-pi05_ur5e_lora}
BS=${2:-8}
LOG=${3:-train.log}
nohup bash -c "
source /root/shared-nvme/code/env.sh &&
cd /root/shared-nvme/code/openpi-main &&
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  /root/.local/bin/uv run scripts/train.py $CONFIG \
  --exp-name exp --overwrite --batch-size $BS --no-wandb_enabled
" > /root/shared-nvme/logs/$LOG 2>&1 < /dev/null &
disown
echo "LAUNCHED config=$CONFIG bs=$BS log=/root/shared-nvme/logs/$LOG"
