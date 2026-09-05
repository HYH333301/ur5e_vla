#!/bin/bash
# Pull the new-scene dataset, compute norm stats, then launch LoRA training.
# Markers: CLEAN-DONE / NORM-DONE|NORM-FAIL / TRAIN-LAUNCHED
source /root/shared-nvme/code/env.sh
cd /root/shared-nvme/code/openpi-main

rm -rf /root/shared-nvme/hf_data/hub/datasets--hyh1234--ur5e_vla_lerobot \
       /root/shared-nvme/hf_data/lerobot
echo CLEAN-DONE

ok=0
for i in 1 2 3; do
  if /root/.local/bin/uv run scripts/compute_norm_stats.py --config-name pi05_ur5e_lora; then
    ok=1; echo NORM-DONE; break
  fi
  echo "norm attempt $i failed, retrying in 20s"
  sleep 20
done
[ "$ok" = 1 ] || { echo NORM-FAIL; exit 1; }

bash /root/shared-nvme/code/ur5e_vla/cloud/train.sh pi05_ur5e_lora 8 train_newscene.log
echo TRAIN-LAUNCHED
