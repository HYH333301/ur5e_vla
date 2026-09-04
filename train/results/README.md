# 训练与评估结果（2026-09-04）

pi05_ur5e_lora：π0.5 base + LoRA（gemma_2b_lora + gemma_300m_lora），70 回合自采集数据
（脚本专家 + 键盘遥操作），batch 8 单卡 4090，20k 步 ≈ 8h，checkpoint `exp/19999`。

## 闭环评估（MuJoCo，20 回合，20 Hz）

| 配置 | 成功率 | 说明 |
|---|---|---|
| 零样本 π0.5 底座（关节空间） | 0/3 | 原地小幅抖动 |
| 零样本 π0.5 底座（EEF 空间探针 `pi05_ur5e_eef`） | 0/3 | 输出"保持不动"（目标偏离 tcp 4.5 mm），视觉域不匹配 |
| **LoRA 微调（本仓库数据）** | **15/20 = 75%** | 成功回合 7–12 s 完成；失败均为 30 s 超时（疑边缘位置抓偏） |

复现：云端 `serve_policy.py`（config=pi05_ur5e_lora，dir=checkpoints/pi05_ur5e_lora/exp/19999）
+ 本地 `train/eval_client.py`，详见 `train/README.md` ④。

## 训练曲线

`train_metrics_pi05_ur5e_lora.csv`（每 100 步，200 点：step/grad_norm/loss/param_norm/lr）/
`train_loss_pi05_ur5e_lora.png`：loss 0.079 → 0.0033，平滑收敛；wandb 未开启，曲线从训练日志抽取。
lr 未随步打印，按 openpi 默认 `CosineDecaySchedule`（warmup 1000 → peak 2.5e-5，cosine 衰减至
2.5e-6 @30k 步，训练在 20k 步结束，末端 ≈8.6e-6）由 step 重构。

`train_config_pi05_ur5e_lora.json`：服务器解析后的完整 TrainConfig（模型 LoRA 变体、AdamW 超参、
freeze_filter、数据管线、seed=42 等）。注意其中 `batch_size: 32` 是默认值，实际运行经
`train.sh` 传了 `--batch-size 8`（显存原因）。

## 显存结论（2×RTX 4090 24G，供复现参考）

默认配置必 OOM；根因是 XLA 默认只预留 75% 显存而编译后峰值 ~18.8 GiB。
解法：`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` + 单卡 batch 8（`train/cloud/train.sh` 已内置）。
