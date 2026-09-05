# 训练与评估结果（新场景）

**新场景**：3 物体（红方块 / 绿球 / 蓝圆柱）+ 2 托盘（黄 / 紫），位置随机、
颜色固定、指令从 5 个模板采样（任务 = 3 物体 × 2 托盘）。

**数据**：51 回合专家先行采集（全部 expert 独力成功，均值 216 步），10993 帧，
HF 仓 `hyh1234/ur5e_vla_lerobot`（2026-09-05 重建推送）。

## 进行中

| 日期 | 配置 | 说明 |
|---|---|---|
| 2026-09-05 | `pi05_ur5e_lora` | LoRA(rank16/32) batch 8 × 20k 步，单卡 4090，1.3 s/it（日志 `/root/shared-nvme/logs/train_newscene.log`） |

## 待办

- [ ] 训练完成后：loss 曲线（`train/extract_metrics.py`）
- [ ] 闭环评估（`train/eval_client.py`，建议 50 回合）

---

旧场景（单方块 → 单目标圆盘；混合数据 72–76%、scripted-only 对照 74%、
零样本 0/3）的结果表与训练曲线见 git 历史（commit `977185d` 及更早）。
