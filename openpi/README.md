# UR5e × openpi（π0/π0.5）：采集 → 转换 → 云端训练 → 部署 → 仿真闭环

本目录把本仓库的 MuJoCo 采集数据接入 openpi 全流程。分工：**本地（Windows）采集、转换、闭环评测；云端（Linux GPU）训练**。openpi 官方只支持 Ubuntu，训练勿在本地跑。

## 目录结构

| 路径 | 说明 |
|---|---|
| `convert_ur5e_data_to_lerobot.py` | HDF5 回合 → LeRobot 数据集（pinned lerobot） |
| `check_dataset.py` | 数据集完整性/对齐校验（可对比原始 HDF5 逐帧） |
| `eval_client.py` | MuJoCo 闭环评测客户端（连 serve_policy 推理） |
| `download_data.py` | 从 HF 私有仓库拉取 LeRobot 数据集（~2.9 GB，不进 git） |
| `patches/` | 打进 openpi 源码的 3 个补丁文件 + 一键应用/回传脚本 |
| `cloud/` | 云容器（paratera）运维：SSH helper、env 模板、权重抓取、训练启动 |

## openpi 补丁机制

openpi 里有两类改动，源码内的 3 个文件以完整拷贝形式存放在 `patches/files/`：

| 补丁文件 | 内容 |
|---|---|
| `src/openpi/policies/ur5e_policy.py` | 新增：三相机的输入/输出变换（Ur5eInputs/Outputs） |
| `src/openpi/training/config.py` | 修改：`LeRobotUr5eDataConfig` + 训练配置 `pi05_ur5e`、`pi05_ur5e_lora` |
| `scripts/serve_policy.py` | 修改：`EnvMode.UR5E` 注册 + 默认 checkpoint |

给一份新 openpi 代码打补丁 / 把 openpi-main 里的手工改动收回本仓库：

```bash
python patches/apply_patches.py                       # 应用到 D:/code/openpi-main
python patches/apply_patches.py --export              # 从 openpi-main 回传到 patches/files/
```

**注意：直接在 openpi-main 里改了源码后，记得跑 `--export` 同步回来再提交。**

## 数据语义（训练与推理必须一致）

- `state` (7,) = 6 关节位置 + 夹爪驱动关节位置
- `action` (7,) = 6 关节位置目标 + 夹爪指令 [0,1]，**绝对值**
- 相机：`image`=front、`side_image`=side、`wrist_image`=wrist（320×240 @20 Hz）→ 模型 base / left_wrist / right_wrist 槽位（mask 全开）
- 配置里 `DeltaActions(make_bool_mask(6, -1))`：6 关节维自动转增量，夹爪维保持绝对；推理端 `AbsoluteActions` 自动还原
- prompt = 每回合 `instruction`（`prompt_from_task=True`）

## ① 本地采集

见仓库根 README：`scripts/teleop_collect.py`（键盘遥操作）、`scripts/collect.py`（脚本专家），输出 HDF5 到 `data/`。

## ② 本地转换：HDF5 → LeRobot

openpi 钉了 lerobot 的特定 commit，转换必须用同一版本 + `datasets==3.6.0`（新版两侧各坏一次：写入的 parquet 元数据旧版读不了；旧代码跑在 datasets 5.x 下 TypeError）。独立 venv，别装进采集环境 graspfruit：

```bash
python -m venv D:/code/ur5e_vla/.venv-lerobot
D:/code/ur5e_vla/.venv-lerobot/Scripts/pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" "datasets==3.6.0" h5py pillow tyro

set HF_LEROBOT_HOME=D:/code/ur5e_vla/data/lerobot
D:/code/ur5e_vla/.venv-lerobot/Scripts/python openpi/convert_ur5e_data_to_lerobot.py ^
  --data-dir D:/code/ur5e_vla/data/ur5e_pickplace D:/code/ur5e_vla/data/ur5e_teleop ^
  --repo-id hyh1234/ur5e_vla_lerobot --push-to-hub
```

默认跳过 `success=False` 回合。图像内嵌 parquet（无 mp4，`total_videos: 0`）。回读校验：

```bash
D:/code/ur5e_vla/.venv-lerobot/Scripts/python openpi/check_dataset.py --repo-id hyh1234/ur5e_vla_lerobot
```

从 HF 拉回本地（换机器/数据被删时）：`python openpi/download_data.py`。

## ③ 云端训练（paratera 容器，2×RTX 4090 24G）

SSH 信息在 `.env`；连容器：`python openpi/cloud/sshlib.py "<命令>"`（慢命令必须 nohup，paramiko 280s 会掐断 exec 通道）。

```bash
# 首次部署：openpi 代码上云 → 打补丁 → uv sync（lerobot/dlimp 改本地源，见 cloud/env.sh.example）
python openpi/cloud/sshlib.py "cd /root/shared-nvme/code/openpi-main && /root/.local/bin/uv sync"

# 归一化统计（assets/pi05_ur5e/hyh1234/ur5e_vla_lerobot/norm_stats.json）
uv run scripts/compute_norm_stats.py pi05_ur5e

# π0.5 base 权重（GCS 直连只有 250KB/s，用 JSON API + aria2 多连接 ~1.1MB/s）
nohup python openpi/cloud/gcs_fetch.py > /root/shared-nvme/logs/gcsfetch.log 2>&1 &

# 训练（nohup 防断连；命令是子命令风格，不是 --config-name！）
bash openpi/cloud/train.sh 8 train.log
tail -f /root/shared-nvme/logs/train.log
```

### 24G 显存结论（5 次试错换来的，别再踩）

- π0.5 LoRA 在 4090 24G 上：batch 32/16/8 及 FSDP 双卡 batch32 **默认全部 OOM**
- 根因：XLA 默认只预留 75% 显存（≈18.3GiB），而编译后单步峰值 ~18.8GiB，差一点点
- **解法：`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` + 单卡 batch 8**（train.sh 已内置；GPU0 占 23.7G，1.5s/步，20k 步 ≈ 8.3h）

## ④ 部署 + 仿真闭环评测

云端起推理服务（4090 单卡，单次推理 ~0.3s）：

```bash
uv run scripts/serve_policy.py ur5e    # 默认读 checkpoints/pi05_ur5e_lora/exp/20000
```

本地 SSH 隧道打通端口（容器端口不对外）：`ssh -L 8000:localhost:8000 <容器>`，然后：

```bash
python openpi/eval_client.py --host localhost --port 8000 --episodes 20 --video-out-path data/rollouts
```

客户端自动加载本仓库 `src/` 的 `Ur5eEnv`（`--ur5e-src` 可改），20 Hz 执行，每 5 步重规划（动作块长 10），逐回合打印成功率并可选录像。

## 调参入口

- 动作改回绝对值：`LeRobotUr5eDataConfig(use_delta_joint_actions=False)`
- 换 π0 基座：config.py 里 `model=Pi0Config(...)`（weight_loader 相应换 pi0_base，需重新下权重）
- 相机减到两路：`ur5e_policy.py` 里把 side 换成零填充 + mask=False
