# UR5e + Robotiq 2F-85 VLA 数据采集（MuJoCo）

在 MuJoCo 仿真里为 UR5e + Robotiq 2F-85 抓放任务采集 VLA 训练数据：每回合随机
方块/目标位置与颜色，双通道采集（脚本专家 / 键盘遥操作），统一 HDF5 格式，
本地采集 → 云端训练。

## 环境

- Python 3.11，依赖：`mujoco>=3.11` `numpy` `h5py` `pillow`

```bash
conda create -n graspfruit python=3.11 -y
conda activate graspfruit
pip install mujoco numpy h5py pillow
```

## 使用

```bash
# 脚本专家批量采集（IK 航点 + 关节插值，自动重试，只存成功回合）
python scripts/collect_scripted.py --episodes 50 --out data/ur5e_pickplace --seed 0

# 键盘遥操作采集（按键见下表）
python scripts/collect_teleop.py

# 数据检查：统计 + 8 帧三相机接触表图
python scripts/replay.py data/ur5e_pickplace/episode_0000.hdf5 --samples 8
```

### 遥操作按键（在 MuJoCo 窗口内按键）

| 按键 | 功能 | 按键 | 功能 |
|---|---|---|---|
| ↑/↓ | 目标 ±y（远离/靠近你） | ←/→ | 目标 ∓x（左/右） |
| W/S | 目标 升/降（8/2 也可用） | G | 夹爪 开/合 切换 |
| -/= | 步长 −/+（1/2.5/5 cm） | R | 目标重置到当前指尖 |
| Enter | 完成回合→判定并保存→自动下一回合 | K | 强制保存本回合 |
| PgDn | 放弃本回合并重置 | | |

- 鼠标拖动转视角、滚轮缩放；绿色小球 = IK 目标位置（不会进采集图像）
- 按键为步进式：每按一次目标移动一步（viewer 只回传单击事件，不回传按住）
- 其余字母/数字多为 viewer 保留键（切线框/隐藏网格/切相机），误按画面最多闪一帧即自动恢复
- 保存时自动压缩停顿空闲步（`--no-trim` 关闭）

## 数据格式（每回合一个 HDF5）

- `observations/`：`qpos qvel gripper tcp_pos tcp_quat cube_pos`（T+1 步）+
  `image_{front,side,wrist}_cam`（T+1 帧，320x240 JPEG，变长 uint8）
- `action`（T,7）：6 关节位置目标 + 夹爪指令 [0,1]，20 Hz
- `phase`（T+1）：专家阶段 id；遥操作回合为 -1
- attrs：`success` `instruction`（如 "pick up the red cube..."）`cube_rgba`
  `target_rgba` `source`（scripted/teleop）

## 接入 openpi 训练（π0 / π0.5）

**职责分工**：本地（本仓库）= 采集 + 转换 + 仿真闭环评估；云端（GPU 容器）= 训练 + 推理服务。
全流程代码在 [`train/`](train/)，详见 [`train/README.md`](train/README.md)，当前结果见
[`train/results/README.md`](train/results/README.md)（LoRA 微调后 MuJoCo 抓放 **75%**）。
openpi 本体在 `D:\code\openpi-main`（zip 快照无 .git），
UR5e 的 3 个源码补丁由 [`train/patches/apply_patches.py`](train/patches/apply_patches.py) 一键应用/回传。

```bash
# 一次性：独立 venv 装 pinned lerobot（不要装进 graspfruit）
python -m venv .venv-lerobot
.venv-lerobot/Scripts/pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" "datasets==3.6.0" h5py pillow tyro

# HDF5 -> LeRobot（--push-to-hub 传到 hyh1234/ur5e_vla_lerobot；数据集可用 train/download_dataset.py 拉回）
set HF_LEROBOT_HOME=D:/code/ur5e_vla/data/lerobot
D:\code\ur5e_vla\.venv-lerobot\Scripts\python train\convert_to_lerobot.py ^
  --data-dir D:\code\ur5e_vla\data\ur5e_pickplace D:\code\ur5e_vla\data\ur5e_teleop ^
  --repo-id hyh1234/ur5e_vla_lerobot
```

云端（paratera 容器）训练/权重抓取/监控脚本在 [`train/cloud/`](train/cloud/)；
评测客户端是 [`train/eval_client.py`](train/eval_client.py)。

**本地 → 云端同步**（容器访问 GitHub 不稳定，走 SFTP 直推）：

```bash
python train/patches/apply_patches.py        # 改动同步进本地 openpi-main（--export 反向收回）
.venv-lerobot/Scripts/python train/cloud/push_to_server.py   # train/ 目录打包推到服务器 code/ur5e_vla/
```

服务器端布局见 `/root/shared-nvme/code/README.md`；数据/权重/checkpoint 均不入 git（数据集在 HF 私有仓，
可用 `train/download_dataset.py` 拉回）。

## 结构

```
model/    ur5e_2f85.xml 组合场景（含三相机、tcp_goal 标记）
          robotiq_2f85/ universal_robots_ur5e/ —— MJCF + mesh（来自 MuJoCo Menagerie）
src/      ik.py DLS 逆解 | env.py 仿真环境 | expert.py 脚本专家 | recorder.py HDF5 记录
scripts/  collect_scripted.py 脚本采集 | collect_teleop.py 遥操作 | replay.py 数据检查
          probe_grasp.py 抓取高度标定 | check_projection.py 相机投影校验
          make_hover.py 悬停位姿烘焙 | test_model.py / test_render.py 环境自检
train/    openpi 全流程：转换/校验/评测客户端/数据下载 + patches 补丁机制 + cloud 云端运维
          results/ 训练曲线与评估结果（75% 里程碑）
data/     采集输出（不进 git）
```

UR5e 与 Robotiq 2F-85 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
