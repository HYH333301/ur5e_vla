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
python scripts/collect.py --episodes 50 --out data/ur5e_pickplace --seed 0

# 键盘遥操作采集（MuJoCo 窗口内按键；注意 viewer 保留键，见脚本内说明；
# 保存时自动压缩停顿空闲步，--no-trim 关闭）
python scripts/teleop_collect.py

# 数据检查：统计 + 8 帧三相机接触表图
python scripts/replay.py data/ur5e_pickplace/episode_0000.hdf5 --samples 8
```

## 数据格式（每回合一个 HDF5）

- `observations/`：`qpos qvel gripper tcp_pos tcp_quat cube_pos`（T+1 步）+
  `image_{front,side,wrist}_cam`（T+1 帧，320x240 JPEG，变长 uint8）
- `action`（T,7）：6 关节位置目标 + 夹爪指令 [0,1]，20 Hz
- `phase`（T+1）：专家阶段 id；遥操作回合为 -1
- attrs：`success` `instruction`（如 "pick up the red cube..."）`cube_rgba`
  `target_rgba` `source`（scripted/teleop）

## 结构

```
model/    ur5e_2f85.xml 组合场景（含三相机、tcp_goal 标记）
          robotiq_2f85/ universal_robots_ur5e/ —— MJCF + mesh（来自 MuJoCo Menagerie）
src/      ik.py DLS 逆解 | env.py 仿真环境 | expert.py 脚本专家 | recorder.py HDF5 记录
scripts/  collect.py 批量采集 | teleop_collect.py 遥操作 | replay.py 数据检查
          grasp_probe.py 抓取高度标定 | proj_qc.py 相机投影校验
          make_hover.py 悬停位姿烘焙 | test_model.py / render_test.py 环境自检
data/     采集输出（不进 git）
```

UR5e 与 Robotiq 2F-85 模型来自 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
