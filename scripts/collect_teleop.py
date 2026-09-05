"""Data collection: expert-first teleop — the scripted expert drives each
episode, and when it fails you take over with the keyboard.

Scene: 3 objects (cube / sphere / cylinder) + 2 colored trays. Each episode
samples a task ("pick up the {color} {shape} and place it in the {color}
container"). The scripted expert executes it; if it fails mid-flight (grasp
miss, drop, plan abort) or ends unsuccessful, control switches to the
keyboard mid-episode. Finish it and Enter saves the episode
(source="rescued"); PgDn discards. Expert solo successes are auto-saved
(source="expert"). During playback, Enter forces an immediate takeover,
PgDn discards.

Annotation: every saved episode prompts for its task instruction (Enter keeps
the sampled one, typing replaces it). If you type your own instruction — e.g.
you rescued the episode by moving a different object — your label defines the
task and the episode is stored as a success.

Pure teleop (--pure-teleop): you drive from the start (source="teleop").

Keys are read from the MuJoCo viewer window (mouse orbits/zooms the camera).
Each keypress moves the IK target by one step (the viewer callback only sees
key presses, not holds), orientation stays tool-down like the scripted expert.
Episodes are one HDF5 per episode; the expert part carries real phase ids,
human segments carry -1. Idle pauses (runs of identical actions) are
compressed on save in pure-teleop mode only; disable with --no-trim.

NOTE: the viewer's visualization panel has single-key shortcuts bound to most
letters (A toggles wireframe, [ ] cycle cameras, digits 0-5 toggle geom
groups ...), so this mapping only uses keys verified not to touch render state
or data. The render state is also pinned every tick as a safety net:
  Up/Dn  target +/-y (away/toward you)  Lt/Rt target -/+x (left/right)
  W/S    target up/down (8/2 also work; their viewer shortcuts are
         neutralized by the render-state pin — at most a 1-frame flash)
  -/=    step size -/+ (10/25/50 mm)    R     re-sync target to current TCP
  Enter  finish episode -> save if success (annotate the instruction), next one
  K      force-save this episode        PgDn  discard episode and re-randomize
  close window / Ctrl+C                 quit

Example:
    python scripts/collect_teleop.py                       # expert-first (default)
    python scripts/collect_teleop.py --pure-teleop         # you drive everything
    python scripts/collect_teleop.py --episodes 20         # stop after 20 saved
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from env import Ur5eEnv, CAMS, EpisodeDone, OBJ, OBJ_NAMES  # noqa: E402
from expert import (PickPlaceExpert, ExpertFailure, PH, TOOL_DOWN,  # noqa: E402
                    GRIP_OPEN, GRIP_CLOSE)
from recorder import EpisodeRecorder, encode_jpeg  # noqa: E402

# GLFW key codes (what the viewer's key_callback receives). Restricted to keys
# verified not to collide with the viewer's built-in visualization shortcuts
# (digits 0-5 toggle geom groups, most letters toggle render flags; digit 2's
# geom-group toggle is neutralized by the per-tick render-state pin).
KEY = dict(G=71, R=82, K=75, ENTER=257,
           UP=265, DOWN=264, LEFT=263, RIGHT=262,
           MINUS=45, EQUAL=61, PGDN=267)
Z_UP = (87, 56, 328)      # 'W' / '8' top row / keypad
Z_DN = (83, 50, 322)      # 'S' / '2' top row / keypad
STEP_SIZES = (0.01, 0.025, 0.05)          # m per keypress (-/= cycles)

WS_LO = np.array([-0.85, -0.35, 0.615])   # workspace clamp for the IK target
WS_HI = np.array([-0.20, 0.35, 0.95])
IK_MAX_POS = 6e-3                          # beyond this the target is unreachable
PHASE = -1                                 # teleop episodes carry no phase labels

HELP_CN = """
== 遥操作采集（在 MuJoCo 窗口里按键；鼠标拖动=转视角，滚轮=缩放） ==
  ↑/↓  目标 ±y（远离/靠近你）   ←/→  目标 -/+x（左/右）
  W/S  目标 升/降（8/2 也可用）  G    夹爪 开/合 切换
  -/=  步长 -/+（1/2.5/5 cm）   R    目标重置到当前指尖
  Enter 完成回合→判定并保存→自动开下一回合
  K    强制保存本回合           PgDn 放弃本回合并重置
  绿色小球 = IK 目标位置（不会进采集图像）
  保存前会在终端询问任务指令（Enter=默认「拿X放到Y容器」，可自行改写）
  注意：数字 0-5 和大多数字母是 viewer 保留键（切渲染模式/隐藏网格），
  误按画面最多闪一帧会自动恢复 —— 请只用上面列出的键
========================================================================"""


class TeleopState:
    """Shared teleop state: IK target, gripper cmd, held action, command queue."""

    def __init__(self, env: Ur5eEnv, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.goal_site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tcp_goal")
        self.cmds: list[str] = []      # appended from the viewer thread
        self.ik_warn = 0.0
        self._begin()

    def _begin(self):
        obs = self.env.get_obs()
        self.target = obs["tcp_pos"].copy()
        self.last_q = obs["qpos"].copy()
        self.grip = GRIP_OPEN
        self.step_i = 1
        self.held = np.append(obs["qpos"], GRIP_OPEN).astype(np.float32)

    def begin_episode(self):
        self._begin()
        self.cmds.clear()

    def begin_takeover(self):
        """Re-sync to the CURRENT state for human takeover mid-episode.

        Unlike begin_episode, keeps the gripper as-is (the expert may have the
        cube in hand) instead of defaulting to open.
        """
        obs = self.env.get_obs()
        self.target = obs["tcp_pos"].copy()
        self.last_q = obs["qpos"].copy()
        self.grip = GRIP_CLOSE if obs["gripper"] > 0.5 else GRIP_OPEN
        self.held = np.append(obs["qpos"], self.grip).astype(np.float32)
        self.cmds.clear()

    # ---- called from the viewer thread on key press ----
    def key(self, k: int):
        if k == KEY["UP"]:
            self.move(1, +1)
        elif k == KEY["DOWN"]:
            self.move(1, -1)
        elif k == KEY["RIGHT"]:
            self.move(0, +1)
        elif k == KEY["LEFT"]:
            self.move(0, -1)
        elif k in Z_UP:
            self.move(2, +1)
        elif k in Z_DN:
            self.move(2, -1)
        elif k == KEY["G"]:
            self.grip = GRIP_CLOSE if self.grip < 0.5 else GRIP_OPEN
        elif k == KEY["MINUS"] and self.step_i > 0:
            self.step_i -= 1
        elif k == KEY["EQUAL"] and self.step_i < len(STEP_SIZES) - 1:
            self.step_i += 1
        elif k == KEY["R"]:
            self.target[:] = self.env.get_obs()["tcp_pos"]
        elif k == KEY["PGDN"]:
            self.cmds.append("reset")
        elif k == KEY["ENTER"]:
            self.cmds.append("finish")
        elif k == KEY["K"]:
            self.cmds.append("save")

    def move(self, axis: int, sign: float):
        self.target[axis] = np.clip(self.target[axis] + sign * STEP_SIZES[self.step_i],
                                    WS_LO[axis], WS_HI[axis])

    def drain_cmds(self) -> list[str]:
        out, self.cmds = self.cmds, []
        return out

    # ---- called once per control tick by the main loop ----
    def act(self) -> np.ndarray:
        """Solve IK for the target; on failure hold the previous action."""
        q, pe, re, _ = self.env.ik.solve_with_restarts(
            self.last_q, self.rng, target_pos=self.target, target_z_dir=TOOL_DOWN,
            q_ref=self.last_q, max_iters=150, pos_tol=2e-3, rot_tol=1.5e-2)
        if pe > IK_MAX_POS or re > 5e-2:
            now = time.perf_counter()
            if now - self.ik_warn > 2.0:
                print(f"  [IK] target unreachable (err={pe * 1e3:.0f} mm), holding")
                self.ik_warn = now
            return self.held
        self.last_q = q
        self.held = np.append(q, self.grip).astype(np.float32)
        return self.held


def check_phase_exit(env: Ur5eEnv, phase: int) -> None:
    """Sanity checks when the expert finishes a phase; abort bad episodes."""
    if phase == PH["close"]:
        q_drv = env.data.qpos[env.grip_qadr]
        if q_drv > 0.6:  # fully closed = nothing between the fingers
            raise EpisodeDone(f"grasp failed: driver closed fully ({q_drv:.2f})")
    elif phase in (PH["lift"], PH["carry"]):
        # transport only: at 'place' the object is SUPPOSED to end up low again
        rest = OBJ[env.task_obj]["rest"]
        if env.obj_pos(env.task_obj)[2] < rest + 0.015:
            raise EpisodeDone(f"{env.task_obj} was dropped during transport")


def annotate(default: str) -> tuple[str, bool]:
    """Confirm or rewrite the episode's task instruction before saving.
    Returns (instruction, custom_typed)."""
    try:
        s = input(f"标注任务指令 (Enter=用默认「{default}」): ").strip()
    except (EOFError, KeyboardInterrupt):
        return default, False
    return (s or default, bool(s))


def episode_attrs(env: Ur5eEnv, success: bool, instruction: str, source: str) -> dict:
    return {
        "success": bool(success),
        "instruction": instruction,
        "source": source,
        "task_obj": env.task_obj,
        "task_cont": env.task_cont,
        "objects": np.asarray(OBJ_NAMES),          # rows of objects_rgba
        "objects_rgba": np.stack([env.obj_rgba[n] for n in OBJ_NAMES]),
        "containers_rgba": np.stack(env.cont_rgba),
    }


PH_NAME = {v: k for k, v in PH.items()}

TAKEOVER_CN = """
== 专家失败，请接管 ==
  任务: '{task}'
  按键同遥操作（↑↓←→ / W S / G / R / -+= / Enter / K / PgDn）
  Enter = 判定成功并保存（source='rescued'）   PgDn = 放弃本回合
  保存前可改写任务指令（如果你完成的是别的任务，请直接输入）
================================"""


def _pin_render(viewer, args):
    """Restore the pinned render state (stray reserved-key presses get reverted)."""
    gg, sg, lab, frm, of, sf = args.render_state
    viewer.opt.geomgroup[:], viewer.opt.sitegroup[:] = gg, sg
    viewer.opt.label, viewer.opt.frame = lab, frm
    viewer.opt.flags[:], viewer.user_scn.flags[:] = of, sf


def expert_playback(env, st, rec, viewer, args, expert, obs, tick):
    """Drive one episode with the scripted expert inside the realtime loop.

    Returns (outcome, tick) where outcome is 'success' | 'failed' | 'discard'
    | 'takeover'. Records obs/actions exactly like the teleop loop (phase ids
    from the expert plan).
    """
    try:
        actions, phases = expert.plan(env.task_obj_xy, env.task_cont_xy,
                                      obs["qpos"], env.task_obj)
    except ExpertFailure as e:
        print(f"  [expert] 规划失败: {e}")
        return "failed", tick
    realtime = True  # pace the 20 Hz loop so the human can watch
    t_next = time.perf_counter()
    prev_phase = phases[0]
    try:
        for t, action in enumerate(actions):
            for c in st.drain_cmds():  # Enter=take over now, PgDn=discard
                if c == "reset":
                    print("  [episode discarded]")
                    return "discard", tick
                if c == "finish":
                    print("  [expert] 手动接管")
                    return "takeover", tick
            if phases[t] != prev_phase:
                check_phase_exit(env, prev_phase)
                prev_phase = phases[t]
            rec.add_action(action)
            obs = env.step(action)
            rec.add_obs(obs, {c: encode_jpeg(f) for c, f in env.render().items()}, phases[t])
            env.model.site_pos[st.goal_site] = obs["tcp_pos"]  # marker rides the TCP
            mujoco.mj_forward(env.model, env.data)
            _pin_render(viewer, args)
            viewer.sync()
            if tick % 20 == 0:
                dist = np.linalg.norm(env.task_obj_xy - env.task_cont_xy) * 1e3
                print(f"  [expert t={tick * 0.05:5.1f}s phase={PH_NAME.get(phases[t], '?'):<6} "
                      f"{env.task_obj}->tgt={dist:4.0f}mm]")
            tick += 1
            if realtime:
                t_next += 0.05
                slack = t_next - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    t_next = time.perf_counter()
    except EpisodeDone as e:
        print(f"  [expert] 失败: {e}")
        return "failed", tick
    success, dist = env.check_success()
    tag = "成功" if success else "完成但未达标"
    print(f"  [expert] {tag} (dist={dist:.3f} m)")
    return ("success" if success else "failed"), tick


def run_episode(env, st, rng, viewer, out: Path, idx: int, args, expert=None):
    """One episode from reset to save/discard. Returns True if saved."""
    obs = env.reset(rng)
    st.begin_episode()
    print(f"  task: {env.instruction}")
    env.model.site_pos[st.goal_site] = st.target
    rec = EpisodeRecorder(CAMS)

    def add_obs(frames_alpha_hidden=True):
        if frames_alpha_hidden:
            env.model.site_rgba[st.goal_site, 3] = 0.0
        jpegs = {c: encode_jpeg(f) for c, f in env.render().items()}
        if frames_alpha_hidden:
            env.model.site_rgba[st.goal_site, 3] = 0.45
        rec.add_obs(obs, jpegs, PHASE)

    add_obs()
    tick = 0
    source = "teleop"
    if expert is not None:
        outcome, tick = expert_playback(env, st, rec, viewer, args, expert, obs, tick)
        if outcome == "discard":
            return False
        if outcome == "success":
            instr, _ = annotate(env.instruction)
            path = out / f"episode_{idx:04d}.hdf5"
            rec.save(path, episode_attrs(env, True, instr, "expert"))
            mb = path.stat().st_size / 1e6
            print(f"  [saved {path.name}: T={rec.n_actions} expert-only {mb:.1f} MB  "
                  f"'{instr}']")
            return True
        # 'failed' or 'takeover': the human continues from the current state
        st.begin_takeover()
        env.model.site_pos[st.goal_site] = st.target
        source = "rescued"
        print(TAKEOVER_CN.format(task=env.instruction))
    realtime = True  # pace the 20 Hz loop; collection is interactive by design
    t_next = time.perf_counter()
    while True:
        # ---- commands from keys ----
        quit_after = not viewer.is_running()
        for c in st.drain_cmds() + (["finish"] if quit_after else []):
            if c == "reset":
                print("  [episode discarded]")
                return False
            if c in ("finish", "save"):
                success, dist = env.check_success()
                if not (success or c == "save"):
                    z = env.obj_pos(env.task_obj)[2]
                    print(f"  [not success: dist={dist:.3f} m {env.task_obj}_z={z:.3f} "
                          f"(want dist<0.045, resting in tray, still) — discarded, new episode]")
                    return False
                # the annotator's instruction defines the task: a typed label
                # (e.g. you rescued by moving a different object) overrides the
                # sampled one and the episode counts as a success
                instr, custom = annotate(env.instruction)
                if custom:
                    success = True
                if not args.no_trim and expert is None:  # expert dwell holds are meaningful
                    t0, t1 = rec.trim_idle()
                    if t1 < t0:
                        print(f"  [trim] idle pauses removed: {t0} -> {t1} "
                              f"ticks (-{100 * (1 - t1 / t0):.0f}%)")
                path = out / f"episode_{idx:04d}.hdf5"
                rec.save(path, episode_attrs(env, success, instr, source))
                mb = path.stat().st_size / 1e6
                print(f"  [saved {path.name}: T={rec.n_actions} dist={dist:.3f} m "
                      f"source={source} {mb:.1f} MB  '{instr}']")
                return True
        if quit_after:
            return False

        rec.add_action(st.act())
        obs = env.step(st.held)

        env.model.site_pos[st.goal_site] = st.target  # marker follows the goal
        mujoco.mj_forward(env.model, env.data)        # place it for the viewer
        add_obs()
        # pin the render state: stray reserved-key presses (letters,
        # digits 0-5, ...) can toggle wireframe or hide geom groups;
        # restoring the baseline here reverts them within one tick
        _pin_render(viewer, args)
        viewer.sync()

        if tick % 20 == 0:
            err = np.linalg.norm(env.get_obs()["tcp_pos"] - st.target) * 1e3
            grip = "CLOSE" if st.grip > 0.5 else "OPEN "
            dist = np.linalg.norm(env.task_obj_xy - env.task_cont_xy) * 1e3
            print(f"  [t={tick * 0.05:5.1f}s T={rec.n_actions:4d} grip={grip} "
                  f"step={STEP_SIZES[st.step_i] * 100:g}cm err={err:3.0f}mm "
                  f"{env.task_obj}->tgt={dist:4.0f}mm]")
        tick += 1
        if tick > args.max_ticks:
            print("  [max duration reached, auto finishing]")
            st.cmds.append("finish")

        if realtime:
            t_next += 0.05
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                t_next = time.perf_counter()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ur5e_pickplace")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--start", type=int, default=-1,
                    help="first episode index; -1 = continue after the highest "
                         "existing episode index in --out")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-ticks", type=int, default=1800, help="auto-finish after N ticks (90 s)")
    ap.add_argument("--pure-teleop", action="store_true",
                    help="you drive from the start (default: expert-first, "
                         "you only rescue failures)")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep idle pause ticks (by default they are compressed at save)")
    ap.add_argument("--episodes", type=int, default=0,
                    help="stop after N saved episodes (0 = run until quit)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = Ur5eEnv(width=args.width, height=args.height)
    st = TeleopState(env, rng)
    expert = None if args.pure_teleop else PickPlaceExpert(env.ik, rng)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.start < 0:  # never overwrite: continue after the highest existing index
        nums = [int(p.stem.rsplit("_", 1)[1]) for p in args.out.glob("episode_*.hdf5")
                if p.stem.rsplit("_", 1)[1].isdigit()]
        args.start = max(nums) + 1 if nums else 0
    print(f"新回合编号从 episode_{args.start:04d} 开始，输出目录 {args.out}")

    viewer = mujoco.viewer.launch_passive(env.model, env.data, key_callback=st.key)
    viewer.cam.lookat[:] = [-0.50, 0.0, 0.65]
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 90, -35
    args.render_state = (np.array(viewer.opt.geomgroup), np.array(viewer.opt.sitegroup),
                         int(viewer.opt.label), int(viewer.opt.frame),
                         np.array(viewer.opt.flags), np.array(viewer.user_scn.flags))
    viewer.sync()
    print(HELP_CN)
    print(f"输出目录: {args.out}   采集 {args.width}x{args.height} @20Hz, "
          f"每回合一个 HDF5\n")
    if not args.pure_teleop:
        print("专家先行：每回合先由脚本专家执行（期间 Enter=立即接管, PgDn=放弃）；")
        print("专家失败自动切给你（source='rescued'），成功则自动保存（source='expert'）。")
    print("场景：方块/球/圆柱 + 两个彩色托盘；保存前终端确认任务指令（Enter=默认）。\n")

    saved = 0
    try:
        while viewer.is_running():
            if args.episodes and saved >= args.episodes:
                break
            mode = "teleop" if args.pure_teleop else "expert+teleop"
            print(f"== episode {args.start + saved} ({mode})")
            if run_episode(env, st, rng, viewer, args.out,
                           args.start + saved, args, expert=expert):
                saved += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        viewer.close()
        env.close()
    print(f"done: {saved} episodes saved to {args.out}")


if __name__ == "__main__":
    main()
