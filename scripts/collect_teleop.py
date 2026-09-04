"""Manual teleop collection: drive the UR5e with the keyboard, record episodes.

Keys are read from the MuJoCo viewer window (mouse orbits/zooms the camera).
Each keypress moves the IK target by one step (the viewer callback only sees
key presses, not holds), orientation stays tool-down like the scripted expert.
Episodes are recorded in the same HDF5 format as scripts/collect_scripted.py, with
phase=-1 and attrs source="teleop". On save, idle pauses (runs of identical
actions) are compressed automatically; disable with --no-trim.

NOTE: the viewer's visualization panel has single-key shortcuts bound to most
letters (A toggles wireframe, [ ] cycle cameras, digits 0-5 toggle geom
groups ...), so this mapping only uses keys verified not to touch render state
or data. The render state is also pinned every tick as a safety net:
  Up/Dn  target +/-y (away/toward you)  Lt/Rt target -/+x (left/right)
  W/S    target up/down (8/2 also work; their viewer shortcuts are
         neutralized by the render-state pin — at most a 1-frame flash)
  -/=    step size -/+ (10/25/50 mm)    R     re-sync target to current TCP
  Enter  finish episode -> save if success, then auto-start the next one
  K      force-save this episode        PgDn  discard episode and re-randomize
  close window / Ctrl+C                 quit

Example:
    python scripts/collect_teleop.py                       # interactive
    python scripts/collect_teleop.py --demo --episodes 3   # auto demo through
                                                          # the same loop
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

from env import Ur5eEnv, CAMS  # noqa: E402
from expert import (HOVER_Z, GRASP_Z, PLACE_Z, TOOL_DOWN,  # noqa: E402
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


class DemoInput:
    """Waypoint follower for --demo: drives target/grip like a scripted player."""

    SPEED = 0.25  # m/s

    def __init__(self, env: Ur5eEnv):
        c, t = env.cube_xy, env.tgt_xy
        self.wps = [
            (np.array([*c, HOVER_Z]), GRIP_OPEN, 6),
            (np.array([*c, GRASP_Z]), GRIP_OPEN, 12),
            (None, GRIP_CLOSE, 14),
            (np.array([*c, HOVER_Z]), GRIP_CLOSE, 8),
            (np.array([*t, HOVER_Z]), GRIP_CLOSE, 8),
            (np.array([*t, PLACE_Z]), GRIP_CLOSE, 8),
            (None, GRIP_OPEN, 14),
            (np.array([*t, HOVER_Z]), GRIP_OPEN, 4),
        ]
        self.i = self.hold = 0
        self.done = False

    def poll(self, st: TeleopState, dt: float):
        if self.done:
            return
        wp, grip, dwell = self.wps[self.i]
        st.grip = grip
        if wp is not None:
            d = wp - st.target
            dist = float(np.linalg.norm(d))
            step = self.SPEED * dt
            if dist > step:
                st.target += d * (step / dist)
                return
            st.target[:] = wp
        self.hold += 1
        if self.hold >= dwell:
            self.hold, self.i = 0, self.i + 1
            if self.i >= len(self.wps):
                self.done = True
                st.cmds.append("finish")


def run_episode(env, st, rng, make_input, viewer, out: Path, idx: int, args):
    """One episode from reset to save/discard. Returns True if saved."""
    obs = env.reset(rng)
    st.begin_episode()
    print(f"  task: {env.instruction}")
    env.model.site_pos[st.goal_site] = st.target
    rec = EpisodeRecorder(CAMS)
    inp = make_input(env)

    def add_obs(frames_alpha_hidden=True):
        if frames_alpha_hidden:
            env.model.site_rgba[st.goal_site, 3] = 0.0
        jpegs = {c: encode_jpeg(f) for c, f in env.render().items()}
        if frames_alpha_hidden:
            env.model.site_rgba[st.goal_site, 3] = 0.45
        rec.add_obs(obs, jpegs, PHASE)

    add_obs()
    realtime = viewer is not None and not args.fast
    tick, t_next = 0, time.perf_counter()
    while True:
        # ---- commands from keys (or demo end) ----
        quit_after = viewer is not None and not viewer.is_running()
        for c in st.drain_cmds() + (["finish"] if quit_after else []):
            if c == "reset":
                print("  [episode discarded]")
                return False
            if c in ("finish", "save"):
                success, dist = env.check_success()
                if not (success or c == "save"):
                    z = env.cube_pos()[2]
                    print(f"  [not success: dist={dist:.3f} m cube_z={z:.3f} "
                          f"(want dist<0.06, z~0.625, still) — discarded, new episode]")
                    return False
                if not args.no_trim:
                    t0, t1 = rec.trim_idle()
                    if t1 < t0:
                        print(f"  [trim] idle pauses removed: {t0} -> {t1} "
                              f"ticks (-{100 * (1 - t1 / t0):.0f}%)")
                path = out / f"episode_{idx:04d}.hdf5"
                rec.save(path, {"success": success, "instruction": env.instruction,
                                "cube_rgba": env.cube_rgba, "target_rgba": env.target_rgba,
                                "source": "teleop"})
                mb = path.stat().st_size / 1e6
                print(f"  [saved {path.name}: T={rec.n_actions} dist={dist:.3f} m "
                      f"{mb:.1f} MB  '{env.instruction}']")
                return True
        if quit_after:
            return False

        if isinstance(inp, DemoInput):
            inp.poll(st, 1.0 / 20.0)

        rec.add_action(st.act())
        obs = env.step(st.held)

        env.model.site_pos[st.goal_site] = st.target  # marker follows the goal
        mujoco.mj_forward(env.model, env.data)        # place it for the viewer
        add_obs()
        if viewer is not None:
            # pin the render state: stray reserved-key presses (letters,
            # digits 0-5, ...) can toggle wireframe or hide geom groups;
            # restoring the baseline here reverts them within one tick
            gg, sg, lab, frm, of, sf = args.render_state
            viewer.opt.geomgroup[:], viewer.opt.sitegroup[:] = gg, sg
            viewer.opt.label, viewer.opt.frame = lab, frm
            viewer.opt.flags[:], viewer.user_scn.flags[:] = of, sf
            viewer.sync()

        if (viewer is not None and tick % 20 == 0) or (viewer is None and tick % 100 == 0):
            err = np.linalg.norm(env.get_obs()["tcp_pos"] - st.target) * 1e3
            grip = "CLOSE" if st.grip > 0.5 else "OPEN "
            dist = np.linalg.norm(env.cube_pos()[:2] - env.tgt_xy) * 1e3
            print(f"  [t={tick * 0.05:5.1f}s T={rec.n_actions:4d} grip={grip} "
                  f"step={STEP_SIZES[st.step_i] * 100:g}cm err={err:3.0f}mm "
                  f"cube->tgt={dist:4.0f}mm]")
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
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ur5e_teleop")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--start", type=int, default=-1,
                    help="first episode index; -1 = continue after the highest "
                         "existing episode index in --out")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-ticks", type=int, default=1800, help="auto-finish after N ticks (90 s)")
    ap.add_argument("--demo", action="store_true",
                    help="auto-collect via waypoints instead of the keyboard")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep idle pause ticks (by default they are compressed at save)")
    ap.add_argument("--episodes", type=int, default=0,
                    help="stop after N saved episodes (0 = run until quit)")
    ap.add_argument("--fast", action="store_true", help="run as fast as possible (demo)")
    ap.add_argument("--no-viewer", action="store_true", help="headless (demo only)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = Ur5eEnv(width=args.width, height=args.height)
    st = TeleopState(env, rng)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.start < 0:  # never overwrite: continue after the highest existing index
        nums = [int(p.stem.rsplit("_", 1)[1]) for p in args.out.glob("episode_*.hdf5")
                if p.stem.rsplit("_", 1)[1].isdigit()]
        args.start = max(nums) + 1 if nums else 0
    print(f"新回合编号从 episode_{args.start:04d} 开始，输出目录 {args.out}")

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(env.model, env.data,
                                              key_callback=st.key if not args.demo else None)
        viewer.cam.lookat[:] = [-0.50, 0.0, 0.65]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, 90, -35
        args.render_state = (np.array(viewer.opt.geomgroup), np.array(viewer.opt.sitegroup),
                             int(viewer.opt.label), int(viewer.opt.frame),
                             np.array(viewer.opt.flags), np.array(viewer.user_scn.flags))
        viewer.sync()
        if not args.demo:
            print(HELP_CN)
            print(f"输出目录: {args.out}   采集 {args.width}x{args.height} @20Hz, "
                  f"格式与 scripts/collect_scripted.py 一致\n")

    make_input = DemoInput if args.demo else (lambda env: None)
    saved = 0
    try:
        while viewer is None or viewer.is_running():
            if args.episodes and saved >= args.episodes:
                break
            mode = "demo" if args.demo else "teleop"
            print(f"== episode {args.start + saved} ({mode})")
            if run_episode(env, st, rng, make_input, viewer, args.out,
                           args.start + saved, args):
                saved += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
    print(f"done: {saved} episodes saved to {args.out}")


if __name__ == "__main__":
    main()
