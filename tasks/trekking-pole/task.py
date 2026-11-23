"""Setting a trekking pole for someone: "Set my trekking pole to 120 cm: open the flip lock, pull the lower
section out to the 120 mark, and shut the lock again."

A two-section trekking pole lies on the bench at 110 cm. Its flip lock is shut. The lower section slides
stiffly, so a hand pulling it alone drags the whole pole along the bench: one hand holds the grip while
the other draws the lower section out to the 120 cm mark. The lock lever is flipped open by its thumb tab
first and shut again after.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import simkit  # noqa: E402
import trekking as T  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "trekking-pole"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = T.OBJECT_NAMES

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in T.object_specs()]
    cfg = IsaacLabArenaEnvironment(name="yam_trekking_pole", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}


class Dynamics:
    """No added mechanics: the lower section slides on a damped prismatic joint and the lever turns on a
    damped hinge. It only records the measured poses and TCPs the scripted demo steers by."""

    def __init__(self, env):
        self.base = env.unwrapped

    def reset(self):
        SHARED.clear()

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        pose = {n: np.asarray(root_poses[n], float) for n in OBJECT_NAMES}
        SHARED.update(pose=pose, tcp={s: np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")})
        return {"length_mm": round(1000 * T.length(pose["upper"], pose["lower"]), 1),
                "lever_deg": round(T.lever_deg(pose["upper"], pose["lever"]), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------
# Two sequences on one clock, one per arm. A step's target is worked out from measured poses when the
# step starts, and each arm's own sag is trimmed out from its measured TCP -- a scripted demo may steer
# by simulator state.

REST = {"left": (0.18, 0.24, T.TOP + 0.20), "right": (0.18, -0.24, T.TOP + 0.20)}
PEG_DOWN = 0.013


def _grip_rot(jaw, app):
    c2 = np.asarray(app, float) / np.linalg.norm(app)
    j = np.asarray(jaw, float)
    c0 = j - np.dot(j, c2) * c2
    c0 /= np.linalg.norm(c0)
    return np.column_stack([c0, np.cross(c2, c0), c2])


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


R_TAB = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))            # right: straight down round the lever's tab
R_ROD = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))            # right: straight down, jaws across the lower section
L_GRIP = _grip_rot((0.866, -0.5, 0.0), (0.0, 0.0, -1.0))         # left: straight down, jaws across the grip (turned 30 deg: the wrist's range)
TAB_OPEN = -0.020                                                # fingers 40 mm apart, round the 8 mm tab
ROD_OPEN = -0.024                                                # fingers 48 mm apart, round the lower section
GRIP_OPEN = -0.040                                               # fingers 80 mm apart, round the grip
GRIP_U = 0.070                                                   # the left hand holds the grip this far from its top
PULL_U = 0.780                                                   # the right hand takes the lower section this far from the grip top


def _pose(s, n):
    return s["pose"][n]


def _up(dz, dx=0.0, dy=0.0):
    return lambda s: tuple(np.asarray(s["cmd"]) + np.array([dx, dy, dz]))


def _above(f, dz=0.06):
    return lambda s: tuple(np.asarray(f(s) if callable(f) else f) + np.array([0.0, 0.0, dz]))


def _on_axis(u, dz):
    """TCP over the pole's axis u from the grip top, dz above the axis."""
    def pos(s):
        p = T.grip_point(_pose(s, "upper"), u)
        return (float(p[0]), float(p[1]), float(p[2]) + dz)
    return pos


def _on_lower(dz):
    """TCP over the lower section, PULL_U from the grip top as the pole is set now."""
    def pos(s):
        lo = np.asarray(_pose(s, "lower"), float)
        up = np.asarray(_pose(s, "upper"), float)
        u_now = PULL_U + (T.length(up, lo) - T.LENGTH0)
        p = T.grip_point(up, u_now)
        return (float(p[0]), float(p[1]), float(p[2]) + dz)
    return pos


def _tab(dz=-0.003):
    def pos(s):
        p = T.tab_top(_pose(s, "lever"))
        return (float(p[0]), float(p[1]), float(p[2]) + dz)
    return pos


def _swing_lever(to_deg, gain=1.0):
    """Still holding the tab: turn the hand about the lever's hinge so the lever goes to ``to_deg``."""
    def frame(s):
        up = _pose(s, "upper")
        h = T.point(up, np.subtract(T.HINGE, T.UPPER_POS))
        d = gain * (to_deg - T.lever_deg(up, _pose(s, "lever")))
        return np.asarray(h, float), _rz(d)

    def pos(s):
        h, r = frame(s)
        c = np.asarray(s["cmd"], float)
        return tuple(np.r_[(h + r @ (c - h))[:2], c[2]])

    def rot(s):
        h, r = frame(s)
        return r @ np.asarray(s["rot"], float)
    return pos, rot


def _draw_on(share=1.0, gain=1.2):
    """Still holding the lower section: draw it on by what the measured length still lacks."""
    def pos(s):
        lack = T.LENGTH_TARGET + 0.001 - T.length(_pose(s, "upper"), _pose(s, "lower"))
        up = np.asarray(_pose(s, "upper"), float)
        axis_out = -T.R_(up)[:, 1]                      # toward the tip
        return tuple(np.asarray(s["cmd"], float) + share * gain * lack * axis_out)
    return pos


def _pad(steps, until):
    return max(until - sum(st[1] for st in steps), 0.0)


def _steps():
    right = [
        ("look", 1.5, REST["right"], R_TAB, FINGER_OPEN),
        ("over the lever's tab", 1.5, _above(_tab(), 0.05), R_TAB, TAB_OPEN),
        ("down round the tab", 1.0, _tab(), None, TAB_OPEN),
        ("grip the tab", 0.6, None, None, FINGER_CLOSED),
        ("flip the lock open", 1.5, *_swing_lever(-85.0), FINGER_CLOSED),
        ("flip the lock open (2)", 0.6, *_swing_lever(-85.0), FINGER_CLOSED),
        ("flip the lock open (3)", 0.6, *_swing_lever(-85.0), FINGER_CLOSED),
        ("let go of the tab", 0.5, None, None, TAB_OPEN),
        ("up off the tab", 0.6, _up(0.05), None, TAB_OPEN),
        ("over the lower section", 1.4, _above(_on_lower(0.007), 0.05), R_ROD, ROD_OPEN),
        ("down round the lower section", 0.8, _on_lower(0.007), None, ROD_OPEN),
        ("grip the lower section", 0.6, None, None, FINGER_CLOSED),
    ]
    left = [
        ("look", 1.5, REST["left"], L_GRIP, FINGER_OPEN),
        ("over the grip", 1.5, _above(_on_axis(GRIP_U, 0.008), 0.05), L_GRIP, GRIP_OPEN),
        ("down round the grip", 1.0, _on_axis(GRIP_U, 0.008), None, GRIP_OPEN),
        ("grip the pole", 0.6, None, None, FINGER_CLOSED),
    ]
    t_ready = sum(st[1] for st in right)
    left.append(("hold the pole", _pad(left, t_ready), None, None, FINGER_CLOSED))
    right += [
        ("draw the lower section out", 5.0, _draw_on(1.0, 1.0), None, FINGER_CLOSED),
        *[(f"draw on to the mark ({k + 1})", 0.7, _draw_on(1.0, 1.0), None, FINGER_CLOSED) for k in range(5)],
        ("steady", 0.4, None, None, FINGER_CLOSED),
        ("let go of the lower section", 0.5, None, None, ROD_OPEN),
        ("up off the pole", 0.6, _up(0.05), None, ROD_OPEN),
    ]
    t_drawn = sum(st[1] for st in right)
    left.append(("hold the pole while it is drawn out", _pad(left, t_drawn), None, None, FINGER_CLOSED))
    r_open = _rz(-85.0) @ R_TAB
    right += [
        ("over the lever's tab", 1.4, _above(_tab(), 0.05), r_open, TAB_OPEN),
        ("down round the tab", 0.8, _tab(), None, TAB_OPEN),
        ("grip the tab", 0.6, None, None, FINGER_CLOSED),
        ("flip the lock shut", 1.5, *_swing_lever(0.0), FINGER_CLOSED),
        ("flip the lock shut (2)", 0.6, *_swing_lever(0.0), FINGER_CLOSED),
        ("let go of the tab", 0.5, None, None, TAB_OPEN),
        ("up off the tab", 0.6, _up(0.06), None, TAB_OPEN),
        ("stand clear", 1.5, REST["right"], None, FINGER_OPEN),
        ("done", 6.0, REST["right"], None, FINGER_OPEN),
    ]
    t_shut = sum(st[1] for st in right[:-3])
    left.append(("keep holding while the lock is shut", _pad(left, t_shut), None, None, FINGER_CLOSED))
    left += [
        ("let go of the pole", 0.5, None, None, GRIP_OPEN),
        ("up off the pole", 0.6, _up(0.06), None, GRIP_OPEN),
        ("stand clear", 1.5, REST["left"], None, FINGER_OPEN),
        ("done", 6.0, REST["left"], None, FINGER_OPEN),
    ]
    return {"left": left, "right": right}


def demo():
    steps = _steps()
    return {side: [Waypoint(sec, REST[side], FINGER_OPEN, label, DOWN) for label, sec, *_ in steps[side]]
            for side in ("left", "right")}


NO_SAG_FIX = ("grip", "let go", "draw", "steady", "hold", "flip")


def _smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


class ArmSequence:
    """Plays one arm's steps: each glides from where the last one ended to its own target."""

    def __init__(self, side, steps, plan):
        self.side, self.steps, self.plan = side, steps, plan
        self.k = -1
        self.t0 = 0.0
        self.nom = None
        self.start = self.goal = None
        self.fb = np.zeros(3)
        self.sent = None

    def _begin(self, k, t):
        from yam_arena import ik
        label, sec, pos, rot, finger = self.steps[k]
        if self.nom is None:
            p0, r0 = ik.tcp_pose(self.plan.q, self.plan.base_pos)
            self.nom = (np.asarray(p0, float), np.asarray(r0, float), float(FINGER_OPEN))
        state = dict(SHARED)
        state["cmd"] = self.nom[0]
        state["rot"] = self.nom[1]
        if "pose" not in state and (callable(pos) or callable(rot)):
            raise RuntimeError("demo needs measured poses before its first measured step")
        gp = np.asarray(pos(state) if callable(pos) else (pos if pos is not None else self.nom[0]), float)
        gr = np.asarray(rot(state) if callable(rot) else (rot if rot is not None else self.nom[1]), float)
        gf = float(finger(state) if callable(finger) else (finger if finger is not None else self.nom[2]))
        self.start, self.goal = self.nom, (gp, gr, gf)
        self.k, self.t0 = k, t

    def action(self, t):
        from yam_arena import ik
        acc, k = 0.0, len(self.steps) - 1
        for i, s in enumerate(self.steps):
            if t < acc + s[1]:
                k = i
                break
            acc += s[1]
        while self.k < k:
            self._begin(self.k + 1, acc if self.k + 1 == k else self.t0)
        label = self.steps[self.k][0]
        sec = self.steps[self.k][1]
        u = _smooth((t - self.t0) / sec) if sec > 0 else 1.0
        pos = self.start[0] + u * (self.goal[0] - self.start[0])
        rot = ik.slerp_matrix(self.start[1], self.goal[1], u)
        finger = self.start[2] + u * (self.goal[2] - self.start[2])
        self.nom = (pos, rot, finger)
        tcp = SHARED.get("tcp", {}).get(self.side)
        if self.sent is not None and tcp is not None and not any(w in label for w in NO_SAG_FIX):
            e = self.sent - tcp
            if float(np.linalg.norm(e)) < 0.03:
                self.fb = np.clip(self.fb + 0.06 * e, -0.02, 0.02)
        self.sent = pos
        q, pe, re = ik.solve(pos + self.fb, ik.matrix_to_quat_xyzw(rot), self.plan.q,
                             base_pos=self.plan.base_pos, iters=60)
        self.plan.q = q
        self.plan.worst_ik_mm = max(self.plan.worst_ik_mm, pe * 1000)
        return np.r_[q, float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))]


class DemoController:
    def __init__(self, plans):
        steps = _steps()
        self.arms = {side: ArmSequence(side, steps[side], plans[side]) for side in ("left", "right")}

    def action(self, t):
        return np.concatenate([self.arms[side].action(t) for side in ("left", "right")])


def make_demo_controller(env, plans):
    return DemoController(plans)
