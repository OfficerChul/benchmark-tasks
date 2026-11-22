"""Handing over binoculars: "Set these binoculars for me: 62 mm between the eyepieces, and the focus
wheel on the 5 m mark."

A pair of roof-prism binoculars lies on the bench on its barrels, opened out to 72 mm. Its centre hinge
is stiff, so the barrels are squeezed in toward each other from both sides, one hand on each, until the
eyepieces stand 62 mm apart; it stays where it is left. Then one hand holds the binoculars down while the
other rolls the stiff centre focus wheel round with a fingertip, stroke by stroke, to the 5 m mark.
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
import bino as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "binocular-setup"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = B.OBJECT_NAMES

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in B.object_specs()]
    cfg = IsaacLabArenaEnvironment(name="yam_binocular_setup", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}


class Dynamics:
    """No added mechanics: the barrels and the focus wheel are rigid bodies on real joints (a stiff
    friction hinge, a stiff wheel). It only records the measured poses and TCPs the scripted demo steers
    by and reports the measured settings."""

    def __init__(self, env):
        self.base = env.unwrapped

    def reset(self):
        SHARED.clear()

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        pose = {n: np.asarray(root_poses[n], float) for n in OBJECT_NAMES}
        SHARED.update(pose=pose, tcp={s: np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")})
        return {"ipd_mm": round(1000 * B.ipd(pose["barrel_l"], pose["barrel_r"]), 2),
                "focus_deg": round(B.focus_deg(pose["barrel_l"], pose["wheel"]), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------
# Two sequences on one clock, one per arm. A step's target is worked out from measured poses when the
# step starts, and each arm's own sag is trimmed out from its measured TCP -- a scripted demo may steer
# by simulator state.

REST = {"left": (0.18, 0.24, B.TOP + 0.20), "right": (0.18, -0.24, B.TOP + 0.20)}
PEG_DOWN = 0.013                       # closed fingertips: TCP down to the tip
PEG_JAW = 0.006                        # ... half thickness across the jaws (their last 13 mm)
PEG_W = 0.013                          # ... half width across their side faces


def _grip_rot(jaw, app):
    c2 = np.asarray(app, float) / np.linalg.norm(app)
    j = np.asarray(jaw, float)
    c0 = j - np.dot(j, c2) * c2
    c0 /= np.linalg.norm(c0)
    return np.column_stack([c0, np.cross(c2, c0), c2])


L_DOWN = _grip_rot((-1.0, 1.0, 0.0), (0.0, 0.0, -1.0))      # left fingertips straight down, jaws diagonal
R_DOWN = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))       # right fingertips straight down, jaws along x


def _pose(s, n):
    return s["pose"][n]


def _ipd_now(s):
    return B.ipd(_pose(s, "barrel_l"), _pose(s, "barrel_r"))


def _focus_now(s):
    return B.focus_deg(_pose(s, "barrel_l"), _pose(s, "wheel"))


GRIP_X = {"left": B.BX + B.L / 2 - 0.016,                # the left hand takes its barrel by the objective end,
          "right": B.BX - B.L / 2 - B.CUP_L / 2}          # the right hand its barrel by the eyepiece cup (clear of the bridge)
PART = -0.033                                              # fingers part-open (66 mm between the pads)
L_GRIP = _grip_rot((-0.5, 0.866, 0.0), (0.0, 0.0, -1.0))   # straight down, jaws across the barrel (turned 30 deg: the wrist's range)
R_GRIP = _grip_rot((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))
LIFT = 0.030
R_ROLL = _grip_rot((0.0, -1.0, 0.0), (math.sin(0.5), 0.0, -math.cos(0.5)))   # right fingertips leaned toward +x: the wrist stays back, clear of the left hand


def _grip_point(side):
    def pos(s):
        p = np.asarray(_pose(s, "barrel_l" if side == "left" else "barrel_r"), float)
        return (GRIP_X[side], p[1], p[2] + 0.004 + PEG_DOWN)
    return pos


def _down_to_rest(s):
    """Straight down until the held barrels are just over the bench (the arms sag about a centimetre
    under them), keeping the spacing they are held at."""
    lowest = min(_pose(s, "barrel_l")[2], _pose(s, "barrel_r")[2]) - B.R
    return tuple(np.asarray(s["cmd"], float) - np.array([0.0, 0.0, max(lowest - B.TOP - 0.010, 0.0)]))


AIR_TARGET = B.IPD_TARGET - 0.0008      # the barrels open a little as they are set down


def _trim(side, gain=0.8, target=B.IPD_TARGET):
    """Still holding: turn each barrel about the hinge by part of what the measured spacing is still off
    by (the wrists give a little, so the fold is closed in steps on the measured spacing)."""
    def frame(s):
        hinge = B.hinge_point(_pose(s, "barrel_l"))
        d_beta = gain * (B.beta_of(target) - B.beta_of(_ipd_now(s)))
        phi = math.radians(-d_beta if side == "left" else d_beta)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(phi), -math.sin(phi)], [0.0, math.sin(phi), math.cos(phi)]])
        return np.asarray(hinge, float), rx

    def pos(s):
        c, rx = frame(s)
        return tuple(c + rx @ (np.asarray(s["cmd"], float) - c))

    def rot(s):
        c, rx = frame(s)
        return rx @ np.asarray(s["rot"], float)
    return pos, rot


def _fold_about(side, lift, ipd_target, lower_to_bench=False):
    """Turn the held barrel about the hinge axis (raised ``lift``) so the barrels stand ``ipd_target``
    apart; optionally then carry the whole pair down onto the bench at that spacing."""
    def frame(s):
        hinge = B.hinge_point(_pose(s, "barrel_l"))
        c = np.array([hinge[0], hinge[1], SHARED.setdefault("hinge_z0", hinge[2]) + lift])
        d_beta = B.beta_of(ipd_target) - B.beta_of(_ipd_now(s))
        phi = math.radians(-d_beta if side == "left" else d_beta)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(phi), -math.sin(phi)], [0.0, math.sin(phi), math.cos(phi)]])
        return c, rx

    def pos(s):
        c, rx = frame(s)
        p = c + rx @ (np.asarray(s["cmd"], float) - c)
        if lower_to_bench:
            h_rest = B.TOP + B.R + B.A * math.sin(math.radians(B.beta_of(ipd_target))) + 0.004
            p = p + np.array([0.0, 0.0, h_rest - c[2]])
        return tuple(p)

    def rot(s):
        c, rx = frame(s)
        return rx @ np.asarray(s["rot"], float)
    return pos, rot


def _on_barrel_top(side, x_off, press=0.002):
    rot = L_DOWN if side > 0 else R_DOWN

    def pos(s):
        p = np.asarray(_pose(s, "barrel_l" if side > 0 else "barrel_r"), float)
        top = p[2] + B.R
        return (B.BX + x_off, p[1], top - press + PEG_DOWN)
    return pos


WHEEL_Y0 = -0.002                        # each stroke starts just past the wheel's top toward -y, so pressing on turns it forward


RIM_SIDE = 1.0                           # the side of the wheel's top the fingertip sweeps down (+1: +y)


def _wheel_tip(ang_deg, depth=0.0, out=0.0, up=0.0):
    """TCP for the right fingertip on the wheel's rim at ``ang_deg`` from its top toward -y (forward turning
    pushes the rim there down), ``depth`` pressed in, ``out`` further out along -y, ``up`` raised."""
    def pos(s):
        w = np.asarray(_pose(s, "wheel"), float)
        a = math.radians(ang_deg(s) if callable(ang_deg) else ang_deg)
        r = B.WHEEL_R + 0.0008 - depth
        tip = np.array([B.WHEEL_X, w[1] + RIM_SIDE * (r * math.sin(a) + out), w[2] + r * math.cos(a) + up])
        return tuple(tip - R_ROLL[:, 2] * PEG_DOWN)
    return pos


FOCUS_DONE = 9.0                         # degrees: close enough to the mark -- the hand stops stroking


def _while_off(f):
    """A stroke target while the wheel is still off its mark; once on it, the fingertip waits clear above."""
    park = _wheel_tip(0.0, up=0.035)
    return lambda s: f(s) if abs(_focus_now(s) - B.FOCUS_TARGET) > FOCUS_DONE else park(s)


def _sweep_to(s):
    """How far round the rim to sweep this stroke: in proportion to what is left to the mark (the rim
    slips a little), at most 40 degrees of rim."""
    return 20.0 + min(0.5 * max(_focus_now(s) - B.FOCUS_TARGET - 15.0, 0.0), 40.0)


def _up(dz, dx=0.0, dy=0.0):
    return lambda s: tuple(np.asarray(s["cmd"]) + np.array([dx, dy, dz]))


def _steps():
    strokes = []
    for k in range(5):
        strokes += [
            (f"back over the wheel ({k + 1})", 0.35, _while_off(_wheel_tip(20.0, up=0.010)), R_ROLL, FINGER_CLOSED),
            (f"fingertip onto the wheel ({k + 1})", 0.3, _while_off(_wheel_tip(20.0, depth=0.0010)), R_ROLL, FINGER_CLOSED),
            (f"roll it round ({k + 1})", 0.6, _while_off(_wheel_tip(_sweep_to, depth=0.0015)), R_ROLL, FINGER_CLOSED),
            (f"off the wheel sideways ({k + 1})", 0.25, _while_off(_up(0.0, dy=0.006 * RIM_SIDE)), None, FINGER_CLOSED),
        ]
    t_focus = 1.2 + sum(st[1] for st in strokes)
    both = lambda side: [
        ("look", 1.5, REST[side], L_GRIP if side == "left" else R_GRIP, FINGER_CLOSED),
        ("over the barrel's end", 1.5, _above(_grip_point(side), 0.06), L_GRIP if side == "left" else R_GRIP, PART),
        ("down round it", 1.0, _grip_point(side), None, PART),
        ("grip the barrel", 0.6, None, None, FINGER_CLOSED),
        ("lift the binoculars", 1.0, _up(LIFT), None, FINGER_CLOSED),
        ("fold the barrels in", 1.5, *_fold_about(side, LIFT, B.IPD_TARGET), FINGER_CLOSED),
        *[(f"fold on to the spacing ({k + 1})", 0.5, *_trim(side, gain=0.8, target=AIR_TARGET), FINGER_CLOSED) for k in range(6)],
        ("steady them", 0.4, None, None, FINGER_CLOSED),
        ("set them down", 1.0, _down_to_rest, None, FINGER_CLOSED),
        ("let go", 0.5, None, None, PART),
        ("up off the barrel", 0.6, _up(0.05), None, PART),
    ]
    left = both("left") + [
        ("over the left barrel", 0.8, _above(_on_barrel_top(1, 0.055), 0.03), L_GRIP, FINGER_CLOSED),
        ("hold the binoculars down", 0.8, _on_barrel_top(1, 0.055), L_GRIP, FINGER_CLOSED),
        ("keep holding while the wheel is turned", t_focus, None, None, FINGER_CLOSED),
        ("up", 0.8, _up(0.06), None, FINGER_CLOSED),
        ("stand clear", 1.5, REST["left"], None, FINGER_OPEN),
        ("done", 6.0, REST["left"], None, FINGER_OPEN),
    ]
    right = both("right") + [
        ("over the focus wheel", 0.8, _wheel_tip(20.0, up=0.015), R_ROLL, FINGER_CLOSED),
        ("wait for the hold", 1.2 + 0.8 - 0.8, None, None, FINGER_CLOSED),
        *strokes,
        ("up", 0.8, _up(0.06), None, FINGER_CLOSED),
        ("stand clear", 1.5, REST["right"], None, FINGER_OPEN),
        ("done", 6.0, REST["right"], None, FINGER_OPEN),
    ]
    return {"left": left, "right": right}


def _above(f, dz=0.06):
    return lambda s: tuple(np.asarray(f(s) if callable(f) else f) + np.array([0.0, 0.0, dz]))


def demo():
    steps = _steps()
    return {side: [Waypoint(sec, REST[side], FINGER_OPEN, label, DOWN) for label, sec, *_ in steps[side]]
            for side in ("left", "right")}


NO_SAG_FIX = ("hold", "keep", "roll", "fingertip", "lift off", "back over", "sideways", "grip", "fold", "trim", "set them", "let go")


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
