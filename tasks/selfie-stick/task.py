"""Readying a selfie stick for a group photo: "Get the selfie stick ready for a group photo: put my phone in
its holder and pull the stick all the way out."

A collapsed selfie stick lies on the bench with a phone beside it. Its three telescoping tubes are stiff,
so a hand pulling the head alone drags the whole stick along the bench: one hand holds the handle while
the other draws the head out to the stops. The spring clamp on the head is narrower than the phone:
one hand holds the spring jaw open by its thumb knob while the other lays the phone between the jaws, then
the jaw is let back onto the phone's edge.
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
import selfie as S  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "selfie-stick"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = S.OBJECT_NAMES

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in S.object_specs()]
    cfg = IsaacLabArenaEnvironment(name="yam_selfie_stick", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}


class Dynamics:
    """No added mechanics: the tubes slide on real damped prismatic joints and the spring jaw on a sprung
    one. It only records the measured poses and TCPs the scripted demo steers by and reports the stick's
    state."""

    def __init__(self, env):
        self.base = env.unwrapped

    def reset(self):
        SHARED.clear()

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        pose = {n: np.asarray(root_poses[n], float) for n in OBJECT_NAMES}
        SHARED.update(pose=pose, tcp={s: np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")})
        return {"extension_mm": round(1000 * S.extension(pose["handle"], pose["tube3"]), 1),
                "jaw_mm": round(1000 * S.jaw_open(pose["tube3"], pose["jaw"]), 1),
                "phone_off_mm": [round(1000 * float(v), 1) for v in S.phone_rel(pose["tube3"], pose["phone"])]}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------
# Two sequences on one clock, one per arm. A step's target is worked out from measured poses when the
# step starts, and each arm's own sag is trimmed out from its measured TCP -- a scripted demo may steer
# by simulator state.

REST = {"left": (0.18, 0.24, S.TOP + 0.20), "right": (0.18, -0.24, S.TOP + 0.20)}
PEG_DOWN = 0.013


def _grip_rot(jaw, app):
    c2 = np.asarray(app, float) / np.linalg.norm(app)
    j = np.asarray(jaw, float)
    c0 = j - np.dot(j, c2) * c2
    c0 /= np.linalg.norm(c0)
    return np.column_stack([c0, np.cross(c2, c0), c2])


def _rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


R_NECK = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))            # right: straight down, jaws across the head's neck
R_KNOB = _grip_rot((0.0, -1.0, 0.0), (math.sin(0.5), 0.0, -math.cos(0.5)))   # right: round the jaw's knob, leaned back toward the arms (clear of the left hand)
L_HANDLE = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))         # left: straight down, jaws across the handle
L_PHONE = _grip_rot((0.0, -1.0, 0.0), (0.0, 0.0, -1.0))          # left: straight down, jaws across the phone's width
PHONE_GRIP = 0.040                                               # the left hand takes the phone this far from its middle (its far end, clear of the clamp and the right hand)
PHONE_SQUEEZE = -0.0220                                          # fingers 64 mm apart: a firm hold on the 71.5 mm phone
PART = -0.040                                                    # fingers 80 mm apart
ROD_OPEN = -0.024                                                # fingers 48 mm apart, round a 28 mm rod
KNOB_OPEN = -0.016                                               # fingers 32 mm apart, round the 12 mm knob
OPEN_TO = 0.024                                                  # the spring jaw drawn this far open (the phone needs 13.5 mm)
SLACK = 0.0035                                                   # the phone goes in this far off the fixed jaw (clear of its lip)


def _pose(s, n):
    return s["pose"][n]


def _up(dz, dx=0.0, dy=0.0):
    return lambda s: tuple(np.asarray(s["cmd"]) + np.array([dx, dy, dz]))


def _above(f, dz=0.06):
    return lambda s: tuple(np.asarray(f(s) if callable(f) else f) + np.array([0.0, 0.0, dz]))


def _rod_grip(name, y_of):
    """TCP for fingertips straight down round the stick (handle or head neck), its axis between the pads."""
    def pos(s):
        p = np.asarray(_pose(s, name), float)
        y = y_of(s) if callable(y_of) else y_of
        return (float(p[0]), float(y), S.AXIS_Z + 0.006 + PEG_DOWN - 0.004)
    return pos


def _handle_y(s):
    return float(_pose(s, "handle")[1])


def _neck_y(s):
    t3 = np.asarray(_pose(s, "tube3"), float)
    return float(S.point(t3, (0.0, S.GRIP_NECK_Y - S.tube_centre(2)[1], 0.0))[1])


def _phone_grip(s):
    p = np.asarray(_pose(s, "phone"), float)
    return (float(p[0]) + PHONE_GRIP, float(p[1]), S.TOP + 0.009)


def _knob(open_by=None, dz=-0.008):
    """TCP round the jaw's knob, near its top; with ``open_by``, where it stands with the jaw that far open."""
    def pos(s):
        j = np.asarray(_pose(s, "jaw"), float)
        p = S.point(j, (S.TAB_DX, S.TAB_DY, S.TAB_TOP_REL + dz))
        if open_by is not None:
            t3 = np.asarray(_pose(s, "tube3"), float)
            p = p + S.R_(t3)[:, 1] * (open_by - S.jaw_open(t3, j))
        return tuple(p)
    return pos


def _move_phone(target_of, gain=1.0):
    """Carry the held phone so its centre lands on ``target_of`` (from the measured phone)."""
    def pos(s):
        here = np.asarray(_pose(s, "phone"), float)[:3]
        return tuple(np.asarray(s["cmd"], float) + gain * (np.asarray(target_of(s), float) - here))
    return pos


def _phone_rot(s):
    """L_PHONE turned about z to line the phone up with the stick as it lies now."""
    r3 = S.R_(_pose(s, "tube3"))
    yaw = math.atan2(r3[1, 0], r3[0, 0]) - S.YAW
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ L_PHONE


def _between_jaws(dz, dy=0.0):
    """The phone's centre midway between the open jaws, ``dz`` above where it sits on the bed (``dy`` short of it)."""
    def target(s):
        return S.point(np.asarray(_pose(s, "tube3"), float), S.SEAT_REL + np.array([0.0, SLACK + dy, dz]))
    return target


def _settle(label, target_of, first, n, finger=PHONE_SQUEEZE):
    return [(label + ("" if k == 0 else f" ({k + 1})"), first if k == 0 else 0.45, _move_phone(target_of), None, finger)
            for k in range(n)]


DRAW_HANDLE, DRAW_HEAD = 0.262, 0.050     # how far each hand draws its end out (together 12 mm past the stops)


def _draw_on(sign, share):
    """Still holding: draw this hand's end on by its share of what the measured extension still lacks."""
    def pos(s):
        left = S.FULL + 0.008 - S.extension(_pose(s, "handle"), _pose(s, "tube3"))
        return tuple(np.asarray(s["cmd"], float) + np.array([0.0, sign * share * max(left, 0.0) * 1.3, 0.0]))
    return pos


def _pad(steps, until):
    """Seconds left to wait so that ``steps`` end at ``until``."""
    return max(until - sum(st[1] for st in steps), 0.0)


def _steps():
    # both: draw the stick out, the left by its handle (+y), the right by its head (-y)
    draw = 10.0
    left = [
        ("look", 1.5, REST["left"], L_HANDLE, FINGER_OPEN),
        ("over the handle", 1.5, _above(_rod_grip("handle", _handle_y), 0.05), L_HANDLE, ROD_OPEN),
        ("down round the handle", 1.0, _rod_grip("handle", _handle_y), None, ROD_OPEN),
        ("grip the handle", 0.6, None, None, FINGER_CLOSED),
        ("draw the stick out", draw, _up(0.0, dy=DRAW_HANDLE), None, FINGER_CLOSED),
        *[(f"draw on to the stops ({k + 1})", 0.7, _draw_on(+1, 0.75), None, FINGER_CLOSED) for k in range(6)],
        ("steady", 0.4, None, None, FINGER_CLOSED),
        ("let go of the handle", 0.5, None, None, ROD_OPEN),
        ("up off the stick", 0.6, _up(0.05), None, ROD_OPEN),
        ("over the phone", 1.8, _above(_phone_grip, 0.05), L_PHONE, PART),
        ("down round the phone", 1.0, _phone_grip, None, PART),
        ("grip the phone", 0.6, None, None, PHONE_SQUEEZE),
        ("lift the phone", 0.8, _up(0.05), None, PHONE_SQUEEZE),
        ("carry it toward the clamp", 2.2, _move_phone(_between_jaws(0.05, -0.040)), _phone_rot, PHONE_SQUEEZE),
        ("down beside the clamp", 0.8, _move_phone(_between_jaws(0.019, -0.040)), _phone_rot, PHONE_SQUEEZE),
        ("down beside the clamp (2)", 0.45, _move_phone(_between_jaws(0.019, -0.040)), _phone_rot, PHONE_SQUEEZE),
    ]
    right = [
        ("look", 1.5, REST["right"], R_NECK, FINGER_OPEN),
        ("over the head", 1.5, _above(_rod_grip("tube3", _neck_y), 0.05), R_NECK, ROD_OPEN),
        ("down round the neck", 1.0, _rod_grip("tube3", _neck_y), None, ROD_OPEN),
        ("grip the neck", 0.6, None, None, FINGER_CLOSED),
        ("draw the stick out", draw, _up(0.0, dy=-DRAW_HEAD), None, FINGER_CLOSED),
        *[(f"draw on to the stops ({k + 1})", 0.7, _draw_on(-1, 0.25), None, FINGER_CLOSED) for k in range(6)],
        ("steady", 0.4, None, None, FINGER_CLOSED),
        ("let go of the neck", 0.5, None, None, ROD_OPEN),
        ("up off the stick", 0.6, _up(0.05), None, ROD_OPEN),
        ("over the jaw's knob", 1.5, _above(_knob(), 0.05), R_KNOB, KNOB_OPEN),
        ("down round the knob", 1.0, _knob(), None, KNOB_OPEN),
        ("grip the knob", 0.6, None, None, FINGER_CLOSED),
        ("pull the jaw open", 1.5, _knob(OPEN_TO), None, FINGER_CLOSED),
        ("pull the jaw open (2)", 0.5, _knob(OPEN_TO), None, FINGER_CLOSED),
    ]
    t_open = sum(st[1] for st in right)
    left.append(("wait for the jaw to open", _pad(left, t_open), None, None, PHONE_SQUEEZE))
    left += [
        ("over the fixed jaw, into the gap", 1.0, _move_phone(_between_jaws(0.019)), None, PHONE_SQUEEZE),
        ("into the gap (2)", 0.45, _move_phone(_between_jaws(0.019)), None, PHONE_SQUEEZE),
        ("lower it between the jaws", 0.8, _move_phone(_between_jaws(0.008)), None, PHONE_SQUEEZE),
        *_settle("hold it just over the bed", _between_jaws(0.004), 0.6, 4),
    ]
    t_held = sum(st[1] for st in left)
    hold = _pad(right, t_held)
    n = max(int(hold / 0.5), 1)
    right += [(f"keep the jaw open ({k + 1})", hold / n, _knob(OPEN_TO), None, FINGER_CLOSED) for k in range(n)]
    right += [
        ("close the jaw to the phone's edge", 1.2, _knob(S.JAW_SEATED + SLACK), None, FINGER_CLOSED),
    ]
    t_edge = sum(st[1] for st in right)
    left.append(("keep the phone there", _pad(left, t_edge), None, None, PHONE_SQUEEZE))
    left += [
        ("let go of the phone", 0.4, None, None, PART),
        ("up off the phone", 1.0, _up(0.05), None, PART),
        ("stand clear", 1.5, REST["left"], None, FINGER_OPEN),
        ("done", 6.0, REST["left"], None, FINGER_OPEN),
    ]
    right += [
        ("keep the jaw at the phone", 0.6, None, None, FINGER_CLOSED),
        ("ease the jaw onto the phone", 1.5, _knob(S.JAW_SEATED - 0.004), None, FINGER_CLOSED),
        ("hold the knob", 0.6, None, None, FINGER_CLOSED),
        ("let go of the knob", 0.5, None, None, KNOB_OPEN),
        ("up off the knob", 0.8, _up(0.05), None, KNOB_OPEN),
        ("stand clear", 1.5, REST["right"], None, FINGER_OPEN),
        ("done", 6.0, REST["right"], None, FINGER_OPEN),
    ]
    return {"left": left, "right": right}


def demo():
    steps = _steps()
    return {side: [Waypoint(sec, REST[side], FINGER_OPEN, label, DOWN) for label, sec, *_ in steps[side]]
            for side in ("left", "right")}


NO_SAG_FIX = ("grip", "let go", "draw", "steady", "ease")


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
