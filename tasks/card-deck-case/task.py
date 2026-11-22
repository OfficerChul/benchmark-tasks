"""Games night: "Put the cards back in their case and close it."

A squared deck of poker-size playing cards lies face down on the bench in front of its empty hard plastic
case, which lies on its back with its mouth toward the deck and its lid standing open. The case is light
and slides easily, so pushing the deck in would shove the case along: one hand holds the case's closed
end while the other slides the deck in and pushes it home, then the lid is tipped over and falls shut
over the mouth.
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
import cards as C  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "card-deck-case"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = C.OBJECT_NAMES
OVERVIEW_EYE = (0.58, -0.30, 1.00)
OVERVIEW_TARGET = (0.29, -0.02, 0.78)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in C.object_specs()]
    cfg = IsaacLabArenaEnvironment(name="yam_card_deck_case", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}


class Dynamics:
    """No added mechanics: every body is a rigid body on real joints. At reset it leans the lid back
    against its hinge stop; then it only records the measured poses and TCPs the scripted demo steers by."""

    def __init__(self, env):
        import torch
        self.torch = torch
        self.base = env.unwrapped

    def _place(self, name, pos, quat):
        t = self.torch
        obj = self.base.scene[name]
        dev = self.base.device
        pose = t.tensor([[*map(float, pos), *map(float, quat)]], dtype=t.float32, device=dev)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=t.zeros((1, 6), dtype=t.float32, device=dev))

    def reset(self):
        SHARED.clear()
        for name, (pos, quat) in C.start_poses().items():
            self._place(name, pos, quat)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        pose = {n: np.asarray(root_poses[n], float) for n in OBJECT_NAMES}
        SHARED.update(pose=pose, tcp={s: np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")})
        return {"lid_deg": round(C.lid_deg(pose["box"], pose["lid"]), 2)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------
# Two sequences on one clock, one per arm. A step's target is worked out from measured poses when the
# step starts, and each arm's own sag is trimmed out from its measured TCP -- a scripted demo may steer
# by simulator state.

REST = {"left": (0.18, 0.24, C.TOP + 0.20), "right": (0.18, -0.24, C.TOP + 0.20)}
PEG_DOWN = 0.013                       # closed fingertips: TCP down to the tip
PEG_JAW = 0.006                        # ... half thickness across the jaws (their last 13 mm)
PEG_W = 0.013                          # ... half width across their side faces
DECK_SQUEEZE = -0.0295                 # right fingers closed to 59 mm on the 63 mm deck
LET_GO = -0.038                        # ... opened to 76 mm to let it go


def _grip_rot(jaw, app):
    """TCP rotation whose jaws close along ``jaw`` and whose fingers point along ``app``."""
    c2 = np.asarray(app, float) / np.linalg.norm(app)
    j = np.asarray(jaw, float)
    c0 = j - np.dot(j, c2) * c2
    c0 /= np.linalg.norm(c0)
    return np.column_stack([c0, np.cross(c2, c0), c2])


def _lean(l, d):
    """Fingers pointing down, leaned ``l`` rad toward horizontal unit ``d``."""
    return np.array([0.0, 0.0, -1.0]) * math.cos(l) + np.asarray(d, float) * math.sin(l)


LJAW = (-1.0, 1.0, 0.0)                                     # the left hand's jaws, diagonal (its wrist's range)
L_BRACE = _grip_rot(LJAW, (0.0, 0.0, -1.0))                 # left fingertips straight down
L_PRESS = _grip_rot(LJAW, _lean(0.4, (0.0, -1.0, 0.0)))     # ... leaned toward the mouth, to tip the lid over
R_PICK = _grip_rot((1.0, 0.0, 0.0), (0.0, 0.0, -1.0))       # right fingers straight down, jaws across the deck
R_PUSH = _grip_rot((1.0, 0.0, 0.0), _lean(0.5, (0.0, 1.0, 0.0)))   # ... closed, leaned toward the case to push


def _box(s):
    return s["pose"]["box"]


def _case_pt(s, p):
    """A station point given where it lies with the case as it started, carried by the case."""
    return C.point(_box(s), np.subtract(p, C.BOX0))


def _case_dir(s, v):
    return C.R(_box(s)) @ np.asarray(v, float)


def _val(v, s):
    return v(s) if callable(v) else v


def _above(f, dz=0.06):
    return lambda s: tuple(np.asarray(f(s) if callable(f) else f) + np.array([0.0, 0.0, dz]))


def _up(dz, dx=0.0, dy=0.0):
    return lambda s: tuple(np.asarray(s["cmd"]) + np.array([dx, dy, dz]))


def _extent(rot, u):
    """How far the closed fingertips reach from the TCP along unit ``u``: they run from the TCP to PEG_DOWN
    along the approach, PEG_JAW thick across the jaws and PEG_W wide across their side faces."""
    r = np.asarray(rot, float)
    return max(PEG_DOWN * float(np.dot(r[:, 2], u)), 0.0) + abs(float(np.dot(r[:, 0], u))) * PEG_JAW \
        + abs(float(np.dot(r[:, 1], u))) * PEG_W


def _touch(point_fn, normal_fn, rot, gap=0.0015):
    """TCP that brings the closed fingertips to ``gap`` off a face through ``point_fn(s)`` whose outward
    normal is ``normal_fn(s)``."""
    def pos(s):
        p = np.asarray(point_fn(s), float)
        n = np.asarray(normal_fn(s), float)
        return tuple(p + n * (gap + _extent(rot, -n)))
    return pos, rot


# -- the left hand: the case's closed end, then the lid -------------------------------------------------

BRACE_Z = C.TOP + 0.012


def _end_brace(gap=0.0):
    """Left fingertips against the case's closed end, so the case cannot slide away from the pushed deck."""
    return _touch(lambda s: _case_pt(s, (C.BX, C.Y_END, BRACE_Z)), lambda s: _case_dir(s, (0.0, 1.0, 0.0)), L_BRACE, gap)


LID_R = 0.016                                               # the left fingertips tip the lid over this far up it
LID_X = 0.012


def _phi_now(s):
    """The lid's turn from straight up, toward the mouth (it hangs shut at 180)."""
    return 180.0 - C.lid_deg(_box(s), s["pose"]["lid"])


def _lid_back(phi, gap=0.0015):
    """Left fingertips on the back of the lid (the side toward the case's closed end), LID_R up it, with the
    lid turned ``phi`` degrees from straight up."""
    def point(s):
        t = math.radians(_val(phi, s))
        local = np.array([LID_X, -math.sin(t), math.cos(t)]) * np.array([1.0, LID_R, LID_R])
        return _case_pt(s, np.add(C.LID_PIVOT, local))

    def normal(s):
        t = math.radians(_val(phi, s))
        return _case_dir(s, (0.0, math.cos(t), math.sin(t)))
    return _touch(point, normal, L_PRESS, gap)


# -- the right hand: the deck --------------------------------------------------------------------------

GRASP_Y = -(C.DECK_L / 2 - 0.016)      # the deck is gripped this far toward its near (-y) end from its centre
DECK_Z = C.Z_IN + 0.0004 + (C.IN_D - 0.0004) / 2               # carried centred in the mouth's height


def _pick_point(s):
    c = C.point(s["pose"]["deck"], (0.0, GRASP_Y, 0.0))
    return np.array([c[0], c[1], C.TOP + 0.003 + PEG_DOWN])


def _deck_rel(s):
    """The deck's pose in the right hand's TCP frame, from the measured deck and TCP."""
    d = s["pose"]["deck"]
    t = np.asarray(s["tcp"]["right"], float)
    r = np.asarray(s["rot"], float)
    return r.T @ (d[:3] - t), r.T @ C.R(d)


def _carry(center_fn):
    """TCP pose that puts the held deck's centre at ``center_fn(s)``, lying square to the case; and that
    centre, which the arm then steers the measured deck toward (across, not up and down)."""
    def rot(s):
        rel_p, rel_r = _deck_rel(s)
        return C.R(_box(s)) @ rel_r.T

    def pos(s):
        rel_p, rel_r = _deck_rel(s)
        return tuple(np.asarray(center_fn(s), float) - rot(s) @ rel_p)
    return pos, rot, center_fn


def _deck_at(front_y):
    """The deck's centre with its far end at ``front_y`` along the case (station frame, as the case
    started), lying on the case's inside floor."""
    return lambda s: _case_pt(s, (C.BX, front_y - C.DECK_L / 2, DECK_Z))


FRONT_OUT = C.YM - 0.006               # lined up, the deck's far end stands this far short of the mouth
FRONT_IN = C.YM - 0.004 - PEG_W - GRASP_Y + C.DECK_L / 2      # slid in until the fingers come near the mouth
PUSH_HOME = C.YM + 0.0030              # its near end is pushed this far inside the mouth


def _deck_end(y=None, gap=0.0015):
    """Right fingertips against the deck's near end (mid-height), or against where that end is to be."""
    def point(s):
        if y is None:
            return C.point(s["pose"]["deck"], (0.0, -C.DECK_L / 2, 0.0))
        return _case_pt(s, (C.BX, y, C.Z_IN + 0.0004 + C.DECK_T / 2))
    return _touch(point, lambda s: _case_dir(s, (0.0, -1.0, 0.0)), R_PUSH, gap)


def _steps():
    t_braced = 1.5 + 1.5 + 1.0 + 0.6                                # the left fingertips are at the case's end
    right = [
        # label, seconds, position, rotation, finger[, deck steering]
        ("look", 1.5, REST["right"], R_PICK, FINGER_OPEN),
        ("over the deck", 1.5, _above(_pick_point, 0.06), R_PICK, FINGER_OPEN),
        ("down round the deck", 1.0, _pick_point, R_PICK, FINGER_OPEN),
        ("grip the deck", 0.6, None, None, DECK_SQUEEZE),
        ("lift the deck", 0.8, _up(0.03), None, DECK_SQUEEZE),
        ("line it up with the mouth", 1.5, *_carry(_deck_at(FRONT_OUT))[:2], DECK_SQUEEZE, _carry(_deck_at(FRONT_OUT))[2]),
        ("slide it in", 1.4, *_carry(_deck_at(FRONT_IN))[:2], DECK_SQUEEZE, _carry(_deck_at(FRONT_IN))[2]),
        ("let the deck go", 0.4, None, None, LET_GO),
        ("up off the deck", 0.5, _up(0.04), None, LET_GO),
        ("behind the deck", 1.0, _above(_deck_end(gap=0.012)[0], 0.03), R_PUSH, FINGER_CLOSED),
        ("down behind it", 0.6, *_deck_end(gap=0.012), FINGER_CLOSED),
        ("fingertips on its end", 0.6, *_deck_end(), FINGER_CLOSED),
        ("push it home", 1.6, *_deck_end(PUSH_HOME, 0.0), FINGER_CLOSED),
        ("hold it there", 0.4, None, None, FINGER_CLOSED),
        ("back off", 0.6, _up(0.03, dy=-0.03), None, FINGER_CLOSED),
        ("stand clear", 1.5, REST["right"], R_PICK, FINGER_OPEN),
        ("done", 10.0, REST["right"], None, FINGER_OPEN),
    ]
    t_pushed = sum(sec for _, sec, *_r in right[:14])
    left = [
        ("look", 1.5, REST["left"], L_BRACE, FINGER_CLOSED),
        ("over the case's closed end", 1.5, _above(_end_brace(0.010)[0], 0.04), L_BRACE, FINGER_CLOSED),
        ("down beside it", 1.0, *_end_brace(0.010), FINGER_CLOSED),
        ("fingertips to the end: hold the case", 0.6, *_end_brace(), FINGER_CLOSED),
        ("hold the case while the deck is pushed in", t_pushed - t_braced, None, None, FINGER_CLOSED),
        ("up off the case", 0.6, _up(0.05, dy=0.01), None, FINGER_CLOSED),
        ("over to the lid", 1.4, _above(_lid_back(_phi_now, 0.010)[0], 0.03), L_PRESS, FINGER_CLOSED),
        ("down behind the lid", 0.6, *_lid_back(_phi_now, 0.010), FINGER_CLOSED),
        ("fingertips on the lid", 0.5, *_lid_back(_phi_now), FINGER_CLOSED),
        ("tip the lid over", 1.0, *_lid_back(30.0), FINGER_CLOSED),
        ("let it fall shut", 1.2, _up(0.04), None, FINGER_CLOSED),
        ("stand clear", 1.5, REST["left"], L_BRACE, FINGER_OPEN),
        ("done", 6.0, REST["left"], None, FINGER_OPEN),
    ]
    return {"left": left, "right": right}


def demo():
    steps = _steps()
    return {side: [Waypoint(sec, REST[side], FINGER_OPEN, label, DOWN) for label, sec, *_ in steps[side]]
            for side in ("left", "right")}


NO_SAG_FIX = ("hold", "grip", "let", "slide", "fingertips", "push", "tip")


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
        self.fb_deck = np.zeros(3)           # steering of a carried deck by its measured place
        self.servo = None
        self.goal_deck = None
        self.sent = None

    def _begin(self, k, t):
        from yam_arena import ik
        label, sec, pos, rot, finger, *rest = self.steps[k]
        self.servo = rest[0] if rest else None
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
        self.goal_deck = np.asarray(self.servo(state), float) if self.servo is not None else None
        if label.startswith("up"):
            self.fb_deck = np.zeros(3)
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
        deck = SHARED.get("pose", {}).get("deck")
        if self.goal_deck is not None and deck is not None and u > 0.5:
            e = self.goal_deck - np.asarray(deck, float)[:3]
            self.fb_deck = np.clip(self.fb_deck + 0.08 * e, -0.03, 0.03)
        self.sent = pos
        q, pe, re = ik.solve(pos + self.fb + self.fb_deck, ik.matrix_to_quat_xyzw(rot), self.plan.q,
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
