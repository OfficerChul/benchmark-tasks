"""Tidying books: "The dust jacket's front flap has come out of this book. Tuck it back inside the
front cover and close the book."

A cloth-bound hardback stands upright on its tail against a bookend, its front cover standing ajar, its
paper dust jacket on but for the front flap, which has come out from inside the cover and sticks out
beyond the fore-edge, its crease flattened. The right hand holds the cover by its top edge; the left
hand's fingertips push the flap round the fore-edge to the inside of the cover and keep it there, for it
springs back out as soon as it is let go, while the cover is swung in over it; the right hand then
presses the cover in from outside, the flap is let go only once the closing cover traps it against the
page edges, and the book is pressed shut.
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
import book as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "book-jacket-flap"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = ("book", "board", "flap")
OVERVIEW_EYE = (0.52, 0.44, 1.17)
OVERVIEW_TARGET = (0.22, 0.00, 0.86)

DOWN = simkit.patch_demo()                           # jaws closing along y


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = [B.bookend_object(), B.book_object(), B.board_object(), B.flap_object()]
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_book_jacket_flap", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state the scripted demo steers by


class Dynamics:
    """No added mechanics: every body here is a real rigid body on real joints. At reset it stands the
    front cover ajar (START_OPEN, the flap carried round with it), then it only records the measured
    poses and TCPs the scripted demo steers by and reports the measured angles."""

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
        for spec in (B.board_object(), B.flap_object()):
            self._place(spec.name, *B.opened(spec.pos))

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        pose = {n: np.asarray(root_poses[n], float) for n in OBJECT_NAMES}
        SHARED.update(pose=pose, tcp={s: np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")})
        return {"cover_deg": round(B.cover_deg(pose["book"], pose["board"]), 2),
                "fold_deg": round(B.fold_deg(pose["board"], pose["flap"]), 2)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------
# Two sequences on one clock, one for each arm. A step's target is worked out from measured poses when
# the step starts, and each arm's own sag is trimmed out from its measured TCP -- a scripted demo may
# steer by simulator state.

REST = {"left": (0.18, 0.24, B.TOP + 0.20), "right": (0.18, -0.24, B.TOP + 0.20)}
PART = -0.010                                         # fingers part-open (20 mm between the pads)
ZT = B.TOP + B.BOOK_H - 0.006                         # TCP height for a pinch on the top edges
ZP = B.TOP + B.BOOK_H + 0.003                         # TCP height for a fingertip push: only the tips below the top edges
PEG_DOWN = 0.013                                      # closed fingertips: from the TCP down to the tip
PEG_JAW = 0.006                                       # ... centre to their jaw-side faces (their last 13 mm)
PEG_W = 0.013                                         # ... centre to their side faces
HOVER = 0.06
R_COVER = 0.050                                       # the cover hand pinches this far out from the spine
S_PUSH = 0.035                                        # the flap is pushed this far out from its crease
FOLD = 175.0                                          # how far the flap is turned in and held (deg)
HANDOVER = 20.0                                       # the right hand lets go of the cover here and presses it from outside
R_PRESS = 0.075                                       # ... this far out from the spine
LET_GO = 11.0                                         # cover angle at which the flap is let go (deg)


def _rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def _grip_rot(jaw, lean_dir, lean):
    """TCP rotation whose jaws close along ``jaw`` (horizontal), fingers pointing down, leaned by
    ``lean`` rad toward ``lean_dir`` (taken square to the jaws)."""
    j = np.asarray(jaw, float) / np.linalg.norm(jaw)
    d = np.asarray(lean_dir, float) - np.dot(lean_dir, j) * j
    d /= np.linalg.norm(d)
    c2 = np.array([0.0, 0.0, -1.0]) * math.cos(lean) + d * math.sin(lean)
    c0 = j - np.dot(j, c2) * c2
    c0 /= np.linalg.norm(c0)
    return np.column_stack([c0, np.cross(c2, c0), c2])


L_PEG = _grip_rot((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), 0.3)     # left hand, fingers closed, used as a fingertip
R_PEG = _grip_rot((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.5)     # right hand, fingers closed


def _hinge(s):
    """The cover's hinge line (x, y) as the book now stands."""
    book = s["pose"]["book"]
    b0 = B.book_object().pos
    return B.point(book, (B.PIVOT[0] - b0[0], B.PIVOT[1] - b0[1], 0.0))[:2]


def _book_yaw(s):
    return B.yaw_deg(s["pose"]["book"]) + B.FRAME_YAW


def _cover_now(s):
    return B.cover_deg(s["pose"]["book"], s["pose"]["board"])


def _frame(s, theta):
    """Unit vectors (station frame) along the cover from spine to fore-edge, and square to it toward the
    pages, with the cover open ``theta`` degrees."""
    t = math.radians(theta + _book_yaw(s))
    return np.array([-math.cos(t), -math.sin(t)]), np.array([-math.sin(t), math.cos(t)])


def _cover_pinch(theta):
    """Where and how the cover hand holds the cover's top edge (board and jacket together), with the
    cover open ``theta`` degrees."""
    def pos(s):
        along, inward = _frame(s, theta)
        c = _hinge(s) + along * R_COVER - inward * 0.0006           # centre on board and jacket panel
        return (c[0], c[1], ZT)

    def rot(s):
        along, inward = _frame(s, theta)
        return _grip_rot((inward[0], inward[1], 0.0), (0.0, 1.0, 0.0), 0.7)
    return pos, rot


def _above(f, dz=HOVER):
    return lambda s: tuple(np.asarray(f(s) if callable(f) else f) + np.array([0.0, 0.0, dz]))


def _swing(deg, about, turn=True):
    """Turn the hand's commanded pose ``deg`` degrees about a vertical line through ``about(s)``
    (``deg`` may be a function of the state); ``turn`` False keeps the hand's orientation."""
    def ang(s):
        return deg(s) if callable(deg) else deg

    def pos(s):
        c = np.asarray(about(s), float)[:2]
        p = np.asarray(s["cmd"], float)
        q = _rz(ang(s)) @ np.r_[p[:2] - c, 0.0]
        return (c[0] + q[0], c[1] + q[1], p[2])

    def rot(s):
        r = np.asarray(s["rot"], float)
        return _rz(ang(s)) @ r if turn else r
    return pos, rot


def _crease(s):
    return B.crease_on_board(s["pose"]["board"])


def _tip_about(about, rot=None):
    """The centre to swing a hand's TCP about so that its fingertips (not its TCP) circle ``about``; ``rot``
    is the hand's (fixed) orientation, the left hand's fingertip grip by default."""
    off = PEG_DOWN * np.asarray(L_PEG if rot is None else rot, float)[:2, 2]
    return lambda s: np.asarray(about(s), float)[:2] - off


def _peg_outside(r):
    """The right hand's fingertips against the outside of the cover (on the jacket), ``r`` out from the spine."""
    def pos(s):
        along, inward = _frame(s, _cover_now(s))
        face = _hinge(s) + along * r - inward * (B.BOARD_T / 2 + B.JKT_T)
        return _peg_behind(face, inward, 0.0, R_PEG)
    return pos


def _fold_now(s):
    return B.fold_deg(s["pose"]["board"], s["pose"]["flap"])


def _lead(rot, u):
    """How far ahead of the TCP (along unit ``u``) the closed, leaned fingertips reach below the top edges:
    they run along the approach from the TCP to PEG_DOWN below it, PEG_JAW thick across the jaws and PEG_W
    wide across their side faces."""
    r = np.asarray(rot, float)
    k = float(np.dot(r[:2, 2], u))
    return max(PEG_DOWN * k, 0.0) + abs(float(np.dot(r[:2, 0], u))) * PEG_JAW + abs(float(np.dot(r[:2, 1], u))) * PEG_W


def _peg_behind(point, u, gap, rot):
    """TCP for closed, leaned fingertips that stand ``gap`` short of the plane through ``point`` facing
    them (normal -u), ready to push along ``u``; at ZP."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    c = np.asarray(point, float)[:2] - u * (gap + _lead(rot, u))
    t = np.array([-u[1], u[0]])
    c = c - t * float(np.dot(t, PEG_DOWN * np.asarray(rot, float)[:2, 2]))    # the tip, not the TCP, on that line
    return np.array([c[0], c[1], ZP])


def _peg_on_flap(s):
    """Fingertips just outside the flap (on the side it turns away from), S_PUSH out from its crease."""
    flap = s["pose"]["flap"]
    p = B.flap_edge(flap, S_PUSH)[:2]
    f = B.flap_dir(flap)[:2]
    out = np.array([-f[1], f[0]])                                # turning in, its free edge moves toward (f_y, -f_x)
    return _peg_behind(p + out * B.JKT_T / 2, -out, 0.003, L_PEG)


def _up(dz, dx=0.0, dy=0.0):
    return lambda s: tuple(np.asarray(s["cmd"]) + np.array([dx, dy, dz]))


def _steps():
    lp = {th: _cover_pinch(th) for th in (B.START_OPEN, HANDOVER)}
    t_held = 1.5 + 1.8 + 1.0 + 0.6                              # the right hand has the cover pinched
    t_fold = t_held + 1.5 + 1.0 + 3 * 0.9                         # the left hand has turned the flap in
    t_regrip = 0.4 + 0.6 + 1.0 + 0.6                              # the right hand moves to the cover's outside
    keep = lambda target: _swing(lambda s: -(_cover_now(s) - target), _tip_about(_hinge), turn=False)
    press = lambda target: _swing(lambda s: -(_cover_now(s) - target), _tip_about(_hinge, R_PEG), turn=False)
    left = [
        # label, seconds, position, rotation, finger, steer
        ("look", 1.5, REST["left"], L_PEG, FINGER_CLOSED, None),
        ("wait for the cover to be held", t_held - 1.5, None, None, FINGER_CLOSED, None),
        ("over the loose flap", 1.5, _above(_peg_on_flap), L_PEG, FINGER_CLOSED, None),
        ("fingertips down beside the flap", 1.0, _peg_on_flap, L_PEG, FINGER_CLOSED, None),
        ("push the flap round the fore-edge", 0.9, *_swing(lambda s: -(75.0 - _fold_now(s)), _tip_about(_crease), turn=False), FINGER_CLOSED, None),
        ("push it on round", 0.9, *_swing(lambda s: -(130.0 - _fold_now(s)), _tip_about(_crease), turn=False), FINGER_CLOSED, None),
        ("press it flat inside the cover", 0.9, *_swing(lambda s: -(FOLD - _fold_now(s)), _tip_about(_crease), turn=False), FINGER_CLOSED, None),
        ("keep it in as the cover swings in", 1.2, *keep(HANDOVER), FINGER_CLOSED, None),
        ("keep holding it in", t_regrip, None, None, FINGER_CLOSED, None),
        ("keep it in as the cover is pressed in", 1.0, *keep(LET_GO), FINGER_CLOSED, None),
        ("let the flap go: up out of the cover", 1.0, _up(0.08), None, FINGER_CLOSED, None),
        ("stand clear", 1.5, REST["left"], None, FINGER_OPEN, None),
        ("done", 6.0, REST["left"], None, FINGER_OPEN, None),
    ]
    right = [
        ("look", 1.5, REST["right"], _grip_rot((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.7), FINGER_OPEN, None),
        ("over the cover's top edge", 1.8, _above(lp[B.START_OPEN][0]), lp[B.START_OPEN][1], PART, None),
        ("down onto the cover", 1.0, lp[B.START_OPEN][0], lp[B.START_OPEN][1], PART, None),
        ("pinch the cover and jacket", 0.6, None, None, FINGER_CLOSED, None),
        ("hold the cover open", t_fold - t_held, None, None, FINGER_CLOSED, None),
        ("swing it in over the flap", 1.2, *lp[HANDOVER], FINGER_CLOSED, None),
        ("let go of the cover", 0.4, None, None, PART, None),
        ("up off the cover", 0.6, _up(0.05), None, PART, None),
        ("over the outside of the cover", 1.0, _above(_peg_outside(R_PRESS)), R_PEG, FINGER_CLOSED, None),
        ("fingertips down on the jacket", 0.6, _peg_outside(R_PRESS), R_PEG, FINGER_CLOSED, None),
        ("press the cover in on the flap", 1.0, *press(LET_GO - 3.0), FINGER_CLOSED, None),
        ("hold it while the flap is let go", 1.0, None, None, FINGER_CLOSED, None),
        ("press it shut", 0.8, *press(-1.0), FINGER_CLOSED, None),
        ("hold it shut", 1.0, None, None, FINGER_CLOSED, None),
        ("up", 1.0, _up(0.08, dx=-0.03), None, FINGER_CLOSED, None),
        ("stand clear", 1.5, REST["right"], None, FINGER_OPEN, None),
        ("done", 4.0, REST["right"], None, FINGER_OPEN, None),
    ]
    return {"left": left, "right": right}


def demo():
    steps = _steps()
    return {side: [Waypoint(sec, REST[side], FINGER_OPEN, label, DOWN) for label, sec, *_ in steps[side]]
            for side in ("left", "right")}


NO_SAG_FIX = ("pinch", "let go", "push", "keep", "press", "swing", "hold", "shut", "off the", "fingertips down")


def _smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3 - 2 * u)


class ArmSequence:
    """Plays one arm's steps: each glides from where the last one ended to its own target."""

    def __init__(self, side, steps, plan):
        self.side, self.steps, self.plan = side, steps, plan
        self.k = -1
        self.t0 = 0.0
        self.nom = None                      # where the plan is now: (pos, rot, finger)
        self.start = self.goal = None
        self.fb = np.zeros(3)                # slow correction for the arm's own sag
        self.sent = None

    def _begin(self, k, t):
        from yam_arena import ik
        label, sec, pos, rot, finger, steer = self.steps[k]
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
