"""Stateful scorer for setting a trekking pole to 120 cm with its flip lock shut, from measured body poses
and the robot's measured TCPs only."""
from __future__ import annotations

import numpy as np

import trekking as T

HZ = 20
HOLD_S = 1.0
SET_TOL = 0.005                       # m: the pole's length within this of 120 cm
NEAR_TOL = 0.030                      # m: ... within this for partial credit
SHUT = 10.0                           # deg: the lever within this of shut
LYING = 10.0                          # deg: the pole's axis level within this ...
ON_BENCH = 0.015                      # m: ... and the grip's underside within this of the bench
DROP_SPEED = 0.5                      # m/s: the pole counts as dropped if it comes down faster than this
CLEAR = 0.06                          # m: a hand counts as let go this far from the pole
FORCED = 0.005                        # m: the length changing this much with the lever shut is forcing the lock


def _seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / float(np.dot(ab, ab)), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _outside(poses, p, margin):
    """A point farther than ``margin`` from the pole (grip end to tip, as a 15 mm rod) and its lever."""
    up = poses["upper"]
    a = T.grip_point(up, 0.0)
    b = T.grip_point(up, T.length(up, poses["lower"]))
    if _seg_dist(p, a, b) - T.RG <= margin:
        return False
    lev = np.asarray(poses["lever"], float)
    local = T.R_(lev).T @ (np.asarray(p, float) - lev[:3])
    out = np.maximum(np.abs(local) - np.array([T.LEV_T / 2 + 0.003, T.LEV_L / 2, T.LEV_H / 2 + 0.002]), 0.0)
    return float(np.linalg.norm(out)) > margin


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.dropped = False
        self.forced = False
        self.locked_at = None             # the length the shut lock holds the pole at

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        p = {n: np.asarray(root_poses[n], float) for n in T.OBJECT_NAMES}
        for n in ("upper", "lower"):
            v = np.asarray(getattr(objects[n], "lin_vel", np.zeros(3)), float) if n in objects else np.zeros(3)
            if v[2] < -DROP_SPEED:
                self.dropped = True
        length = T.length(p["upper"], p["lower"])
        lev = T.lever_deg(p["upper"], p["lever"])
        near = bool(abs(length - T.LENGTH_TARGET) <= NEAR_TOL)
        set_ = bool(abs(length - T.LENGTH_TARGET) <= SET_TOL)
        shut = bool(abs(lev) <= SHUT)
        if shut:
            if self.locked_at is None:
                self.locked_at = length
            elif abs(length - self.locked_at) > FORCED:
                self.forced = True
        else:
            self.locked_at = None
        lying = bool(T.axis_tilt(p["upper"]) <= LYING and p["upper"][2] - T.RG - T.TOP <= ON_BENCH)
        clear = all(_outside(p, np.asarray(robot[s]["tcp"], float)[:3], CLEAR) for s in ("left", "right"))
        score = 0.0
        if near:
            score = 0.25
        if set_:
            score = 0.5
        if set_ and shut:
            score = 0.75
        full = score == 0.75 and lying and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.dropped or self.forced:
            score = 0.0
        return {"score": score, "success": score == 1.0, "length_mm": round(1000 * length, 1),
                "lever_deg": round(lev, 1), "set": set_, "shut": shut, "lying": lying,
                "dropped": bool(self.dropped), "forced": bool(self.forced), "hands_clear": bool(clear), "held_frames": self.held}
