"""Stateful scorer for setting a pair of binoculars to a user's eye spacing and focus mark, from measured
body poses and the robot's measured TCPs only."""
from __future__ import annotations

import numpy as np

import bino as B

HZ = 20
HOLD_S = 1.0
IPD_TOL = 0.0015                      # m: eye spacing within this of 62 mm
FOCUS_TOL = 15.0                      # deg: focus wheel within this of the 5 m mark
LYING = 20.0                          # deg: each barrel's axis within this of level ...
ON_BENCH = 0.015                      # m: ... and its underside within this of the bench
DROP_SPEED = 0.5                      # m/s: the binoculars count as dropped if a barrel comes down faster than this
CLEAR = 0.06                          # m: a hand counts as let go this far from the binoculars


def _outside(poses, p, margin):
    """A point farther than ``margin`` from both barrels (as cylinders with their eyepiece cups)."""
    for n in ("barrel_l", "barrel_r"):
        q = np.asarray(poses[n], float)
        local = B.R_(q).T @ (np.asarray(p, float) - q[:3])
        along = max(abs(local[0] + B.CUP_L / 2) - (B.L + B.CUP_L) / 2, 0.0)
        across = max(float(np.hypot(local[1], local[2])) - B.R, 0.0)
        if float(np.hypot(along, across)) <= margin:
            return False
    return True


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.dropped = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        p = {n: np.asarray(root_poses[n], float) for n in B.OBJECT_NAMES}
        bl, br = p["barrel_l"], p["barrel_r"]
        for n in ("barrel_l", "barrel_r"):
            v = np.asarray(getattr(objects[n], "lin_vel", np.zeros(3)), float) if n in objects else np.zeros(3)
            if v[2] < -DROP_SPEED:
                self.dropped = True
        ipd = B.ipd(bl, br)
        focus = B.focus_deg(bl, p["wheel"])
        spaced = bool(abs(ipd - B.IPD_TARGET) <= IPD_TOL)
        focused = bool(abs(focus - B.FOCUS_TARGET) <= FOCUS_TOL)
        lying = all(B.tilt_deg(q) <= LYING and q[2] - B.R - B.TOP <= ON_BENCH for q in (bl, br))
        clear = all(_outside(p, np.asarray(robot[s]["tcp"], float)[:3], CLEAR) for s in ("left", "right"))
        score = 0.0
        if spaced or focused:
            score = 0.25
        if spaced and focused:
            score = 0.5
        if spaced and focused and lying:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.dropped:
            score = 0.0
        return {"score": score, "success": score == 1.0, "ipd_mm": round(1000 * ipd, 2), "focus_deg": round(focus, 1),
                "spaced": spaced, "focused": focused, "lying": bool(lying), "dropped": bool(self.dropped),
                "hands_clear": bool(clear), "held_frames": self.held}
