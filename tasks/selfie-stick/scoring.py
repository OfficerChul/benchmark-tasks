"""Stateful scorer for readying a selfie stick (phone clamped in, stick out to full length), from measured
body poses and the robot's measured TCPs only."""
from __future__ import annotations

import numpy as np

import selfie as S

HZ = 20
HOLD_S = 1.0
FULL_TOL = 0.010                      # m: pulled out to within this of the full extension (every tube at its stop)
SEAT_ALONG = 0.004                    # m: phone against the fixed jaw within this (along the stick)
SEAT_ACROSS = 0.015                   # m: phone's centre within this of the clamp's centre line (across the stick)
SEAT_DOWN = 0.003                     # m: phone on the clamp's bed within this
SEAT_TILT = 8.0                       # deg: phone flat on the bed within this
JAW_ON = 0.002                        # m: the spring jaw resting on the phone's edge (it is gripping it), within this
LYING = 10.0                          # deg: the stick's axis within this of level ...
ON_BENCH = 0.015                      # m: ... and the handle's underside within this of the bench
DROP_SPEED = 0.5                      # m/s: the phone counts as dropped if it comes down faster than this
CLEAR = 0.06                          # m: a hand counts as let go this far from the stick and the phone


def _box_dist(pose, half, p):
    q = np.asarray(pose, float)
    local = S.R_(q).T @ (np.asarray(p, float) - q[:3])
    return float(np.linalg.norm(np.maximum(np.abs(local) - np.asarray(half, float), 0.0)))


def _outside(poses, p, margin):
    """A point farther than ``margin`` from the stick (handle to head tip) and the phone."""
    h = np.asarray(poses["handle"], float)
    ext = S.extension(h, poses["tube3"])
    y_lo, y_hi = -S.LH / 2, (S.HEAD_E + S.NECK + S.BED_L + ext) - (S.Y_MOUTH - S.LH / 2)
    local = S.R_(h).T @ (np.asarray(p, float) - h[:3])
    along = max(y_lo - local[1], local[1] - y_hi, 0.0)
    across = max(float(np.hypot(local[0], local[2])) - S.RH, 0.0)
    if float(np.hypot(along, across)) <= margin:
        return False
    return _box_dist(poses["phone"], (S.PHONE_L / 2, S.PHONE_W / 2, S.PHONE_T / 2), p) > margin


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.dropped = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        p = {n: np.asarray(root_poses[n], float) for n in S.OBJECT_NAMES}
        v = np.asarray(getattr(objects["phone"], "lin_vel", np.zeros(3)), float) if "phone" in objects else np.zeros(3)
        if v[2] < -DROP_SPEED:
            self.dropped = True
        ext = S.extension(p["handle"], p["tube3"])
        rel = S.phone_rel(p["tube3"], p["phone"])
        tilt = S.phone_tilt(p["tube3"], p["phone"])
        jaw = S.jaw_open(p["tube3"], p["jaw"])
        extended = bool(ext >= S.FULL - FULL_TOL)
        on_phone = abs(jaw - (S.JAW_SEATED + float(rel[1]))) <= JAW_ON    # the jaw's face is at the phone's edge
        clamped = bool(abs(rel[1]) <= SEAT_ALONG and abs(rel[0]) <= SEAT_ACROSS and abs(rel[2]) <= SEAT_DOWN
                       and tilt <= SEAT_TILT and on_phone)
        lying = bool(S.axis_tilt(p["handle"]) <= LYING and p["handle"][2] - S.RH - S.TOP <= ON_BENCH)
        clear = all(_outside(p, np.asarray(robot[s]["tcp"], float)[:3], CLEAR) for s in ("left", "right"))
        score = 0.0
        if extended or clamped:
            score = 0.25
        if extended and clamped:
            score = 0.5
        if extended and clamped and lying:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.dropped:
            score = 0.0
        return {"score": score, "success": score == 1.0, "extension_mm": round(1000 * ext, 1),
                "length_mm": round(1000 * S.length(p["handle"], p["tube3"]), 1),
                "phone_off_mm": [round(1000 * float(x), 1) for x in rel], "phone_tilt_deg": round(tilt, 1),
                "jaw_mm": round(1000 * jaw, 1), "extended": extended, "clamped": clamped, "lying": lying,
                "dropped": bool(self.dropped), "hands_clear": bool(clear), "held_frames": self.held}
