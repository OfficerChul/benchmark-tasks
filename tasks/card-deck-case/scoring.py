"""Stateful scorer for sliding a deck of playing cards back into its hard plastic case and closing the lid,
from measured body poses and the robot's measured TCPs only."""
from __future__ import annotations

import numpy as np

import cards as C

HZ = 20
HOLD_S = 1.0
IN_TILT = 10.0                        # deg: the deck lies square in the case
PROUD = 0.001                         # m: its near end sticks out of the mouth no more than this
SHUT = 6.0                            # deg: the lid hangs down over the mouth
FLAT = 15.0                           # deg: the case still lies on its back
CLEAR = 0.06                          # m: a hand counts as let go this far outside the case


def _outside(box, p, margin):
    """A point farther than ``margin`` from the closed case (its own frame)."""
    q = C.to_box(box, p) + C.BOX0
    lo = np.array([C.BX - C.IN_W / 2 - C.WALL, C.YM - C.WALL, C.TOP])
    hi = np.array([C.BX + C.IN_W / 2 + C.WALL, C.Y_END, C.Z_OUT])
    d = np.maximum(np.maximum(lo - q, q - hi), 0.0)
    return float(np.linalg.norm(d)) > margin


def deck_in(box, deck):
    depth, xz, tilt = C.deck_in_box(box, deck)
    inside = abs(xz[0]) <= C.IN_W / 2 and abs(xz[1]) <= C.IN_D / 2
    return bool(depth >= -PROUD and tilt <= IN_TILT and inside)


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.was_torn = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        p = {n: np.asarray(root_poses[n], float) for n in C.OBJECT_NAMES}
        box, lid, deck = p["box"], p["lid"], p["deck"]
        self.was_torn = self.was_torn or C.torn(box, lid)
        lid_a = C.lid_deg(box, lid)
        inside = deck_in(box, deck)
        shut = bool(-SHUT <= lid_a <= SHUT)
        lying = bool(C.tilt_deg(box) <= FLAT and box[2] < C.BOX0[2] + 0.01)
        clear = all(_outside(box, np.asarray(robot[s]["tcp"], float)[:3], CLEAR) for s in ("left", "right"))
        score = 0.0
        if inside:
            score = 0.25
        if inside and shut:
            score = 0.5
        if inside and shut and lying:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.was_torn:
            score = 0.0
        depth, xy, tilt = C.deck_in_box(box, deck)
        return {"score": score, "success": score == 1.0, "deck_in": inside, "deck_depth_mm": round(1000 * depth, 1),
                "deck_tilt_deg": round(tilt, 1), "lid_deg": round(lid_a, 1), "lid_shut": shut,
                "case_lying": lying, "case_xy": [round(float(v), 3) for v in box[:2]], "torn": bool(self.was_torn),
                "hands_clear": bool(clear), "held_frames": self.held}
