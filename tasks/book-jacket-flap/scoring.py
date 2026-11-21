"""Stateful scorer for tucking a hardback's loose dust-jacket flap back inside its front cover, from
measured body poses and the robot's measured TCPs only."""
from __future__ import annotations

import numpy as np

import book as B

HZ = 20
HOLD_S = 1.0
TURNED_IN = 150.0                     # deg: the flap is round the fore-edge, on the inside of the cover
TUCKED = 165.0                        # deg: ... and lies flat against it
SHUT = 6.0                            # deg: the cover is shut on the pages (the flap is between)
MOVED = 0.020                         # m: the book stays where it stood
TILT = 10.0                           # deg: ... upright, and not turned round on the bench
FELL = 30.0                           # deg: the book has fallen over
CLEAR = 0.06                          # m: a hand counts as let go this far outside the book's outline


def _outside(p, margin):
    """A point (station frame) farther than ``margin`` from a box round the shut book, its flap and the bookend."""
    lo, hi = B.outline()
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return float(np.linalg.norm(d)) > margin


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.book0 = None
        self.moved = 0.0
        self.tilt = 0.0
        self.yaw0 = None
        self.turned = 0.0
        self.turned_in = False
        self.was_torn = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        book = np.asarray(root_poses["book"], float)
        board = np.asarray(root_poses["board"], float)
        flap = np.asarray(root_poses["flap"], float)
        if self.book0 is None:
            self.book0 = book[:2].copy()
            self.yaw0 = B.yaw_deg(book)
        self.moved = max(self.moved, float(np.linalg.norm(book[:2] - self.book0)))
        self.tilt = max(self.tilt, B.tilt_deg(book))
        self.turned = max(self.turned, abs(B.wrap(B.yaw_deg(book) - self.yaw0)))
        self.was_torn = self.was_torn or B.torn(board, flap)
        cover = B.cover_deg(book, board)
        fold = B.fold_deg(board, flap)
        self.turned_in = self.turned_in or fold >= TURNED_IN
        shut_on_flap = bool(cover <= SHUT and fold >= TUCKED)
        square = bool(self.moved <= MOVED and self.tilt <= TILT and self.turned <= TILT)
        clear = all(_outside(np.asarray(robot[s]["tcp"], float)[:3], CLEAR) for s in ("left", "right"))
        score = 0.0
        if self.turned_in:
            score = 0.25
        if shut_on_flap:
            score = 0.5
        if shut_on_flap and square:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.tilt > FELL or self.was_torn:
            score = 0.0
        return {"score": score, "success": score == 1.0, "cover_deg": round(cover, 1),
                "fold_deg": round(fold, 1), "turned_in": bool(self.turned_in), "shut_on_flap": shut_on_flap,
                "book_in_place": square, "book_moved_mm": round(1000 * self.moved, 1), "book_tilt_deg": round(self.tilt, 1),
                "book_turned_deg": round(self.turned, 1),
                "torn": bool(self.was_torn), "hands_clear": bool(clear), "held_frames": self.held}
