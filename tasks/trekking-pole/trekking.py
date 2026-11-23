"""Geometry and measurements for setting a trekking pole to someone's length with its flip lock.

Station frame: metres, x away from the arm bases, y left, z up; TOP is the bench top.

A two-section trekking pole lies on the bench along y, its grip toward +y (the left arm) and its tip
toward -y (the right arm), resting on its foam grip and its flip-lock collar. The lower section slides
stiffly in the upper one (damped). The flip-lock lever is hinged on the collar's side facing the arms:
shut, it lies flat along the collar; open, it stands out toward the arms. Length marks are printed on
the lower section every 5 cm, read at the collar's edge.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

FOAM = (0.12, 0.12, 0.13)
ALU = (0.20, 0.32, 0.55)               # anodised blue upper shaft
ALU2 = (0.72, 0.73, 0.75)              # bare lower section
PLASTIC = (0.15, 0.15, 0.16)
LEVER = (0.85, 0.35, 0.15)             # orange lever
WHITE = (0.95, 0.95, 0.93)
RED = (0.85, 0.15, 0.12)

X0 = 0.340                             # the pole's axis, along y
RG = 0.015                             # grip and collar radius (8-sided, across flats): the pole rests on these
AXIS_Z = TOP + RG
Y_END = 0.570                          # the grip's top end
GRIP_L = 0.130
R_UP = 0.009                           # upper shaft
UP_L = 0.600                           # grip top to the collar's top edge
COLLAR_L = 0.035
R_LO = 0.0075                          # lower section
LO_L = 0.620                           # lower section, inner end to tip
LENGTH0 = 1.100                        # the pole's length at the start (m)
LENGTH_TARGET = 1.200                  # the length to set
LO_MIN, LO_MAX = 1.000, 1.250          # the lower section's travel, as pole lengths
LO_DAMP = 600.0                        # N s/m: the lower section is stiff to slide

LEV_T, LEV_H, LEV_L = 0.004, 0.014, 0.045   # flip-lock lever: thickness (x), height (z), length (along the pole)
TAB_W, TAB_H = 0.008, 0.030           # the raised thumb tab at the lever's free end
TAB_Y = -LEV_L / 2 + 0.004             # ... on the lever's frame
LEV_OPEN = -95.0                       # its travel (deg about z): 0 shut, -95 open
LEVER_DAMP = 0.02                      # N m s per radian

U_COLLAR0 = UP_L                       # collar from u = UP_L to UP_L + COLLAR_L (u: distance from the grip end)
U_COLLAR1 = UP_L + COLLAR_L


def y_of(u):
    return Y_END - u


def _p(shape, size, offset, **kw):
    return Piece(shape, tuple(float(v) for v in size), tuple(float(v) for v in offset), **kw)


def _octagon(r, length, offset, color, collide=True):
    """An 8-sided rod along y: four slabs turned 45 degrees apart, under a round cover."""
    side = 2 * r * math.tan(math.pi / 8)
    out = [_p("box", (side, length, 2 * r), offset, rot=(0.0, k * math.pi / 4, 0.0), color=color, collide=collide)
           for k in range(4)]
    rr = r / math.cos(math.pi / 8)
    out.append(_p("cylinder", (rr, length, rr), offset, axis="Y", color=color, collide=False))
    return out


POLE = ("upper", "lower", "lever")


def _others(name):
    return tuple(n for n in POLE if n != name)


UPPER_POS = (X0, y_of(UP_L / 2), AXIS_Z)      # the upper section's origin: halfway along it


def upper() -> Part:
    c = UPPER_POS
    def at(u):
        return u - UP_L / 2                   # u along the pole -> the upper body's local -y offset
    side = 2 * R_UP * math.tan(math.pi / 8)
    pieces = list(_octagon(R_UP, UP_L - GRIP_L, (0.0, -(at((GRIP_L + UP_L) / 2)), 0.0), ALU))
    pieces += _octagon(RG, GRIP_L, (0.0, -at(GRIP_L / 2), 0.0), FOAM)
    pieces += _octagon(RG, COLLAR_L, (0.0, -at(U_COLLAR0 + COLLAR_L / 2), 0.0), PLASTIC)
    pieces.append(_p("cylinder", (RG + 0.002, 0.012, RG + 0.002), (0.0, -at(0.006), 0.0), axis="Y", color=PLASTIC, collide=False))
    pieces.append(_p("cylinder", (RG + 0.0012, 0.004, RG + 0.0012), (0.0, -at(GRIP_L + 0.002), 0.0), axis="Y", color=PLASTIC,
                     collide=False))
    # the wrist strap's loop at the top of the grip
    pieces.append(_p("box", (0.004, 0.030, 0.006), (0.0, -at(0.0) + 0.012, RG - 0.002), color=RED, collide=False))
    return Part("upper", pos=c, size=(side, 0.010, 2 * R_UP), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=0.17, color=ALU, friction=(1.2, 1.0), restitution=0.0, contact_offset=0.002, no_sleep=True,
                no_collide=_others("upper"))


# The lower section: its inner end sits inside the upper shaft. Its origin is its own midpoint.
U_LO_TIP0 = LENGTH0
LOWER_POS = (X0, y_of(U_LO_TIP0 - LO_L / 2), AXIS_Z)


def lower() -> Part:
    c = LOWER_POS
    side = 2 * R_LO * math.tan(math.pi / 8)
    pieces = list(_octagon(R_LO, LO_L, (0.0, 0.0, 0.0), ALU2)[1:])
    # rubber tip and carbide point
    pieces.append(_p("cylinder", (0.0095, 0.040, 0.0095), (0.0, -LO_L / 2 + 0.020, 0.0), axis="Y", color=PLASTIC,
                     collide=False))
    pieces.append(_p("cone", (0.006, 0.014, 0.006), (0.0, -LO_L / 2 - 0.004, 0.0), axis="Y", color=PLASTIC,
                     collide=False))
    # length marks every 5 cm: a mark at pole length Lm sits on the lower section at the collar edge when the
    # pole is Lm long; on the lower body that is (tip - (Lm - U_COLLAR1)) from the tip
    for k in range(0, 6):
        lm = 1.00 + 0.05 * k
        d_from_tip = lm - U_COLLAR1
        yloc = -LO_L / 2 + d_from_tip
        big = abs(lm - LENGTH_TARGET) < 1e-6
        pieces.append(_p("cylinder", (R_LO / math.cos(math.pi / 8) + 0.0004, 0.004 if big else 0.0018,
                                      R_LO / math.cos(math.pi / 8) + 0.0004),
                         (0.0, yloc, 0.0), axis="Y", color=RED if big else WHITE, collide=False))
    return Part("lower", pos=c, size=(side, LO_L, 2 * R_LO), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=0.09, color=ALU2, friction=(1.0, 0.9), restitution=0.0, contact_offset=0.002, no_sleep=True,
                no_collide=_others("lower"),
                spring=Spring(axis="Y", lower=-(LO_MAX - LENGTH0), upper=LENGTH0 - LO_MIN, stiffness=0.0,
                              damping=LO_DAMP, target=0.0, parent="upper", parent_pos=UPPER_POS))


HINGE_U = U_COLLAR0                    # the lever's hinge: at the collar's grip-side edge
HINGE_X = X0 - RG - LEV_T / 2 - 0.0008
HINGE = (HINGE_X, y_of(HINGE_U), AXIS_Z)
LEVER_POS = (HINGE_X, y_of(HINGE_U + LEV_L / 2), AXIS_Z)


def lever() -> Part:
    """The flip-lock lever, shut: a plate along the collar on its side facing the arms, its free end toward
    the tip, with a raised thumb tab standing up at that end."""
    pieces = [
        _p("box", (TAB_W, TAB_W, TAB_H), (0.0, TAB_Y, LEV_H / 2 + TAB_H / 2), color=LEVER),
        _p("box", (0.0006, LEV_L - 0.010, 0.002), (-LEV_T / 2 - 0.0003, 0.0, 0.0), color=WHITE, collide=False),
    ]
    return Part("lever", pos=LEVER_POS, size=(LEV_T, LEV_L, LEV_H), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=0.008, color=LEVER, friction=(0.9, 0.8), restitution=0.0, contact_offset=0.0015, no_sleep=True,
                no_collide=_others("lever"),
                hinge=Hinge(pivot=HINGE, axis="Z", damping=LEVER_DAMP, lower_deg=LEV_OPEN, upper_deg=0.0,
                            parent="upper", parent_pos=UPPER_POS))


def object_specs():
    return [upper(), lower(), lever()]


OBJECT_NAMES = ("upper", "lower", "lever")


# -- measurements ---------------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_(pose):
    return quat_matrix(np.asarray(pose, float)[3:7])


def point(pose, local):
    p = np.asarray(pose, float)
    return p[:3] + R_(p) @ np.asarray(local, float)


D_LO = LOWER_POS[1] - UPPER_POS[1]


def length(up, lo):
    """The pole's length, grip top to tip (m), from the lower section's travel in the upper one."""
    pu, pl = np.asarray(up, float), np.asarray(lo, float)
    along = float(np.dot(pl[:3] - pu[:3], R_(pu)[:, 1])) - D_LO
    return LENGTH0 - along


def lever_deg(up, lev):
    """The lever's turn about the pole's vertical from shut (deg): 0 shut, about -95 open."""
    r = R_(up).T @ R_(lev)
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def axis_tilt(up):
    v = R_(up)[:, 1]
    return math.degrees(math.asin(min(1.0, abs(float(v[2])))))


def grip_point(up, u):
    """A point on the pole's axis u from the grip end, on the upper section's frame."""
    return point(up, (0.0, -(u - UP_L / 2), 0.0))


def tab_top(lev):
    """The lever's thumb tab, its top centre (measured)."""
    return point(lev, (0.0, TAB_Y, LEV_H / 2 + TAB_H))
