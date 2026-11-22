"""Geometry and measurements for setting a pair of binoculars for a user: eye spacing and focus.

Station frame: metres, x away from the arm bases, y left, z up; TOP is the bench top.

A pair of 10x42 roof-prism binoculars lies on the bench on its two barrels, eyepieces toward the arms.
The barrels (left at +y, right at -y) turn on one central hinge along x; the centre focus wheel sits on
the hinge axis just ahead of the eyepieces. Each barrel's axis stands A from the hinge axis, at an angle
beta below the hinge's level, so the eye spacing (the distance between the barrel axes) is 2 A cos(beta).
Lying on the bench, folding the barrels in (beta growing) lifts the hinge.

Angles (degrees):
  spread   the hinge's turn from its spawn pose, right barrel relative to left (folding in is positive);
  focus    the focus wheel's turn relative to the left barrel, from its start (the "infinity" mark).
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

ARMOUR = (0.13, 0.15, 0.13)           # black-green rubber armour
GLASS = (0.10, 0.16, 0.22)
METAL = (0.55, 0.56, 0.58)
WHITE = (0.95, 0.95, 0.93)
RIB = (0.92, 0.80, 0.25)                # yellow knurl ribs on the focus wheel

BX = 0.300                            # middle of the barrels' length
A = 0.037                             # hinge axis to each barrel's axis
R = 0.0215                            # barrel (rubber armour) half width across its flats (16-sided section)
L = 0.120                             # barrel length, eyepiece cups excluded
SIDE16 = 2 * R * math.tan(math.pi / 16)  # the barrels' 16-sided section is built of eight slabs this thick
CUP_R, CUP_L = 0.0190, 0.022          # eyepiece cups
IPD_START = 0.07274                   # eye spacing at the start (m): its barrels lie on a flat, as they do folded to 62 mm
IPD_TARGET = 0.062                    # the user's eye spacing
IPD_MIN, IPD_MAX = 0.056, 0.074       # the hinge's travel
BARREL_MASS = 0.31                    # each barrel with its half of the bridge (650 g the pair)
WHEEL_R, WHEEL_W = 0.0110, 0.016      # focus wheel
WHEEL_X = BX - L / 2 + 0.020          # ... centre, 20 mm in from the eyepiece end of the barrels
WHEEL_MASS = 0.012
FOCUS_TARGET = -120.0                 # degrees: the "5 m" mark, a third of a turn with the wheel's top rolling toward +y (left)

BETA0 = math.degrees(math.acos(IPD_START / (2 * A)))
H0 = TOP + R + A * math.sin(math.radians(BETA0))          # hinge height lying on the bench at the start
HINGE = (BX, 0.0, H0)

# The hinge and the focus wheel turn stiffly (damped).
HINGE_DAMP = 0.08                      # N m s per radian (as PhysX applies the drive here)
WHEEL_DAMP = 2.0


def beta_of(ipd):
    return math.degrees(math.acos(ipd / (2 * A)))


def spread_of(ipd):
    """Hinge angle (deg) for an eye spacing (m)."""
    return 2.0 * (beta_of(ipd) - BETA0)


def ipd_of(spread):
    return 2 * A * math.cos(math.radians(BETA0 + spread / 2.0))


def _p(shape, size, offset, **kw):
    return Piece(shape, tuple(float(v) for v in size), tuple(float(v) for v in offset), **kw)


def _barrel(side: int) -> Part:
    """One barrel (side +1 left, -1 right) with its half of the hinge bridge; spawned at the start
    spacing, lying on the bench."""
    b = math.radians(BETA0)
    c = (BX, side * A * math.cos(b), H0 - A * math.sin(b))
    hx = -side * A * math.cos(b)                              # hinge axis, from the barrel's centre
    hz = A * math.sin(b)
    pieces = [
        # 16-sided section: the root slab plus seven more, each turned 22.5 degrees further about the
        # barrel's axis (hidden under the round armour), so the barrel rests on a flat both at the start
        # and folded to the target spacing
        *[_p("box", (L, 2 * R, SIDE16), (0.0, 0.0, 0.0), rot=(k * math.pi / 8, 0.0, 0.0), color=ARMOUR)
          for k in range(1, 8)],
        _p("cylinder", (R / math.cos(math.pi / 16), L, R / math.cos(math.pi / 16)), (0.0, 0.0, 0.0), axis="X",
           color=ARMOUR, collide=False),
        *[_p("box", (CUP_L, 2 * CUP_R, 2 * CUP_R * math.tan(math.pi / 8)), (-L / 2 - CUP_L / 2, 0.0, 0.0),
             rot=(k * math.pi / 4, 0.0, 0.0), color=ARMOUR) for k in range(4)],
        _p("cylinder", (CUP_R / math.cos(math.pi / 8), CUP_L, CUP_R / math.cos(math.pi / 8)), (-L / 2 - CUP_L / 2, 0.0, 0.0),
           axis="X", color=ARMOUR, collide=False),
        _p("cylinder", (CUP_R - 0.004, 0.001, CUP_R - 0.004), (-L / 2 - CUP_L - 0.0004, 0.0, 0.0), axis="X",
           color=GLASS, collide=False),
        _p("cylinder", (R - 0.003, 0.001, R - 0.003), (L / 2 + 0.0004, 0.0, 0.0), axis="X", color=GLASS,
           collide=False),
    ]
    # bridge arm from the barrel to the hinge, ahead of and behind the focus wheel
    dy, dz = hx / A, hz / A                                   # unit line from the barrel's axis to the hinge axis
    roll = math.atan2(dz, dy)
    s_mid = (R + A) / 2
    for x0, x1 in ((-L / 2 + 0.002, WHEEL_X - BX - WHEEL_W / 2 - 0.004), (WHEEL_X - BX + WHEEL_W / 2 + 0.004, L / 2 - 0.030)):
        pieces.append(_p("box", (x1 - x0, A - R + 0.006, 0.012), ((x0 + x1) / 2, dy * s_mid, dz * s_mid),
                         rot=(roll, 0.0, 0.0), color=ARMOUR))
    if side > 0:
        # the hinge knuckles (on the left barrel's half), either side of the wheel
        for x0, x1 in ((-L / 2 + 0.002, WHEEL_X - BX - WHEEL_W / 2 - 0.001), (WHEEL_X - BX + WHEEL_W / 2 + 0.001, L / 2 - 0.030)):
            pieces.append(_p("cylinder", (0.009, x1 - x0, 0.009), ((x0 + x1) / 2, hx, hz), axis="X", color=ARMOUR))
        # printed focus marks on the bridge just behind the wheel: infinity at the top, and the target mark
        for ang, size in ((0.0, 0.0020), (FOCUS_TARGET, 0.0035)):
            a = math.radians(ang)
            pieces.append(_p("box", (0.0006, size, 0.006),
                             (WHEEL_X - BX + WHEEL_W / 2 + 0.0012, hx - math.sin(a) * (WHEEL_R - 0.002),
                              hz + math.cos(a) * (WHEEL_R - 0.002)), color=WHITE, collide=False))
    hinge = None
    if side < 0:
        hinge = Hinge(pivot=HINGE, axis="X", damping=HINGE_DAMP,
                      lower_deg=spread_of(IPD_MAX), upper_deg=spread_of(IPD_MIN), parent="barrel_l",
                      parent_pos=_barrel_pos(1))
    return Part("barrel_l" if side > 0 else "barrel_r", pos=c, size=(L, 2 * R, SIDE16), shape="box",
                exact_z=True, pieces=tuple(pieces), mass=BARREL_MASS, color=ARMOUR, friction=(0.55, 0.45),
                restitution=0.05, contact_offset=0.002, hinge=hinge, no_sleep=True)


def _barrel_pos(side):
    b = math.radians(BETA0)
    return (BX, side * A * math.cos(b), H0 - A * math.sin(b))


def wheel_object() -> Part:
    """The focus wheel: an octagon of three more slabs round its root slab (its knurled rim, under a round
    cover), on the hinge axis."""
    side = 2 * WHEEL_R * math.tan(math.pi / 8)
    pieces = [_p("box", (WHEEL_W, 2 * WHEEL_R, side), (0.0, 0.0, 0.0), rot=(k * math.pi / 4, 0.0, 0.0), color=ARMOUR,
                 ) for k in (1, 2, 3)]
    pieces.append(_p("cylinder", (WHEEL_R / math.cos(math.pi / 8), WHEEL_W, WHEEL_R / math.cos(math.pi / 8)), (0.0, 0.0, 0.0),
                     axis="X", color=ARMOUR, collide=False))
    rim = WHEEL_R / math.cos(math.pi / 8)
    for k in range(8):                                        # light knurl ribs round the rim, so its turning shows
        ang = (k + 0.5) * math.pi / 4
        pieces.append(_p("box", (WHEEL_W + 0.0002, 0.0026, 0.0012), (0.0, -math.sin(ang) * rim, math.cos(ang) * rim),
                         rot=(ang, 0.0, 0.0), color=RIB, collide=False))
    pieces.append(_p("box", (WHEEL_W + 0.0006, 0.0030, 0.0016), (0.0, 0.0, rim), color=WHITE, collide=False))
    return Part("wheel", pos=(WHEEL_X, 0.0, H0), size=(WHEEL_W, 2 * WHEEL_R, side), shape="box",
                exact_z=True, pieces=tuple(pieces), mass=WHEEL_MASS, color=ARMOUR, friction=(1.0, 0.9), restitution=0.0,
                contact_offset=0.0015, no_sleep=True,
                hinge=Hinge(pivot=HINGE, axis="X", damping=WHEEL_DAMP, lower_deg=-200.0,
                            upper_deg=320.0, parent="barrel_l", parent_pos=_barrel_pos(1)))


def object_specs():
    return [_barrel(1), _barrel(-1), wheel_object()]


OBJECT_NAMES = ("barrel_l", "barrel_r", "wheel")


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


def _turn_x(parent, child):
    """Turn of ``child`` relative to ``parent`` about the parent's x axis (deg)."""
    r = R_(parent).T @ R_(child)
    return math.degrees(math.atan2(r[2, 1], r[1, 1]))


def spread_deg(bl, br):
    return _turn_x(bl, br)


def ipd(bl, br):
    """Eye spacing: distance between the two barrel axes (m), from the measured barrel poses."""
    pl, pr = np.asarray(bl, float)[:3], np.asarray(br, float)[:3]
    ax = R_(bl) @ np.array([1.0, 0.0, 0.0])
    d = pl - pr
    return float(np.linalg.norm(d - np.dot(d, ax) * ax))


def focus_deg(bl, wheel):
    return _turn_x(bl, wheel)


def tilt_deg(pose):
    """How far a barrel's axis has turned out of the horizontal (deg)."""
    v = R_(pose) @ np.array([1.0, 0.0, 0.0])
    return math.degrees(math.asin(min(1.0, abs(float(v[2])))))


def hinge_point(bl):
    return point(bl, np.subtract(HINGE, _barrel_pos(1)))
