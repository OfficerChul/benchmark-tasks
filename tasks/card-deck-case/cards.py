"""Geometry and measurements for sliding a deck of playing cards back into its hard plastic case and
closing the lid.

Station frame: metres, x away from the arm bases, y left, z up; TOP is the bench top.

The case lies on its back on the bench, its mouth facing -y (toward the right hand) and its closed end
toward +y. Its lid is hinged along the mouth edge of the upper wall: it stands open leaning back over the
case against its stop, and shut it hangs down over the mouth. The deck lies face down on the bench in
front of the mouth, its long edges along y. The case is light and slides easily on the bench, so pushing
the deck in shoves the case along unless the other hand holds it.

Angles (degrees, from measured body orientations, relative to the case):
  lid    0 hanging shut over the mouth, growing as it swings out and up (200 at the start, against its
         stop, leaning back over the case).
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

SHELL = (0.10, 0.22, 0.52)            # the case's blue plastic
SHELL_IN = (0.08, 0.16, 0.38)
CARD_EDGE = (0.95, 0.95, 0.93)
CARD_BACK = (0.68, 0.10, 0.12)

# -- the deck ------------------------------------------------------------------------------------
DECK_W = 0.063                        # poker size: 63 x 88 mm
DECK_L = 0.088
DECK_T = 0.017                        # 54 cards
DECK_MASS = 0.095
DECK_POS = (0.300, -0.140)            # lying face down in front of the mouth, its long edges along y

# -- the case (lying on its back) -----------------------------------------------------------------
BX = 0.300                            # the case's centre line (x)
YM = -0.020                           # the mouth's plane
WALL = 0.0015                         # moulded plastic
IN_W = 0.066                          # inside width (x)
IN_D = 0.0205                         # inside depth, now upright (z)
IN_H = 0.0905                         # inside length, mouth to closed end (y)
BOX_MASS = 0.022
Y_END = YM + IN_H + WALL              # outside of the closed end
Z_IN = TOP + WALL                     # the inside floor (the back wall's inner face)
Z_OUT = TOP + 2 * WALL + IN_D         # the upper wall's outer face
FLOOR_FRICTION = (0.22, 0.18)         # the case's plastic back on the bench
INSIDE_FRICTION = (0.45, 0.35)        # cards on the plastic inside

LID_PIVOT = (BX, YM - WALL / 2, Z_OUT)                  # the lid's hinge line, at the upper wall's mouth edge
LID_H = IN_D + 2 * WALL - 0.002                         # hinge line to the lid's free edge (it clears the bench)
LID_W = IN_W + 2 * WALL
LID_MASS = 0.006
LID_BACK = 20.0                       # the hinge stop: the open lid leans back this far past upright
LID_DAMP = 0.00005                    # N m s per radian, as PhysX applies it here
HINGE_BREAK_N = 40.0                  # pulled harder than this, the lid comes off its hinge


def _p(size, offset, **kw):
    return Piece("box", tuple(float(v) for v in size), tuple(float(v) for v in offset), **kw)


def _tiles(size, center, n, axis_split):
    """A slab of ``size`` at ``center`` cut into ``n`` tiles along axis ``axis_split`` (short enough for GPU
    contact): [(size, centre), ...]."""
    out = []
    for i in range(n):
        sz = list(size)
        sz[axis_split] = size[axis_split] / n
        c = list(center)
        c[axis_split] = center[axis_split] - size[axis_split] / 2 + (i + 0.5) * sz[axis_split]
        out.append((tuple(sz), tuple(c)))
    return out


def box_object() -> Part:
    """The case: its back (on the bench), upper wall, two side walls and closed end, one rigid body (its
    root: the first tile of the back)."""
    ow = IN_W + 2 * WALL
    ln = IN_H + WALL
    yc = YM + ln / 2
    floor = _tiles((ow, ln, WALL), (BX, yc, TOP + WALL / 2), 3, 1)
    upper = _tiles((ow, ln, WALL), (BX, yc, Z_OUT - WALL / 2), 3, 1)
    sides = [t for s in (-1, 1) for t in _tiles((WALL, ln, IN_D), (BX + s * (IN_W / 2 + WALL / 2), yc, Z_IN + IN_D / 2), 2, 1)]
    end = [((IN_W, WALL, IN_D), (BX, Y_END - WALL / 2, Z_IN + IN_D / 2))]
    root_size, root_c = floor[0]
    pieces = [_p(sz, np.subtract(c, root_c), color=SHELL, friction=FLOOR_FRICTION) for sz, c in floor[1:]]
    pieces += [_p(sz, np.subtract(c, root_c), color=SHELL, friction=INSIDE_FRICTION) for sz, c in upper + sides + end]
    # the inside floor the cards slide on (its own surface, above the back)
    liner = ((IN_W - 0.001, IN_H - 0.001, 0.0004), (BX, YM + IN_H / 2, Z_IN + 0.0002))
    pieces.append(_p(liner[0], np.subtract(liner[1], root_c), color=SHELL_IN, friction=INSIDE_FRICTION))
    return Part("box", pos=root_c, size=root_size, exact_z=True, pieces=tuple(pieces), mass=BOX_MASS, color=SHELL,
                friction=FLOOR_FRICTION, restitution=0.0, contact_offset=0.0012, max_depenetration=0.3,
                com=(0.0, YM + ln / 2 - root_c[1], Z_IN + IN_D / 2 - root_c[2]))


def lid_object() -> Part:
    """The lid, spawned standing straight up from its hinge; hinged to the case along the upper wall's mouth
    edge. Turned +180 degrees about x from here it hangs shut over the mouth."""
    c0 = (BX - LID_W / 4, LID_PIVOT[1], LID_PIVOT[2] + LID_H / 2)
    other = _p((LID_W / 2, WALL, LID_H), (LID_W / 2, 0.0, 0.0), color=SHELL)
    return Part("lid", pos=c0, size=(LID_W / 2, WALL, LID_H), exact_z=True, pieces=(other,),
                mass=LID_MASS, color=SHELL, friction=INSIDE_FRICTION, restitution=0.0, contact_offset=0.001,
                max_depenetration=0.3,
                hinge=Hinge(pivot=LID_PIVOT, axis="X", damping=LID_DAMP, lower_deg=-LID_BACK, upper_deg=179.0,
                            parent="box", parent_pos=box_object().pos, break_force=HINGE_BREAK_N))


def deck_object() -> Part:
    """The squared deck, lying face down, long edges along y."""
    return Part("deck", pos=(DECK_POS[0], DECK_POS[1], TOP + DECK_T / 2), size=(DECK_W, DECK_L, DECK_T), exact_z=True,
                pieces=(_p((DECK_W - 0.0006, DECK_L - 0.0006, 0.0004), (0.0, 0.0, DECK_T / 2 + 0.0001), color=CARD_BACK,
                           collide=False),),
                mass=DECK_MASS, color=CARD_EDGE, friction=(0.45, 0.35), restitution=0.0, contact_offset=0.0015,
                max_depenetration=0.3)


def object_specs():
    return [box_object(), lid_object(), deck_object()]


OBJECT_NAMES = ("box", "lid", "deck")


def turned(pos, pivot, axis, deg):
    """A spawn position turned ``deg`` about a line through ``pivot`` along a station axis, and that turn
    as a quaternion (x, y, z, w)."""
    a = math.radians(deg)
    r = rot(axis, deg)
    p = np.asarray(pivot) + r @ (np.asarray(pos) - np.asarray(pivot))
    u = np.eye(3)["xyz".index(axis.lower())]
    return tuple(p), (*(u * math.sin(a / 2)), math.cos(a / 2))


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis.upper() == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis.upper() == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def start_poses():
    """Spawn-to-start poses: the lid leaning back against its stop."""
    lid = lid_object()
    return {lid.name: turned(lid.pos, LID_PIVOT, "X", -LID_BACK)}


# -- measurements ---------------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R(pose):
    return quat_matrix(np.asarray(pose, float)[3:7])


def point(pose, local):
    p = np.asarray(pose, float)
    return p[:3] + R(p) @ np.asarray(local, float)


def to_box(box, p):
    """A station-frame point in the box's own frame (origin at its spawn centre, axes as spawned)."""
    b = np.asarray(box, float)
    return R(b).T @ (np.asarray(p, float) - b[:3])


def _rel_angle(parent, child, axis):
    r = R(parent).T @ R(child)
    if axis == "X":
        return math.degrees(math.atan2(r[2, 1], r[1, 1]))
    return math.degrees(math.atan2(r[0, 2], r[0, 0]))


def lid_deg(box, lid):
    """0 hanging shut over the mouth, 180 standing straight up, 200 leaning back against the stop."""
    return 180.0 - _rel_angle(box, lid, "X")


def tilt_deg(pose):
    v = R(pose) @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.acos(float(np.clip(v[2], -1, 1))))


BOX0 = np.asarray(box_object().pos)


def deck_in_box(box, deck):
    """How the deck lies in the case, in the case's frame: (how far its near end is inside the mouth, its
    centre's offsets across the case (x) and up (z) from the inside's centre line, its tilt from the case's
    length in degrees)."""
    c = to_box(box, np.asarray(deck, float)[:3])
    r = R(box).T @ R(deck)
    long_axis = r @ np.array([0.0, 1.0, 0.0])                 # the deck's long edges, spawned along y
    tilt = math.degrees(math.acos(min(1.0, abs(float(long_axis[1])))))
    near = c[1] - DECK_L / 2 * abs(float(long_axis[1]))
    mouth = YM - BOX0[1]
    axis = np.array([BX - BOX0[0], Z_IN + IN_D / 2 - BOX0[2]])
    return near - mouth, np.array([c[0], c[2]]) - axis, tilt


def lid_crease(box):
    """The lid's hinge line (mid-width), as carried by the case."""
    return point(box, np.subtract(LID_PIVOT, box_object().pos))


def torn(box, lid):
    """The lid has come off its hinge: the hinge points carried by the case and by the lid no longer meet."""
    return float(np.linalg.norm(lid_crease(box) - point(lid, np.subtract(LID_PIVOT, lid_object().pos)))) > 0.004
