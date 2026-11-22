"""Geometry and measurements for readying a selfie stick: phone in its spring clamp, stick out to full length.

Station frame: metres, x away from the arm bases, y left, z up; TOP is the bench top.

A collapsed selfie stick lies on the bench along y, its foam handle toward +y (the left arm) and its
clamp head toward -y (the right arm). Three telescoping tubes slide out of the handle, each stiff in the
one before it (damped), out to its stop. The clamp sits on the head: a fixed jaw on the handle side and a
spring jaw at the tip, 58 mm apart at rest, each with a lip over the phone's edge. The spring jaw is
opened by a thumb knob on a short arm behind it. A phone lies flat on the bench beside the stick.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece, Spring

TOP = 0.7475

FOAM = (0.10, 0.10, 0.11)              # black foam handle
CHROME = (0.70, 0.71, 0.73)            # steel tubes
PLASTIC = (0.16, 0.16, 0.17)           # head and clamp
PAD = (0.42, 0.43, 0.45)               # grey rubber pads on the jaws
PHONE = (0.18, 0.19, 0.22)
SCREEN = (0.05, 0.07, 0.12)
ACCENT = (0.85, 0.30, 0.20)            # the red button on the handle

X0 = 0.300                             # the stick's axis, along y
# The stick is laid out below in its own frame (u along the stick, from the handle toward the head), then
# turned half round: the handle lies toward +y (the left arm) and the head toward -y (the right arm).
YAW = math.pi
YC2 = 0.102                            # world y = YC2 - u
RH = 0.014                             # handle and head radius (across flats, 8-sided)
AXIS_Z = TOP + RH
LH = 0.140                             # handle length
Y_MOUTH = -0.090                       # the handle's open end (the tubes come out here)
TUBE_R = (0.0115, 0.0100, 0.0085)
LT = 0.130                             # each tube's length
COLLAR = 0.012                         # how far each tube stands out of the one before, collapsed
TRAVEL = 0.100                         # each tube's stroke
FULL = 3 * TRAVEL                      # the stick's whole extension
TUBE_DAMP = 600.0                      # N s/m: each tube is stiff in the one before it

NECK = 0.085                           # head: a plain grip neck, then the clamp
BED_L = 0.068                          # clamp bed length (along y)
BED_W = 0.026                          # clamp width (along x)
HOUSING_W = 0.036                      # the clamp housing under the bed
BED_T = 0.004
JAW_T = 0.004                          # jaw plate thickness (along y)
JAW_H = 0.0115                         # jaw face height above the bed, up to its lip
LIP = 0.0015                           # how far each lip reaches over the phone
LIP_T = 0.0020
GAP0 = 0.058                           # between the jaws at rest
JAW_TRAVEL = 0.030
JAW_K = 80.0                           # N/m
JAW_PRELOAD = 0.010                    # the spring is preloaded this far at rest

TAB_R, TAB_H = 0.006, 0.030           # the round thumb knob behind the spring jaw (pulled to open it)
TAB_DY = JAW_T / 2 + 0.0200            # ... its axis behind the jaw plate's centre
TAB_DX = 0.065                         # ... and off to the side (toward the arms once the stick is laid down)
TAB_TOP_REL = -(JAW_H + LIP_T) / 2 + TAB_H   # ... its top above the jaw's centre

PHONE_L, PHONE_W, PHONE_T = 0.147, 0.0715, 0.0078
PHONE_MASS = 0.171
PHONE_POS = (0.420, 0.150)             # where it lies at the start, face up, its length along x

BED_Z = AXIS_Z + RH                    # clamp bed's underside (on top of the head)
BED_TOP = BED_Z + BED_T


def tube_y_end(k):
    """Collapsed: tube k's outer end (k = 0, 1, 2)."""
    return Y_MOUTH + COLLAR * (k + 1)


def tube_centre(k):
    return (X0, tube_y_end(k) - LT / 2, AXIS_Z)


HEAD_E = tube_y_end(2)                 # the last tube's outer end: the head starts here
FIXED_FACE = HEAD_E + NECK + JAW_T     # the fixed jaw's inner face (collapsed)
SPRING_FACE = FIXED_FACE + GAP0        # the spring jaw's inner face at rest
SEAT_Y = FIXED_FACE + PHONE_W / 2      # the phone's centre, clamped against the fixed jaw
GRIP_NECK_Y = HEAD_E + 0.062           # where a hand takes the head (clear of the clamp)
GRIP_HANDLE_Y = Y_MOUTH - LH / 2       # the handle's middle

SEAT_REL = np.array([0.0, SEAT_Y - tube_centre(2)[1], BED_TOP + PHONE_T / 2 - AXIS_Z])   # phone, seated, in the last tube's frame
JAW_SEATED = PHONE_W - GAP0            # spring jaw travel with the phone in (13.5 mm)


def _w(c):
    """A point of the stick's own layout (x, u, z) in the station frame."""
    return (c[0], YC2 - c[1], c[2])


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


STICK = ("handle", "tube1", "tube2", "tube3", "jaw")


def _others(name):
    return tuple(n for n in STICK if n != name)


def handle() -> Part:
    c = (X0, Y_MOUTH - LH / 2, AXIS_Z)
    side = 2 * RH * math.tan(math.pi / 8)
    pieces = [*_octagon(RH, LH, (0.0, 0.0, 0.0), FOAM)[1:],
              _p("cylinder", (0.0042, 0.0016, 0.0042), (0.0, LH / 2 - 0.030, RH + 0.0002), axis="Z", color=ACCENT,
                 collide=False),
              _p("cylinder", (RH + 0.0012, 0.006, RH + 0.0012), (0.0, LH / 2 - 0.003, 0.0), axis="Y", color=PLASTIC,
                 collide=False),
              _p("cylinder", (RH + 0.0012, 0.006, RH + 0.0012), (0.0, -LH / 2 + 0.003, 0.0), axis="Y", color=PLASTIC,
                 collide=False)]
    return Part("handle", pos=_w(c), yaw=YAW, size=(side, LH, 2 * RH), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=0.20, color=FOAM, friction=(1.8, 1.6), restitution=0.0, contact_offset=0.002,
                no_sleep=True, no_collide=_others("handle"))


def tube(k) -> Part:
    r = TUBE_R[k]
    c = tube_centre(k)
    side = 2 * r * math.tan(math.pi / 8)
    pieces = list(_octagon(r, LT, (0.0, 0.0, 0.0), CHROME)[1:])
    pieces.append(_p("cylinder", (r / math.cos(math.pi / 8) + 0.0008, 0.004, r / math.cos(math.pi / 8) + 0.0008),
                     (0.0, LT / 2 - 0.002, 0.0), axis="Y", color=PLASTIC, collide=False))
    parent = "handle" if k == 0 else f"tube{k}"
    parent_pos = _w((X0, Y_MOUTH - LH / 2, AXIS_Z) if k == 0 else tube_centre(k - 1))
    if k == 2:
        pieces += _head(c)
    name = f"tube{k + 1}"
    return Part(name, pos=_w(c), yaw=YAW, size=(side, LT, 2 * r), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=(0.025, 0.020, 0.060)[k], color=CHROME, friction=((0.9, 0.8), (0.9, 0.8), (1.8, 1.6))[k], restitution=0.0,
                contact_offset=0.002, no_sleep=True, no_collide=_others(name),
                spring=Spring(axis="Y", lower=0.0, upper=TRAVEL, stiffness=0.0, damping=TUBE_DAMP, target=0.0,
                              parent=parent, parent_pos=parent_pos))


def _head(c):
    """The head on the last tube's end, in that tube's frame: a grip neck, then the clamp housing with
    its bed and the fixed jaw."""
    y0 = HEAD_E - c[1]
    out = list(_octagon(RH, NECK, (0.0, y0 + NECK / 2, 0.0), PLASTIC))
    bed_c = y0 + NECK + BED_L / 2
    # the clamp housing: flat-bottomed and wider than the stick, so the head lies steady with a phone in it
    out.append(_p("box", (HOUSING_W, BED_L, 2 * RH), (0.0, bed_c, 0.0), color=PLASTIC))
    out.append(_p("box", (BED_W, BED_L, BED_T), (0.0, bed_c, RH + BED_T / 2), color=PLASTIC))
    out.append(_p("box", (BED_W - 0.004, BED_L - 0.004, 0.0008), (0.0, bed_c, RH + BED_T + 0.0002), color=PAD,
                  collide=False))
    fy = FIXED_FACE - c[1]
    zb = RH + BED_T
    out.append(_p("box", (BED_W, JAW_T, JAW_H + LIP_T), (0.0, fy - JAW_T / 2, zb + (JAW_H + LIP_T) / 2), color=PLASTIC))
    out.append(_p("box", (BED_W, LIP, LIP_T), (0.0, fy + LIP / 2, zb + JAW_H + LIP_T / 2), color=PLASTIC))
    out.append(_p("box", (BED_W - 0.002, 0.0008, JAW_H - 0.001), (0.0, fy + 0.0004, zb + JAW_H / 2), color=PAD,
                  collide=False))
    return out


def jaw() -> Part:
    """The spring jaw: a plate with its lip, sliding along y on the head, sprung toward the fixed jaw."""
    t3 = tube_centre(2)
    zb = BED_TOP
    c = (X0, SPRING_FACE + JAW_T / 2, zb + (JAW_H + LIP_T) / 2)
    pieces = [
        _p("box", (BED_W, LIP, LIP_T), (0.0, -JAW_T / 2 - LIP / 2, (JAW_H + LIP_T) / 2 - LIP_T / 2), color=PLASTIC),
        _p("box", (BED_W - 0.002, 0.0008, JAW_H - 0.001), (0.0, -JAW_T / 2 - 0.0004, -LIP_T / 2), color=PAD,
           collide=False),
        _p("box", (BED_W, 0.010, 0.003), (0.0, JAW_T / 2 + 0.005, -(JAW_H + LIP_T) / 2 + 0.0015), color=PLASTIC),
        _p("box", (0.008, TAB_DY + 0.004, 0.004), (TAB_DX, (TAB_DY - JAW_T / 2) / 2 + JAW_T / 2, -(JAW_H + LIP_T) / 2 + 0.002),
           color=PLASTIC),
        _p("box", (TAB_DX + 0.004, 0.008, 0.004), (TAB_DX / 2, JAW_T / 2 + 0.004, -(JAW_H + LIP_T) / 2 + 0.002), color=PLASTIC),
        *[_p("box", (2 * TAB_R * math.tan(math.pi / 8), 2 * TAB_R, TAB_H), (TAB_DX, TAB_DY, -(JAW_H + LIP_T) / 2 + TAB_H / 2),
             rot=(0.0, 0.0, k * math.pi / 4), color=PLASTIC) for k in range(4)],
        _p("cylinder", (TAB_R / math.cos(math.pi / 8), TAB_H, TAB_R / math.cos(math.pi / 8)),
           (TAB_DX, TAB_DY, -(JAW_H + LIP_T) / 2 + TAB_H / 2), axis="Z", color=PAD, collide=False),
    ]
    return Part("jaw", pos=_w(c), yaw=YAW, size=(BED_W, JAW_T, JAW_H + LIP_T), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=0.010, color=PLASTIC, friction=(0.9, 0.8), restitution=0.0, contact_offset=0.0015,
                no_sleep=True, no_collide=_others("jaw"),
                spring=Spring(axis="Y", lower=0.0, upper=JAW_TRAVEL, stiffness=JAW_K, damping=2.0,
                              target=-JAW_PRELOAD, parent="tube3", parent_pos=_w(t3)))


def phone() -> Part:
    c = (PHONE_POS[0], PHONE_POS[1], TOP + PHONE_T / 2)
    pieces = [
        _p("box", (PHONE_L - 0.006, PHONE_W - 0.005, 0.0006), (0.0, 0.0, PHONE_T / 2 + 0.0001), color=SCREEN,
           collide=False),
        _p("cylinder", (0.0045, 0.0008, 0.0045), (-PHONE_L / 2 + 0.016, PHONE_W / 2 - 0.016, -PHONE_T / 2 - 0.0002),
           axis="Z", color=(0.05, 0.05, 0.05), collide=False),
    ]
    return Part("phone", pos=c, size=(PHONE_L, PHONE_W, PHONE_T), shape="box", exact_z=True, pieces=tuple(pieces),
                mass=PHONE_MASS, color=PHONE, friction=(0.6, 0.5), restitution=0.0, contact_offset=0.0015,
                no_sleep=True)


def object_specs():
    return [handle(), tube(0), tube(1), tube(2), jaw(), phone()]


OBJECT_NAMES = ("handle", "tube1", "tube2", "tube3", "jaw", "phone")


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


def _along(a, b, d0):
    """How far body ``b`` stands out along body ``a``'s y axis beyond the spawn offset ``d0``."""
    pa, pb = np.asarray(a, float), np.asarray(b, float)
    return float(np.dot(pb[:3] - pa[:3], R_(pa)[:, 1])) - d0


D_T3 = tube_centre(2)[1] - (Y_MOUTH - LH / 2)
D_JAW = (SPRING_FACE + JAW_T / 2) - tube_centre(2)[1]


def extension(h, t3):
    """How far the stick is pulled out (m): the head's travel from the handle, 0 to 0.300."""
    return _along(h, t3, D_T3)


def jaw_open(t3, j):
    """The spring jaw's travel from rest (m)."""
    return _along(t3, j, D_JAW)


def length(h, t3):
    """The stick's overall length, handle end to head tip (m)."""
    return (HEAD_E + NECK + BED_L) - (Y_MOUTH - LH) + extension(h, t3)


def phone_rel(t3, ph):
    """The phone's centre in the head's frame, from where it sits clamped (m)."""
    p3, pp = np.asarray(t3, float), np.asarray(ph, float)
    return R_(p3).T @ (pp[:3] - p3[:3]) - SEAT_REL


def phone_tilt(t3, ph):
    """Angle between the phone's face and the clamp bed (deg)."""
    c = float(np.dot(R_(t3)[:, 2], R_(ph)[:, 2]))
    return math.degrees(math.acos(min(1.0, abs(c))))


def axis_tilt(h):
    """How far the stick's axis is out of the horizontal (deg)."""
    v = R_(h)[:, 1]
    return math.degrees(math.asin(min(1.0, abs(float(v[2])))))
