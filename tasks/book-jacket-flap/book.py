"""Geometry and measurements for tucking a hardback's loose dust-jacket flap back inside its front cover.

On the bench: a cloth-bound hardback standing upright on its tail against a steel bookend, its paper
dust jacket on, except that the jacket's front flap has come out from inside the front cover and sticks
out loose beyond the book's fore-edge.

Station frame: metres, x away from the arm bases, y left, z up; TOP is the bench top. The book stands
with its front cover facing the arms, its spine to the right (-y) and its fore-edge to the left (+y);
the bookend stands behind it. The front cover opens toward the arms about a vertical hinge at the spine.

The layout is written in a book frame (b-coordinates: bx runs from the fore-edge toward the spine, by
from the front cover toward the back cover) and turned into the station frame by ``W``: a quarter turn
clockwise seen from above, then a shift. Every body spawns square to the station axes.

Angles (degrees, from measured body yaws):
  cover   0 shut on the pages, growing as it opens toward the arms;
  fold    the flap's fold at its crease along the cover's fore-edge: 0 flat out, in line with the
          jacket's front panel; 180 turned right round the fore-edge, flat against the inside of the
          cover; negative bent back over the outside.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

CLOTH = (0.20, 0.27, 0.42)            # navy book cloth on the boards
PAGE = (0.94, 0.92, 0.85)
JACKET = (0.86, 0.47, 0.20)           # the jacket's printed outside
JACKET_IN = (0.95, 0.93, 0.88)        # its plain inside (the flap's inner face)
TITLE = (0.97, 0.93, 0.80)
STEEL = (0.62, 0.64, 0.67)

# -- where the book stands ------------------------------------------------------------------------
X_PAGES = 0.270                       # station x of the pages' mid-thickness
Y_SPINE = -0.105                      # station y of the spine (the cover's hinge line)
FRAME_YAW = -90.0                     # book frame to station frame

# -- the book, in book-frame coordinates ----------------------------------------------------------
XS = 0.340                            # spine
YB = 0.060                            # mid-thickness of the pages
BOOK_H = 0.200                        # standing height (the boards' and the jacket's height)
BOARD_W = 0.135                       # spine to fore-edge
BOARD_T = 0.003
PAGES_T = 0.024
SQUARE = 0.006                        # the boards stand this far proud of the page edges at the fore-edge
PAGE_W = BOARD_W - 0.002 - SQUARE
PAGE_H = BOOK_H                       # the text block stands on the shelf with the boards
BOOK_MASS = 0.70                      # text block, spine and back board (the front board is its own body)
BOARD_MASS = 0.060
JKT_T = 0.0012                        # jacket paper, as modelled (see fidelity)
PANEL_MASS = 0.006
FLAP_W = 0.050                        # the flap, crease to free edge
FLAP_MASS = 0.004
SEG = BOOK_H / 5                      # paper sheets are built of five stacked strips (each short enough for GPU contact)

Y_IN = YB - PAGES_T / 2 - 0.0005      # inside face of the front board, shut (0.5 mm off the pages)
Y_MID = Y_IN - BOARD_T / 2            # the board's mid-plane: the cover hinge line lies in it
Y_OUT = Y_IN - BOARD_T                # outside face of the front board
Y_PANEL = Y_OUT - JKT_T / 2           # the jacket's front panel, lying on the board
PIVOT_B = (XS, Y_MID)                 # cover and jacket hinge line (vertical), at the spine
X_FORE = XS - BOARD_W                 # the board's fore-edge, shut
CREASE_GAP = 0.0035                   # the jacket's fold stands this far beyond the board's fore-edge
CREASE_B = (X_FORE - CREASE_GAP, Y_MID)  # the flap's crease line (vertical), with the cover shut
CREASE_R = Y_MID - Y_PANEL            # the flap's mid-plane stands this far out from its crease line

# Paper creases, modelled as elastic hinges (see fidelity). A flap that has come out has had its crease
# flattened: it springs back out when let go, so it has to be held turned in until the closing cover
# traps it. The cover's hinge is stiff (damped) and stays where it is left, as a hardback's does.
FLAP_REST = -5.0                      # the flattened crease relaxes a little past flat, bent back
FLAP_K = 0.000008                     # spring rate toward FLAP_REST (N m per degree)
COVER_DAMPING = 0.002                 # the cover stays where it is left (N m s per degree); no spring
START_OPEN = 28.0                     # the cover stands this far ajar at the start (deg)
FLAP_TEAR_N = 30.0                     # a flap pulled harder than this tears off at its crease

# -- bookend (fixed), behind the back cover -------------------------------------------------------
END_T = 0.002
END_W = 0.110
END_H = 0.150
END_Y = YB + PAGES_T / 2 + 0.0005 + BOARD_T + JKT_T + END_T / 2 + 0.0002
END_X = XS - 0.070


# -- book frame to station frame ------------------------------------------------------------------

def W(bx, by, z=None):
    """A book-frame point in the station frame."""
    p = (X_PAGES + (by - YB), Y_SPINE - (bx - XS))
    return p if z is None else (p[0], p[1], z)


def Wv(v):
    """A book-frame vector (or body offset) in the station frame."""
    return (float(v[1]), float(-v[0]), float(v[2])) if len(v) == 3 else (float(v[1]), float(-v[0]))


def Ws(size):
    return (float(size[1]), float(size[0]), float(size[2]))


PIVOT = W(*PIVOT_B)                   # station frame
CREASE = W(*CREASE_B)


def _box(name, bsize, bcenter, z, pieces=(), **kw):
    return Part(name, pos=W(bcenter[0], bcenter[1], z), size=Ws(bsize), exact_z=True, pieces=tuple(pieces), **kw)


def _piece(bsize, boffset, **kw):
    return Piece("box", Ws(bsize), Wv(boffset), **kw)


def book_object() -> Part:
    """Text block, back board, spine and the jacket's back half, as one body (its root: the pages)."""
    zc = TOP + BOOK_H / 2
    cx = XS - 0.002 - PAGE_W / 2
    y_back = YB + PAGES_T / 2 + 0.0005 + BOARD_T / 2
    pieces = (
        _piece((BOARD_W, BOARD_T, BOOK_H), (XS - BOARD_W / 2 - cx, y_back - YB, 0.0), color=CLOTH),
        _piece((0.005, PAGES_T + 2 * BOARD_T + 0.001, BOOK_H), (XS + 0.0025 - cx, 0.0010, 0.0), color=CLOTH),
        # the jacket's back panel and spine, wrapped on (its back flap is tucked inside the back board)
        _piece((BOARD_W + 0.001, JKT_T, BOOK_H), (XS - BOARD_W / 2 - cx, y_back + BOARD_T / 2 + JKT_T / 2 - YB, 0.0),
               color=JACKET),
        _piece((JKT_T, PAGES_T + 2 * BOARD_T + 2 * JKT_T + 0.001, BOOK_H), (XS + 0.0056 - cx, 0.0010, 0.0),
               color=JACKET, collide=False),
        _piece((JKT_T + 0.0004, 0.016, 0.110), (XS + 0.0058 - cx, 0.0010, 0.020), color=TITLE, collide=False),
    )
    return _box("book", (PAGE_W, PAGES_T, PAGE_H), (cx, YB), zc, pieces, mass=BOOK_MASS, color=PAGE,
                friction=(0.7, 0.6), restitution=0.0, contact_offset=0.002, com=(0.0, 0.0, -0.010))


def board_object() -> Part:
    """The front board with the jacket's front panel on its outside (the panel moves with the board: the
    jacket is held on by its wrap round the spine and its back flap tucked inside the back board)."""
    z = TOP + BOOK_H / 2
    bc = (XS - BOARD_W / 2, Y_MID)
    x0, x1 = CREASE_B[0], XS + 0.0030
    panel = tuple(_piece((x1 - x0, JKT_T, SEG), ((x0 + x1) / 2 - bc[0], Y_PANEL - bc[1], k * SEG), color=JACKET,
                         friction=(0.6, 0.5)) for k in (-2, -1, 0, 1, 2))
    title = _piece((0.090, 0.0004, 0.040), ((x0 + x1) / 2 + 0.006 - bc[0], Y_PANEL - JKT_T / 2 - 0.0002 - bc[1], 0.045),
                   color=TITLE, collide=False)
    return _box("board", (BOARD_W, BOARD_T, BOOK_H), bc, z, (*panel, title), mass=BOARD_MASS + PANEL_MASS, color=CLOTH,
                friction=(0.7, 0.6), restitution=0.0, contact_offset=0.0015, max_depenetration=0.3,
                hinge=Hinge(pivot=(PIVOT[0], PIVOT[1], z), axis="Z", damping=COVER_DAMPING, lower_deg=0.0, upper_deg=125.0,
                            parent="book", parent_pos=book_object().pos))


def opened(pos, deg=START_OPEN):
    """A spawn position turned ``deg`` about the cover's hinge line (station frame), and that turn as a
    quaternion (x, y, z, w): where the cover and its flap stand at the start."""
    c = PIVOT
    a = math.radians(deg)
    dx, dy = pos[0] - c[0], pos[1] - c[1]
    p = (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a), pos[2])
    return p, (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def flap_object() -> Part:
    """The jacket's front flap, spawned flat out (fold 0); its crease springs it to its rest at once."""
    z = TOP + BOOK_H / 2
    inner = _piece((FLAP_W - 0.004, 0.0004, BOOK_H - 0.004), (0.0, JKT_T / 2 + 0.0002, 0.0), color=JACKET_IN,
                   collide=False)
    rest = tuple(_piece((FLAP_W, JKT_T, SEG), (0.0, 0.0, k * SEG), color=JACKET) for k in (-2, -1, 1, 2))
    return _box("flap", (FLAP_W, JKT_T, SEG), (CREASE_B[0] - FLAP_W / 2, Y_PANEL), z, (*rest, inner), mass=FLAP_MASS,
                color=JACKET, friction=(0.9, 0.8), restitution=0.0, contact_offset=0.001, max_depenetration=0.3,
                hinge=Hinge(pivot=(CREASE[0], CREASE[1], z), axis="Z", damping=0.00002, lower_deg=-178.0, upper_deg=60.0,
                            stiffness=FLAP_K, target_deg=-FLAP_REST, parent="board", parent_pos=board_object().pos,
                            break_force=FLAP_TEAR_N))


def bookend_object() -> Part:
    base = _piece((END_W, 0.090, 0.0015), (0.0, 0.045 - END_T / 2, -END_H / 2 + 0.00075), color=STEEL)
    return _box("bookend", (END_W, END_T, END_H), (END_X, END_Y), TOP + END_H / 2, (base,), color=STEEL,
                friction=(0.5, 0.4))


# -- measurements -------------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_deg(pose):
    r = quat_matrix(np.asarray(pose, float)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(pose):
    v = quat_matrix(np.asarray(pose, float)[3:7]) @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.acos(float(np.clip(v[2], -1, 1))))


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def cover_deg(book, board):
    return wrap(yaw_deg(board) - yaw_deg(book))


def fold_deg(board, flap):
    """Inward fold of the flap at its crease; 180 lies flat against the inside of the cover. Folds past
    180 read as such (up to 190) rather than wrapping to -180."""
    a = -wrap(yaw_deg(flap) - yaw_deg(board))
    return a + 360.0 if a < -170.0 else a


def point(pose, local):
    p = np.asarray(pose, float)
    return p[:3] + quat_matrix(p[3:7]) @ np.asarray(local, float)


def crease_on_board(board):
    """Where the crease line runs, as carried by the cover (station frame, at mid-height)."""
    b0 = board_object().pos
    return point(board, (CREASE[0] - b0[0], CREASE[1] - b0[1], 0.0))


def crease_on_flap(flap):
    f0 = flap_object().pos
    return point(flap, (CREASE[0] - f0[0], CREASE[1] - f0[1], 0.0))


def torn(board, flap):
    """The flap's crease has come apart (its hinge broke): the two crease lines no longer meet."""
    return float(np.linalg.norm(crease_on_board(board) - crease_on_flap(flap))) > 0.005


def flap_edge(flap, s=FLAP_W, z=0.0):
    """A point on the flap's mid-plane ``s`` from its crease toward its free edge, ``z`` from mid-height."""
    return point(flap, Wv((FLAP_W / 2 - s, 0.0, z)))


def flap_dir(flap):
    """Unit vector in the flap's plane from its crease toward its free edge (horizontal)."""
    return quat_matrix(np.asarray(flap, float)[3:7]) @ np.asarray(Wv((-1.0, 0.0, 0.0)))


def outline():
    """Corners (lo, hi) of a box round the shut book, its flap and the bookend (station frame)."""
    a = W(X_FORE - FLAP_W - 0.005, Y_OUT - JKT_T - 0.005)
    b = W(XS + 0.010, END_Y + 0.003)
    lo = np.array([min(a[0], b[0]), min(a[1], b[1]), TOP])
    hi = np.array([max(a[0], b[0]), max(a[1], b[1]), TOP + BOOK_H])
    return lo, hi
