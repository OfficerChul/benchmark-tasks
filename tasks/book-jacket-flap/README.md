# book-jacket-flap

**Instruction:** "The dust jacket's front flap has come out of this book. Tuck it back inside the front
cover and close the book."

A cloth-bound hardback stands upright against a bookend with its front cover ajar. Its paper dust jacket
is on, but the jacket's front flap has come out from inside the cover and sticks out beyond the
fore-edge, its crease flattened. Turned back round the fore-edge, the flap springs straight back out as
soon as it is let go. So it has to be held inside the cover while the cover is swung in over it, and let
go only once the closing cover traps it against the page edges.

It takes both hands at once:

1. the right hand takes the cover by its top edge, near the spine, and holds it;
2. the left hand's fingertips push the loose flap round the cover's fore-edge until it lies flat against
   the inside of the cover, and keep it there;
3. the right hand swings the cover in over the flap, lets go, and presses the cover on from outside
   while the left hand still holds the flap in;
4. once the cover is in far enough to trap the flap against the page edges (about 15 degrees), the left
   hand lets the flap go and lifts out, and the right hand presses the book shut.

Let the flap go any earlier and it springs back out, and the cover closes on it from outside.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.7475.

| object | type | size / mass | pose |
|---|---|---|---|
| hardback | free | 200 mm tall; boards 135 x 200 x 3 mm; text block 24 mm thick, its fore-edge 6 mm in from the boards' edge; 0.70 kg + 66 g front board | upright on its tail, front cover facing the arms, spine to the right: cover hinge at (0.256, -0.105); pages' mid-thickness at x 0.270 |
| front cover | hinged at the spine, damped, no spring (stays where it is left) | 135 x 200 x 3 mm, with the jacket's front panel on it | ajar 28 degrees; its fore-edge at (0.193, 0.014) |
| jacket front flap | hinged at its crease, 3.5 mm beyond the cover's fore-edge | 50 x 200 mm, 4 g | straight out in line with the cover; crease flattened, it rests bent 5 degrees outward |
| bookend | fixed, steel | plate 110 x 150 x 2 mm on a 90 mm base | behind the back cover |

Both arms work, each seen by its own wrist camera, and the whole scene is in the overhead camera's view.

## Scoring

Fold is the flap's angle at its crease: 0 straight out in line with the cover, 180 turned right round the
fore-edge, flat against the inside of the cover. The cover's angle is 0 shut on the pages.

- **0.25** the flap has been turned in round the fore-edge to 150 degrees or more at some time.
- **0.5** the cover is shut (within 6 degrees of the pages, so its fore-edge is 14 mm or less from them)
  with the flap folded 165 degrees or more inside it, between the cover and the pages.
- **0.75** also the book stayed where it stood: it never moved more than 20 mm, tilted more than
  10 degrees or turned more than 10 degrees on the bench.
- **1.0** also both hands are 6 cm or more outside a box round the shut book, its flap and the bookend,
  held for 1 s.
- **0**, whatever else, if the book fell over (tilted more than 30 degrees) or the flap was torn off at
  its crease at any time. Otherwise 0.

In a real trial:
- the fore-edge gap is measured with a rule;
- the flap is checked from above to lie inside the cover, not sticking out;
- the book's place is marked on paper round its foot;
- the video shows the hands clear.

Run limit: 1,200 control steps (60 s).

## Recorded runs (seed 0)

- **Scripted demo:** 1.0, the trial ending on success at step 386 (19.3 s). The flap was turned in to
  175 degrees and held there. The cover was pressed in to 13 degrees before the flap was let go: the flap
  sprang back to 154 degrees, caught against the page edges, and was pressed in to 173 degrees as the
  cover shut. The cover ended 4.5 degrees from the pages; the book moved 1 mm and tilted 0.6 degrees.
  Replaying its recorded trajectory through the scorer (`runtime/rescore.py`) gives the same score.
- **Control:** the same demo with the flap let go as soon as it has been turned in (the cover still
  28 degrees open), before the cover is swung in. The flap springs back out from 173 to 33 degrees within
  about 5 s, the cover closes on it from outside, and the run ends at 0.25.
- **Dream-1** (`dreamlabs/dream-1-beta`): 0.0 (993 steps, 49.7 s). It swung the cover part-way shut
  (to 9.5 degrees) with the flap still outside.
- Doing nothing scores 0.

## Scripted demo

The right hand comes down on the cover's top edge 5 cm out from the spine, its wrist leaning back to the
right, and pinches the cover with the jacket on it.

The left hand's closed fingertips (leaned 0.3 rad) come down just outside the loose flap, 35 mm out from
its crease. They swing round the crease in three moves, pushing the flap round the fore-edge to
75 degrees, 130 degrees and 175 degrees (flat inside the cover), and keep it there.

The right hand swings the cover in to 20 degrees while the left hand's fingertips follow, then lets go.
It comes down with closed fingertips on the outside of the cover and presses the cover in (to
13 degrees in the recorded run), the left hand still following. The left hand then lifts out, the right
hand presses the cover shut, holds it 1 s and lifts away.

The demo reads measured poses to find the flap, its crease, the cover and its hinge, and each arm's
measured TCP to trim out its sag, as a scripted demo may. It never moves an object except through the
grippers.

## Simulation model

- **Paper.** The jacket is 1.2 mm sheet, built of welded 40 mm strips; the flap turns at its crease.
- **The crease** is a damped hinge with a weak spring back toward its flattened rest, 5 degrees bent
  outward. It tears at 30 N and stops at 178 degrees, where the flap lies flat inside the cover.
- **The jacket's front panel** moves with the front cover, held on by its wrap round the spine and its
  back flap tucked inside the back board. The back half of the jacket is part of the book.
- **The cover's joint** is damped, with no spring: a hardback's cover stays where it is left.
- The text block, spine and back board are one rigid body standing free on the bench.
- PhysX at 240 Hz, control at 20 Hz.

## Reproduction

`runtime/` holds the exported runner. From this task directory, with the `yam-inspect-isaac:6.1.0`
image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
mkdir -p /tmp/rc-task-home/tmp ./outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$PWD/outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --livestream 0 --viz none
python runtime/rescore.py --task task.py outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect outputs/TASK/RUN/episode-01/result.json
```

`--policy dream` runs Dream-1 instead. It needs `DREAM_API_KEY` and a budget ledger mounted at
`/dream-budget`, made once with `runtime/dream_budget.py init`.

On an arm64 host the base image has no WebRTC streaming library: drop that `COPY`/`RUN` pair from
`runtime/Dockerfile`, and run the container as the image's own user instead of `--user`.

Seed 0. At reset the cover and its flap are stood 28 degrees ajar, then the scene settles for 30 ticks
before control starts. One trial takes about 1.5 minutes of wall time on one GPU. Each episode writes:
- four 20 fps camera videos;
- setup and final PNGs;
- the measured state trajectory;
- `result.json`.

The scorer reads only measured body poses and the robot's measured TCPs.

## Files

- `task.py`: scene, start pose, scripted demo and its two-arm controller.
- `scoring.py`: scorer.
- `book.py`: geometry and measurements.
- `simkit.py`: compound rigid parts and joints.
- `runtime/`: runner.

No third-party assets: every shape is built from primitives in `book.py`.
