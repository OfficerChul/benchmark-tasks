# card-deck-case

**Instruction:** "Put the cards back in their case and close it."

A squared deck of poker-size playing cards lies face down on the bench in front of its hard plastic case,
which lies on its back with its mouth toward the deck and its lid standing open. The case is light and
its back slides easily on the bench, while the cards slide on its inside with more friction: a push on
the deck shoves an unheld case along with it, and the deck never goes in. So it takes both hands:

1. one hand holds the case's closed end so the case cannot slide away;
2. the other picks the deck up, lines it up with the mouth, slides it in, lets go and pushes it home
   with its fingertips until its end is flush with the mouth;
3. the lid is tipped over; it falls shut over the mouth under its own weight, which it can only do once
   the deck is all the way in.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.7475.

| object | type | size / mass | pose |
|---|---|---|---|
| deck | free, rigid | 54 poker-size cards, 63 x 88 x 17 mm, 95 g | face down, long edges along y, centred at (0.300, -0.140) |
| case | free, hard plastic shell | walls 1.5 mm; inside 66 wide x 90.5 long x 20.5 deep; 22 g | on its back, centre line x 0.300, mouth at y -0.020 facing the deck, closed end at y 0.072 |
| lid | hinged to the case along the mouth edge of its upper face | 69 x 21.5 x 1.5 mm, 6 g | open, leaning back over the case 20 degrees past upright against its stop |

Friction: the case's back on the bench about 0.2; the cards on the case's inside about 0.45.

Both arms work, each seen by its own wrist camera, and the whole scene is in the overhead camera's view.

## Scoring

Measured in the case's frame. The deck is *in* when its near end is no more than 1 mm outside the mouth,
it lies within 10 degrees of square to the case, and its centre is inside the case's walls.

- **0.25** the deck is in.
- **0.5** also the lid is shut, hanging within 6 degrees of closed over the mouth.
- **0.75** also the case still lies on its back on the bench (tilted no more than 15 degrees, not
  lifted).
- **1.0** also both hands are 6 cm or more outside the case, held for 1 s.
- **0** if the lid has come off its hinge at any time. Otherwise 0.

In a real trial:
- the deck's end is checked flush with the mouth with a rule laid across the mouth;
- the lid is checked down over the mouth;
- the video shows the hands clear.

Run limit: 1,200 control steps (60 s).

## Recorded runs (seed 0)

- **Scripted demo:** 1.0, the trial ending on success at step 460 (23.0 s). The deck's end went 0.9 mm
  inside the mouth; the lid fell shut to 2.5 degrees; the case stayed where it lay. Replaying the recorded
  trajectory through the scorer (`runtime/rescore.py`) gives the same score.
- **Control:** the same demo with the left hand kept off the case. Pushed, the deck carried the case
  15 cm along the bench and never went in: 0.0.
- **Dream-1** (`dreamlabs/dream-1-beta`): 0.0 (321 steps, 16.1 s); it did not reach the deck.
- Doing nothing scores 0.

## Scripted demo

The left hand's closed fingertips come down against the case's closed end and stay there.

The right hand grips the deck across its width near its near end, lifts it, lines it up with the mouth
and slides it in until its fingers come up to the mouth, steering the deck by its measured place. It
lets go, comes down behind the deck with its fingertips leaned toward the case, and pushes the deck home
until its end is flush with the mouth, then backs away.

The left hand then lifts off the case, comes down behind the open lid and tips it past upright; the lid
falls shut over the mouth. Both hands stand clear.

The demo reads measured poses to find the deck, the case and the lid, and each arm's measured TCP to trim
out its sag, as a scripted demo may. It never moves an object except through the grippers.

## Simulation model

- **The deck** is one rigid block, 63 x 88 x 17 mm.
- **The case** is a rigid shell of 1.5 mm plastic walls, each wall built of tiles short enough for PhysX
  to collide with them on the GPU. Its back has its own bench friction (about 0.2) and its inside its own
  card friction (about 0.45).
- **The lid** turns on a free hinge with a stop 20 degrees past upright and falls shut under its own
  weight. Pulled harder than 40 N it comes off the hinge.
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

Seed 0. At reset the lid is leaned back against its stop, then the scene settles for 30 ticks before
control starts. One trial takes about 1.5 minutes of wall time on one GPU. Each episode writes:
- four 20 fps camera videos;
- setup and final PNGs;
- the measured state trajectory;
- `result.json`.

The scorer reads only measured body poses and the robot's measured TCPs.

## Files

- `task.py`: scene, start pose, scripted demo and its two-arm controller.
- `scoring.py`: scorer.
- `cards.py`: geometry and measurements.
- `simkit.py`: compound rigid parts and joints.
- `runtime/`: runner.

No third-party assets: every shape is built from primitives in `cards.py`.
