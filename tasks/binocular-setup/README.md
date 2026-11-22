# binocular-setup

**Instruction:** "Set these binoculars for me: 62 mm between the eyepieces, and the focus wheel on the
5 m mark."

A pair of roof-prism binoculars lies on the bench on its two barrels, opened out to 72.7 mm. Two settings
make them ready for someone else.

- **Eye spacing.** The barrels turn on one stiff centre hinge. Folding them in takes a hand on each
  barrel: one hand alone only lifts its own barrel, and the other swings in to the stop.
- **Focus.** The stiff centre focus wheel is then rolled round a third of a turn to the 5 m mark with a
  fingertip, stroke by stroke, while the other hand holds the binoculars still.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.7475.

| object | type | size / mass | pose |
|---|---|---|---|
| binoculars | two barrels on a centre hinge (along x), damped | 10x42 roof prism, 650 g; barrels 43 mm across, 120 mm long, eyepiece cups 38 mm across, 22 mm long | lying on both barrels, eyepieces toward the arms (cups x 0.218 to 0.24, barrels to x 0.36), either side of y 0; 72.7 mm between the barrel axes; hinge travel 56 to 74 mm |
| focus wheel | on the hinge axis, damped | 22 mm across, 16 mm wide | x 0.26; its white line at the top, on the infinity mark; the 5 m mark a third of a turn round (the wheel's top rolling toward +y) |

Both arms work, each seen by its own wrist camera, and the whole scene is in the overhead camera's view.

## Scoring

Eye spacing is the distance between the two barrels' axes, which is the distance between the eyepiece
centres.

- **0.25** the eye spacing is within 1.5 mm of 62 mm, or the wheel's line is within 15 degrees of the
  5 m mark.
- **0.5** both.
- **0.75** also the binoculars lie on the bench on both barrels: each barrel level within 20 degrees, its
  underside within 15 mm of the bench.
- **1.0** also both hands are 6 cm or more from the binoculars, held for 1 s.
- **0**, whatever else, if the binoculars were dropped at any time (a barrel came down faster than
  0.5 m/s). Otherwise 0.

In a real trial:
- the spacing is read with calipers between the eyepiece centres;
- the wheel's line is read against the marks from above;
- the video shows the hands clear and no drop.

Run limit: 1,200 control steps (60 s).

## Recorded runs (seed 0)

- **Scripted demo:** 1.0, the trial ending on success at step 489 (24.5 s). The barrels were folded from
  72.7 mm to 62.0 mm in the air and set down, and the wheel was rolled to within 5 degrees of the 5 m mark.
  Replaying its recorded trajectory through the scorer (`runtime/rescore.py`) gives the same score.
- **Control:** the same demo with the left hand left out: 0.0. The right hand alone lifts its barrel, the
  left barrel swings in to the hinge's 56 mm stop.
- **Dream-1** (`dreamlabs/dream-1-beta`): 0.0 (993 steps, 49.7 s). It reached down to the binoculars but did
  not fold them.
- Doing nothing scores 0.

## Scripted demo

1. The left hand takes the left barrel by its objective end, and the right hand takes the right barrel
   by its eyepiece cup: both clear of the bridge.
2. Both lift the binoculars 3 cm and turn their barrels in opposite ways about the hinge axis. The
   spacing is then closed in steps on the measured spacing, since the wrists give a little under the
   load.
3. They lower the binoculars straight down onto the bench and let go.
4. The left hand holds the left barrel down near its objective end.
5. The right hand's fingertip sweeps the wheel's rim down on its +y side, stroke by stroke, each stroke
   sized on what is left to the mark. It stops once the wheel is on the mark.
6. Both hands lift away.

The demo reads measured poses to find the barrels, the hinge and the wheel, and each arm's measured TCP to
trim out its sag, as a scripted demo may. It never moves an object except through the grippers.

## Simulation model

- **Hinge and wheel.** Both are damped revolute joints on the binoculars' axis.
- **Barrels.** The rubber armour is a 16-sided section under its round surface; the barrels lie on it on
  the bench.
- **Physics.** Rigid bodies, PhysX at 240 Hz, control at 20 Hz.

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

Seed 0. The scene settles for 30 ticks before control starts. One trial takes about 3 minutes of wall
time on one GPU. Each episode writes:
- four 20 fps camera videos;
- setup and final PNGs;
- the measured state trajectory;
- `result.json`.

The scorer reads only measured body poses and velocities and the robot's measured TCPs.

## Files

- `task.py`: scene, scripted demo and its two-arm controller.
- `scoring.py`: scorer.
- `bino.py`: geometry and measurements.
- `simkit.py`: compound rigid parts and joints.
- `runtime/`: runner.

No third-party assets: every shape is built from primitives in `bino.py`.
