# trekking-pole

**Instruction:** "Set my trekking pole to 120 cm: open the flip lock, pull the lower section out to the 120
mark, and shut the lock again."

A two-section trekking pole lies on the bench at 110 cm with its flip lock shut.

- **The lock.** The lever is flipped open by its thumb tab before the section will move, and shut again at
  the mark.
- **The section.** The lower section slides stiffly. A hand pulling it alone drags the whole pole along the
  bench, so one hand holds the grip while the other draws the section out to the 120 cm mark.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.7475.

| object | type | size / mass | pose |
|---|---|---|---|
| trekking pole | two aluminium sections, the lower one sliding stiffly (damped) between 100 and 125 cm | 268 g; 110 cm long at the start; foam grip 30 mm across, 130 mm long; blue upper shaft 18 mm; collar 30 mm across, 35 mm long, 60 to 63.5 cm from the grip top; lower section 15 mm with a rubber tip | along y at x 0.34, grip top at y 0.57, tip at y -0.53, resting on its grip and collar |
| length marks | printed rings on the lower section | every 5 cm, 100 to 125 cm (120 cm in red), read at the collar's edge | |
| flip-lock lever | orange plate hinged on the collar (damped) | 45 mm long, 14 mm tall, raised thumb tab 8 mm square, 30 mm tall; swings 95 degrees out | shut along the collar on its side facing the arms, free end toward the tip |

Both arms work, each seen by its own wrist camera, and the whole scene is in the overhead camera's view.

## Scoring

The pole's length is from the grip top to the tip.

- **0.25** the length is within 3 cm of 120 cm.
- **0.5** within 5 mm.
- **0.75** also the flip lock is shut (its lever within 10 degrees of shut).
- **1.0** also the pole lies on the bench (its axis level within 10 degrees, the grip's underside within
  15 mm of the bench) and both hands are 6 cm or more from the pole and its lever, held for 1 s.
- **0**, whatever else, if at any time the pole was dropped (came down faster than 0.5 m/s) or the lock was
  forced (the length changed by more than 5 mm while the lever was shut). Otherwise 0.

In a real trial:
- the length is read with a tape measure, the lever by eye;
- the video shows the lever opened before the section moves, the hands clear and no drop.

Run limit: 1,200 control steps (60 s).

## Recorded runs (seed 0)

- **Scripted demo:** 1.0, the trial ending on success at step 579 (29.0 s). The lock was flipped open, the
  lower section drawn out from 110.0 to 120.2 cm with the grip held, and the lock flipped shut. Replaying its
  recorded trajectory through the scorer (`runtime/rescore.py`) gives the same score.
- **Control:** the same demo with the left hand left out: 0.25. The right hand alone drags the whole pole
  20 cm along the bench and draws the section out only to 119.2 cm.
- **Dream-1** (`dreamlabs/dream-1-beta`): 0.0 (945 steps, 47.3 s). Its right hand went to the lock and
  turned the lever 13 degrees; the section did not move.
- Doing nothing scores 0.

## Scripted demo

1. The right hand pinches the lever's thumb tab and swings it open about the hinge, re-aiming on the
   measured lever angle. The left hand takes the grip.
2. The right hand takes the lower section and draws it out, then keeps drawing in steps on the measured
   length until it reads 120 cm, while the left hand holds the grip.
3. The right hand pinches the tab again and swings the lever shut.
4. Both hands let go and lift away.

The demo reads measured poses to find the pole, the lever and the section, and each arm's measured TCP to
trim out its sag, as a scripted demo may. It never moves an object except through the grippers.

## Simulation model

- **Section.** The lower section slides on a damped prismatic joint with stops at 100 and 125 cm.
- **Lock.** The lever turns on a damped hinge with stops. The lock's clamping force is not modelled: the
  scorer treats the length changing by more than 5 mm while the lever is shut as forcing the lock.
- **Shapes.** The grip, shaft, collar and lower section are 8-sided under round surfaces.
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
- `trekking.py`: geometry and measurements.
- `simkit.py`: compound rigid parts and joints.
- `runtime/`: runner.

No third-party assets: every shape is built from primitives in `trekking.py`.
