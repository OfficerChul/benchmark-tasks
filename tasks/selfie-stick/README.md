# selfie-stick

**Instruction:** "Get the selfie stick ready for a group photo: put my phone in its holder and pull the stick
all the way out."

A collapsed selfie stick lies on the bench with a phone beside it. Readying it takes two hands twice over.

- **Drawing it out.** The three telescoping tubes are stiff to slide. A hand pulling the head alone drags the
  whole stick along the bench, so one hand holds the handle while the other draws the head out to the stops.
- **Clamping the phone.** The spring holder is narrower than the phone. One hand holds the spring jaw open by
  its thumb knob while the other lays the phone between the jaws; the jaw is then let back onto the phone's
  edge.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.7475.

| object | type | size / mass | pose |
|---|---|---|---|
| selfie stick | foam handle, three telescoping steel tubes (damped, each with a stop), head with a spring clamp | 315 g; 329 mm collapsed, 629 mm drawn out; handle 28 mm across, 140 mm long; tubes 23, 20, 17 mm across, 100 mm stroke each | lying along y at x 0.30, clamp head at y 0.003, handle end at y 0.332 |
| phone clamp | 26 mm bed, fixed jaw on the handle side, spring jaw at the tip, 1.5 mm lips | jaws 58 mm apart at rest, 11.5 mm tall; spring jaw opens up to 30 mm against 80 N/m (0.8 N at rest, 1.9 N on the phone); 12 mm thumb knob on a 65 mm arm behind the spring jaw | on the head |
| phone | box | 147 x 71.5 x 7.8 mm, 171 g | face up at x 0.42, y 0.15, its length along x |

Both arms work, each seen by its own wrist camera, and the whole scene is in the overhead camera's view.

## Scoring

Extension is the head's travel out of the handle, 0 to 300 mm (the stick's overall length minus 329 mm).

The phone is clamped when all of these hold:
- its centre is within 15 mm of the clamp's centre line across the stick;
- it is within 4 mm of sitting against the fixed jaw along the stick;
- it lies on the clamp bed (within 3 mm) and flat on it (within 8 degrees);
- the spring jaw rests on the phone's edge (within 2 mm).

Levels:
- **0.25** the stick is drawn out to within 10 mm of full length, or the phone is clamped.
- **0.5** both.
- **0.75** also the stick lies on the bench: its axis level within 10 degrees, the handle's underside within
  15 mm of the bench.
- **1.0** also both hands are 6 cm or more from the stick and the phone, held for 1 s.
- **0**, whatever else, if the phone was dropped at any time (came down faster than 0.5 m/s). Otherwise 0.

In a real trial:
- the length is read with a tape measure;
- the phone's seat and the jaw against its edge are checked by eye and with a ruler;
- the video shows the hands clear and no drop.

Run limit: 1,200 control steps (60 s).

## Recorded runs (seed 0)

- **Scripted demo:** 1.0, the trial ending on success at step 767 (38.4 s). The stick was drawn out to
  its stops (629 mm), and the phone was clamped 2 mm from the clamp's centre line, flat on the bed, with the
  spring jaw on its edge. Replaying its recorded trajectory through the scorer (`runtime/rescore.py`) gives
  the same score.
- **Control:** the same demo with the left hand left out: 0.0. The right hand alone drags the whole stick
  along the bench and only draws it out to 238 mm; the phone is never placed.
- **Dream-1** (`dreamlabs/dream-1-beta`): 0.0 (273 steps, 13.7 s). Its left hand reached down toward the
  phone.
- Doing nothing scores 0.

## Scripted demo

1. The left hand takes the handle and the right hand the head's neck. Together they draw the stick out,
   then keep drawing in steps on the measured extension until the tubes are against their stops.
2. Both let go. The right hand takes the spring jaw's knob and pulls the jaw open to 24 mm, re-aiming on the
   measured jaw opening while it holds it.
3. The left hand picks up the phone near its far end and carries it, lined up with the stick, low over the
   fixed jaw and into the gap. It holds it level just over the bed, re-aiming on the measured phone pose.
4. The right hand lets the jaw close to the phone's edge. The left hand lets go and the phone settles on the
   bed; the right hand eases the jaw onto it and lets go of the knob.
5. Both hands lift away.

The demo reads measured poses to find the stick, the knob and the phone, and each arm's measured TCP to trim
out its sag, as a scripted demo may. It never moves an object except through the grippers.

## Simulation model

- **Tubes.** Each tube slides on a damped prismatic joint with a stop at 100 mm. The nested tubes do not
  collide with each other.
- **Clamp.** The spring jaw slides on a sprung prismatic joint with limits.
- **Shapes.** The handle, tubes and neck are 8-sided under round surfaces; the clamp housing is
  flat-bottomed.
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
- `selfie.py`: geometry and measurements.
- `simkit.py`: compound rigid parts and joints.
- `runtime/`: runner.

No third-party assets: every shape is built from primitives in `selfie.py`.
