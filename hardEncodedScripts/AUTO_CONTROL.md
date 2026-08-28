# MuJoCo online autoControl

`auto_control.py` imitates intermittent human keyboard intervention while the normal
Actor and Learner are running. It reads exact MuJoCo peg/socket geometry, plans one
small correction, and injects the same `left/right/up/down/shift` tokens used by the
keyboard teleoperator. The existing HIL processor still decides action override and
sets `rl.is_intervention`; the script does not write replay buffers or contact the
Learner directly.

## Start order

1. Start the online Learner normally.
2. Start autoControl so it is already waiting for the Actor truth socket.
3. Start the Actor with `--env.teleop_mode=keyboard`:

```bash
python hardEncodedScripts/auto_control.py --intervention-interval-s=0.5 --step-m=0.005 --actor-fps=30 --episodes=100
```

Starting autoControl before Actor is important when the Learner still needs to
decode a large offline video dataset. Otherwise Actor can finish several episodes
before synthetic intervention begins, and those early episodes contain no
`rl.is_intervention` samples. The controller safely waits while Actor is absent.

`Ctrl+C` stops autoControl and clears any pending synthetic key pulse. It does not
stop Actor or Learner. `--episodes=0` runs continuously.

## Main parameters

| Parameter | Meaning |
| --- | --- |
| `--intervention-interval-s` | Wall-clock interval between planned interventions. |
| `--step-m` | Maximum Euclidean Cartesian correction planned per intervention. |
| `--actor-fps` | Actor control frequency; must match `--env.fps`. |
| `--teleop-speed-m-s` | Full-scale keyboard speed; default matches MuJoCo (`0.1 m/s`). |
| `--clearance-m` | Peg-tip fixture-X coordinate used as the safe alignment plane. |
| `--align-tolerance-m` | Required lateral accuracy before entering the hole. |
| `--retreat-lateral-error-m` | In-hole lateral error that triggers retreat. |
| `--target-depth-m` | Planned insertion depth. |
| `--episodes` | Number of physically successful episodes, or `0` for unlimited. |
| `--dry-run` | Read truth and print plans without injecting interventions. |

The planner retreats to a safe plane when badly misaligned, aligns at the plane,
then inserts with lateral closed-loop correction. It does not press `Enter`;
completion must satisfy the environment's physical depth, lateral-error, and
axis-alignment test.

## Intentional core hooks

Two narrow, local-only hooks are required because autoControl is a separate process:

- `src/share/robots/mujoco/mujoco_robot.py` sends read-only truth to
  `/tmp/share_mujoco_god_view.sock`, but only while that receiver path exists.
  With autoControl stopped it does no JSON serialization and changes no dynamics.
- `KeyboardVelocityTeleop` listens on `/tmp/share_keyboard_teleop.sock` for local
  one-control-cycle key pulses. Physical keyboard events are unchanged. The socket
  is mode `0600`, accepts no robot targets, and feeds the normal HIL processor.

Both are Unix-domain datagrams and are not reachable over the network. No Actor,
Learner, policy, replay-buffer, or dataset code was modified. For custom paths, set
`SHARE_MUJOCO_GOD_VIEW_SOCKET` and `SHARE_KEYBOARD_TELEOP_SOCKET` before Actor, then
pass matching `--socket-path` and `--keyboard-socket-path` to autoControl.

An Actor launched before these code changes were loaded must be restarted once.
After that, autoControl may be started and stopped without restarting Actor/Learner.

## Experiment interpretation

These samples are marked as interventions, but they are oracle-guided synthetic
interventions rather than human demonstrations. Keep that distinction in experiment
names and reports so results are not mistaken for human-HIL performance.

The 2026-08-26 staged-SAC run verified the data path with 934 online transitions:
192 were marked interventions and 4 of 6 complete episodes succeeded before policy
collapse. This validates the injection/labeling path, but does not validate plain SAC
actor updates: after unfreezing, both the 400- and 500-update checkpoints scored 0/10.
