MuJoCo backend
===============

The built-in backend is a LeRobot ``Robot`` implementation, so MP-Net,
primitives, processors, recording, and actor/learner code do not branch on
simulation. It ships the Google DeepMind MuJoCo Menagerie UR5e and Robotiq
2F-85 models, the ACT/ALOHA peg/socket insertion workpiece, front/side/wrist
cameras, and a six-axis wrist force/torque sensor. Reset loads the free-joint
peg between the fingers and physically closes the 2F-85; opening the gripper
releases it under gravity. The task still starts from this initialized grasp so
it focuses on the insertion stage. Upstream commits and licenses are recorded
in ``src/share/robots/mujoco/assets/README.md``.
The ACT socket keeps its four collision walls and an empty center; no reference
or placeholder geometry occupies the insertion channel.
The controller's ``tool_tcp`` is colocated with Menagerie's existing
``gripper_pinch`` site. The workpiece tip remains a separate ``object_tip`` site
used only for physical insertion-success measurements.
The scene includes overhead and two diagonal side lights. When the interactive
viewer is enabled, it displays the wrist-camera image, rolling three-axis force
and torque plots, and a numeric task-frame wrench overlay. Pass
``--env.viewer_wrist_camera_overlay=false``, ``--env.viewer_wrench_plot=false``,
or ``--env.viewer_wrench_overlay=false`` to hide the corresponding panel.

The six UR arm actuators retain MuJoCo Menagerie's position-servo model.
Task-space ``ee_pos`` commands are converted to joint-position targets with
damped least-squares IK; they are not mapped directly to joint torque. Gravity
compensation approximates the feed-forward behavior of an industrial position
controller. As in the physical UR interface, rotational ``ee_pos`` observations
use a rotation vector, while ``TaskFrame`` configuration continues to use
extrinsic XYZ Euler angles.

The backend scales the stock Menagerie position gains by 16 by default and
scales velocity damping by the square root of that factor. This keeps the held
tool stiff without changing the external position-command interface or reaching
the UR torque limits in the standard insertion motion. Override it with
``position_servo_stiffness_scale`` when using a custom scene or payload.

The SAC action is seven-dimensional: relative Cartesian translation and
rotation followed by gripper position. Its configured min/max statistics map
normalized policy output to ``+/-0.1 m/s``, ``+/-0.5 rad/s``, and ``[0, 1]``.
The Cartesian command is integrated over ``control_dt``, converted by damped
least-squares IK, and only then sent to MuJoCo's joint-position actuators.
Each IK correction is based on and applied to the current measured joint
position; previous servo targets are not accumulated into the next IK result.

Install and preview
-------------------

.. code-block:: bash

   # LeRobot 0.5.x requires Python 3.12 or newer.
   python -m pip install -e '.[mujoco,test]'
   share-mujoco-demo

The demo now runs ``insert -> release -> reset`` automatically. Its keyboard
layout uses arrow keys for XY, left/right Shift for down/up in Z, and
comma/period to close/open the gripper.
Keyboard rotation and letter motion keys are disabled. To inspect the
registered wrist camera instead of the free viewer camera:

.. code-block:: bash

   share-mujoco-demo --viewer-camera=wrist

Offline recording gives the operator 900 control steps (30 seconds at 30 Hz)
by default. Success is measured from the free workpiece's ``object_tip`` site in
the fixture frame, not from the gripper TCP. It requires at least 70 mm insertion
depth, at most 2.0 mm radial lateral error, and peg/socket axis alignment of at least 0.98.
The successful frame is stored with reward 1 and ``done=true``; a time limit
stores reward 0 with ``truncated=true``. Either outcome makes the next recording
iteration perform a full simulation reset and start a new insertion episode.
The default insertion workspace does not treat the workbench as a TCP safety
plane: ``min_tcp_z=-1.2 m`` only supplies a broad numerical bound. ``tool_tcp``
is a mathematical frame and may cross the table plane when geometry permits;
motion is resisted by contacts on the actual gripper, workpiece, fixture, and
workbench geoms. A full reset also clears stale keyboard state and restores the
teleoperator's closed-gripper target before the first action, so an open command
from the previous episode cannot drop the peg.

Use ``--manual`` to disable the automatic downward insertion command while
retaining keyboard Cartesian and gripper control.

Keyboard episode controls are shared by recording, the online actor, and the
interactive demo: ``/`` marks the current episode as a manual failure,
``Enter`` optionally marks success, and ``Esc`` stops the whole process. Offline
recording discards a manually failed episode; the online actor publishes it as
a terminal zero-reward sample so the learner can train from the failure.

The MuJoCo and UR backends also enforce a gripper command safety interval. The
default ``gripper_min_command_interval_s=0.5`` suppresses repeated targets and
holds the previous target when a different command arrives too soon. This final
backend guard applies equally to keyboard and policy commands and persists
across primitive switches.

For a finite headless smoke run:

.. code-block:: bash

   MUJOCO_GL=egl share-mujoco-demo --headless --steps 100

Offline-to-online pipeline
--------------------------

Collect keyboard demonstrations (motion keys are defined by
``KeyboardVelocityTeleopConfig``):

.. code-block:: bash

   share-record \
     --env.type=mujoco_ur5e_insertion \
     --env.viewer=true \
     --env.teleop_mode=keyboard \
     --dataset.repo_id=local/mujoco-insertion \
     --dataset.root=outputs/mujoco/offline-demos \
     --dataset.num_episodes=20 \
     --dataset.single_task='Insert the held peg into the fixture' \
     --dataset.push_to_hub=false \
     --use_policy=false \
     --play_sounds=false

A zero-frame dataset stub left by a failed startup is archived automatically as
``insert.incomplete-<timestamp>``. Existing datasets with saved episodes are
resumed automatically when the same command is restarted after Ctrl-C.

Start the learner. It loads each adaptive primitive's demonstrations from
``<dataset.root>/<primitive-id>`` and then waits for online transitions. With
the default SAC config, optimization starts after 100 online transitions and
each batch mixes online replay with the offline demonstrations:

.. code-block:: bash

   share-learner \
     --env.type=mujoco_ur5e_insertion \
     --dataset.repo_id=local/mujoco-insertion \
     --dataset.root=outputs/mujoco/offline-demos \
     --dataset.video_backend=pyav \
     --output_dir=outputs/mujoco/run \
     --job_name=mujoco-insertion \
     --batch_size=128 \
     --wandb.enable=false

In a second terminal, start the actor. It runs simulation, sends online replay
to the learner, and receives updated actor parameters:

.. code-block:: bash

   share-actor \
     --env.type=mujoco_ur5e_insertion \
     --env.teleop_mode=keyboard \
     --env.viewer=true \
     --dataset.repo_id=local/mujoco-insertion \
     --dataset.root=outputs/mujoco/offline-demos \
     --output_dir=outputs/mujoco/run \
     --job_name=mujoco-insertion \
     --wandb.enable=false

The default transport is ``127.0.0.1:50051``. Set the same
``--env.learner_host`` and ``--env.learner_port`` values on both commands when
running the processes on separate hosts or on a non-default port.
