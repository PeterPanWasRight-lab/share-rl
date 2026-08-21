# MuJoCo assets

The robot and workpiece assets in this directory come from maintained open-source
robotics repositories. They are vendored so the default environment is reproducible
and does not download models at runtime.

- `menagerie/universal_robots_ur5e`: Google DeepMind MuJoCo Menagerie, commit
  `da76818e269b82289eba39808e2fb91d679d6994`; derived from ROS-Industrial and
  distributed under its included BSD-3-Clause license.
- `menagerie/robotiq_2f85`: the Menagerie Robotiq 2F-85 from the same commit,
  distributed under its included ROS-Industrial BSD license.
- `act/bimanual_viperx_insertion.xml`: the original ACT/ALOHA insertion task at
  commit `742c753c0d4a5d87076c8f69e5628c79a8cc5488`, MIT licensed. `act/peg.xml`
  extracts its peg verbatim for composition with the UR5e; `scene.xml` uses its
  socket dimensions and contact parameters verbatim.

`share.robots.mujoco.model` composes these models with MuJoCo's `MjSpec` API. It
adds only the mounting transforms, cameras, TCP site, and force/torque sensors.
The UR and gripper mesh, kinematic, inertial, collision, and linkage definitions
remain in the upstream files. The peg has a free joint. Episode reset places it
between the fingers and closes the real 2F-85 linkage, so opening the gripper
releases the workpiece under gravity.

Upstream sources:

- https://github.com/google-deepmind/mujoco_menagerie
- https://github.com/tonyzhaozh/act
