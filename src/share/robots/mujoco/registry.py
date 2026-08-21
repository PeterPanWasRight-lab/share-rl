from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mujoco_robot import MujocoRobot


_ROBOTS: weakref.WeakValueDictionary[str, MujocoRobot] = weakref.WeakValueDictionary()


def register_robot(robot_id: str, robot: MujocoRobot) -> None:
    if robot_id in _ROBOTS and _ROBOTS[robot_id] is not robot:
        raise RuntimeError(f"A connected MuJoCo robot already uses id {robot_id!r}.")
    _ROBOTS[robot_id] = robot


def unregister_robot(robot_id: str, robot: MujocoRobot) -> None:
    if _ROBOTS.get(robot_id) is robot:
        del _ROBOTS[robot_id]


def get_robot(robot_id: str) -> MujocoRobot:
    try:
        return _ROBOTS[robot_id]
    except KeyError as exc:
        raise RuntimeError(
            f"MuJoCo camera could not find connected robot id {robot_id!r}. "
            "MP-Net must connect the robot before its cameras."
        ) from exc

