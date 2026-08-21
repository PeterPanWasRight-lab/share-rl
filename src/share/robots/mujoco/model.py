from __future__ import annotations

from pathlib import Path


ASSET_ROOT = Path(__file__).parent / "assets"
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
ACT_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"


def build_ur5e_2f85_model():
    """Compose unmodified Menagerie models with the ACT insertion workpiece."""
    import mujoco

    scene = mujoco.MjSpec.from_file(str(ASSET_ROOT / "scene.xml"))
    ur5e = mujoco.MjSpec.from_file(
        str(ASSET_ROOT / "menagerie" / "universal_robots_ur5e" / "ur5e.xml")
    )
    gripper = mujoco.MjSpec.from_file(
        str(ASSET_ROOT / "menagerie" / "robotiq_2f85" / "2f85.xml")
    )
    peg = mujoco.MjSpec.from_file(str(ASSET_ROOT / "act" / "peg.xml"))

    ur5e.compiler.conflict = mujoco.mjtConflict.mjCONFLICT_MERGE
    ur5e.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    ur5e.option.impratio = 10.0

    attachment_site = ur5e.site("attachment_site")
    gripper_frame = attachment_site.parent.add_frame()
    gripper_frame.pos = attachment_site.pos
    gripper_frame.quat = attachment_site.quat
    gripper_frame.attach_body(gripper.body("base_mount"), prefix="gripper_")

    gripper_base = ur5e.body("gripper_base")
    tool_tcp = gripper_base.add_site()
    tool_tcp.name = "tool_tcp"
    tool_tcp.pos = [0.0, 0.0, 0.255]
    tool_tcp.size = [0.003, 0.0, 0.0]

    wrist_camera = gripper_base.add_camera()
    wrist_camera.name = "wrist"
    wrist_camera.pos = [0.08, 0.0, 0.07]
    wrist_camera.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    wrist_camera.targetbody = "fixture"
    wrist_camera.fovy = 58.0

    robot_frame = scene.worldbody.add_frame()
    robot_frame.attach_body(ur5e.body("base"), prefix="")

    # ACT's peg is a free body. Its reset pose places one end between the open
    # 2F-85 pads; MujocoRobot.reset_simulation closes the real linkage around it.
    peg_body = peg.body("peg")
    peg_body.pos = [-0.13399703, 0.49200009, 0.28220037]
    peg_body.quat = [-0.49999908, -0.50000275, -0.49999908, 0.49999908]
    peg_frame = scene.worldbody.add_frame()
    peg_frame.attach_body(peg_body, prefix="object_")

    for name, sensor_type in (
        ("tcp_force", mujoco.mjtSensor.mjSENS_FORCE),
        ("tcp_torque", mujoco.mjtSensor.mjSENS_TORQUE),
    ):
        sensor = scene.add_sensor()
        sensor.name = name
        sensor.type = sensor_type
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = "attachment_site"

    return scene.compile()


__all__ = ["ACT_COMMIT", "ASSET_ROOT", "MENAGERIE_COMMIT", "build_ur5e_2f85_model"]
