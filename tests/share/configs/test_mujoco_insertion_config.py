import pytest
from lerobot.configs.types import FeatureType, NormalizationMode

from share.configs.mujoco_insertion import MujocoInsertionEnvConfig
from share.configs.rl import _ENV_DERIVED_POLICY_ATTRS
from share.envs.manipulation_primitive.task_frame import PolicyMode


def test_state_only_policy_excludes_both_cameras():
    cfg = MujocoInsertionEnvConfig(viewer=False, state_only_policy=True)

    assert cfg.policy_excluded_image_keys == (
        "observation.images.front",
        "observation.images.wrist",
    )


def test_insertion_policy_learns_xyz_and_locks_entry_orientation():
    cfg = MujocoInsertionEnvConfig(viewer=False)
    primitive = cfg.primitives["insert"]
    frame = primitive.task_frame

    assert frame.policy_mode == [PolicyMode.RELATIVE] * 3 + [None] * 3
    assert frame.policy_action_dim == 3
    assert primitive.delta == [0.0] * 6
    assert primitive.policy.dataset_stats["action"] == {
        "min": [-0.1, -0.1, -0.1],
        "max": [0.1, 0.1, 0.1],
    }


def test_insertion_state_machine_owns_gripper_commands():
    cfg = MujocoInsertionEnvConfig(viewer=False, teleop_mode="keyboard")

    assert cfg.primitives["reset"].processor.gripper.enable is False
    assert cfg.primitives["reset"].processor.gripper.static_pos == 1.0
    assert cfg.primitives["insert"].processor.gripper.enable is False
    assert cfg.primitives["insert"].processor.gripper.static_pos == 1.0
    assert cfg.primitives["release"].processor.gripper.enable is False
    assert cfg.primitives["release"].processor.gripper.static_pos == 0.0
    assert cfg.primitives["done"].processor.gripper.static_pos is None
    assert cfg.teleop["main"].gripper_enabled is False

    timeout = next(
        transition
        for transition in cfg.transitions
        if transition.source == "insert" and transition.reason == "insertion_timeout"
    )
    assert timeout.target == "done"


def test_mujoco_actor_enables_calibrated_force_backoff():
    cfg = MujocoInsertionEnvConfig(viewer=False)

    assert cfg.force_backoff.enabled
    assert cfg.force_backoff.force_thresholds_n == [20.0, 20.0, 20.0]
    assert cfg.force_backoff.wrench_to_backoff_sign == [-1.0, -1.0, -1.0]


def test_visual_policy_matches_pretrained_resnet_contract():
    cfg = MujocoInsertionEnvConfig(viewer=False)

    assert cfg.policy_image_size == 64
    assert cfg.primitives["insert"].processor.image_preprocessing.resize_size == (64, 64)
    assert (cfg.cameras["front"].width, cfg.cameras["front"].height) == (64, 64)
    assert (cfg.cameras["wrist"].width, cfg.cameras["wrist"].height) == (64, 64)
    policy = cfg.primitives["insert"].policy
    assert policy.normalization_mapping[FeatureType.VISUAL] is NormalizationMode.MEAN_STD
    assert policy.dataset_stats["observation.images.front"]["mean"] == [0.485, 0.456, 0.406]
    assert policy.dataset_stats["observation.images.front"]["std"] == [0.229, 0.224, 0.225]
    assert policy.vision_encoder_name == "helper2424/resnet10"
    assert policy.freeze_vision_encoder is True
    assert policy.pretrained_vision_input_size == 128
    assert policy.proprio_latent_dim == 64
    assert policy.bc_random_crop_padding == 4
    assert cfg.primitives["insert"].processor.observation.stack_frames == 1



def test_conservative_sac_settings_are_forwarded_to_policy():
    cfg = MujocoInsertionEnvConfig(
        policy_actor_update_after=500,
        policy_sac_bc_loss_weight=2.0,
        policy_freeze_shared_encoder_during_sac=True,
    )

    policy = cfg.primitives["insert"].policy
    assert policy.actor_update_after == 500
    assert policy.sac_bc_loss_weight == 2.0
    assert policy.freeze_shared_encoder_during_sac is True


def test_visual_policy_can_fall_back_to_trainable_random_cnn():
    cfg = MujocoInsertionEnvConfig(policy_vision_encoder_name=None)

    assert cfg.primitives["insert"].policy.vision_encoder_name is None
    assert cfg.primitives["insert"].policy.freeze_vision_encoder is False


def test_domain_randomization_profile_is_forwarded_to_robot():
    cfg = MujocoInsertionEnvConfig(
        domain_randomization=True,
        fixture_xy_randomization_m=0.01,
    )

    assert cfg.robot["main"].randomize_fixture_xy == pytest.approx(0.01)
    assert cfg.robot["main"].randomize_fixture_z == pytest.approx(0.001)
    assert cfg.robot["main"].randomize_fixture_yaw_deg == pytest.approx(3.0)
    assert cfg.robot["main"].randomize_camera_position_m == pytest.approx(0.005)
    assert cfg.robot["main"].randomize_camera_rotation_deg == pytest.approx(1.5)
    assert cfg.robot["main"].randomize_light_intensity_fraction == pytest.approx(0.20)
    assert cfg.robot["main"].randomize_contact_friction_fraction == pytest.approx(0.15)


def test_policy_rejects_empty_frame_stack():
    with pytest.raises(ValueError, match="at least 1"):
        MujocoInsertionEnvConfig(policy_frame_stack=0)


def test_visual_policy_rejects_tiny_images():
    with pytest.raises(ValueError, match="at least 32"):
        MujocoInsertionEnvConfig(policy_image_size=16)


def test_top_level_policy_override_preserves_env_normalization_mapping():
    assert "normalization_mapping" in _ENV_DERIVED_POLICY_ATTRS
    assert "vision_encoder_name" in _ENV_DERIVED_POLICY_ATTRS
