from __future__ import annotations

from types import SimpleNamespace
import queue

import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.rl.buffer import ReplayBuffer
from lerobot.transport.utils import transitions_to_bytes

try:
    from lerobot.datasets.utils import dataset_to_policy_features
except ImportError:  # Older LeRobot versions do not expose this helper.
    dataset_to_policy_features = None

from share.scripts import learner_server
from share.scripts.learner_server import (
    initialize_offline_replay_buffers,
    initialize_replay_buffers,
    process_transitions,
    save_training_checkpoint,
)


def _policy_stub() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            online_buffer_capacity=8,
            offline_buffer_capacity=8,
            input_features={"observation.state": None},
        )
    )


def _add_demo_transitions(buffer: ReplayBuffer, count: int = 3) -> None:
    for index in range(count):
        state = {"observation.state": torch.tensor([[float(index), float(index + 1)]])}
        next_state = {"observation.state": torch.tensor([[float(index + 1), float(index + 2)]])}
        buffer.add(
            state=state,
            action=torch.tensor([[0.1, 0.2]]),
            reward=1.0,
            next_state=next_state,
            done=index == count - 1,
            truncated=False,
            complementary_info={"is_intervention": 1.0},
        )


def _transition(index: int, *, is_intervention: bool, done: bool = False) -> dict:
    return {
        "id": "insert",
        "state": {"observation.state": torch.tensor([[float(index), 0.0]])},
        "action": torch.tensor([[float(index)]], dtype=torch.float32),
        "reward": 1.0,
        "next_state": {"observation.state": torch.tensor([[float(index + 1), 0.0]])},
        "done": done,
        "truncated": False,
        "complementary_info": {"is_intervention": is_intervention},
    }


def test_learner_replay_buffers_resume_from_checkpoint_layout(tmp_path):
    """Offline buffer is reconstructed from the on-policy dataset on resume."""
    policy = _policy_stub()
    saved_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )
    _add_demo_transitions(saved_buffer)  # all 3 transitions have is_intervention=1.0

    primitive_root = tmp_path / "out" / "insert"
    # Save only the online dataset — no separate dataset-offline
    learner_server._save_replay_buffer_to_lerobot_dataset(
        saved_buffer,
        repo_id="repo-insert",
        fps=10,
        root=str(primitive_root / "dataset"),
        task_name="insert",
    )

    cfg = SimpleNamespace(
        resume=True,
        output_dir=tmp_path / "out",
        dataset=SimpleNamespace(repo_id="repo", root=None),
        env=SimpleNamespace(task="task"),
    )

    online_buffers = initialize_replay_buffers(
        cfg=cfg,
        policies={"insert": policy},
        device="cpu",
        storage_device="cpu",
    )
    offline_buffers = initialize_offline_replay_buffers(
        cfg=cfg,
        policies={"insert": policy},
        device="cpu",
        storage_device="cpu",
    )

    assert len(online_buffers["insert"]) == len(saved_buffer)
    # All 3 transitions are interventions, so offline buffer should have 3 entries
    assert len(offline_buffers["insert"]) == len(saved_buffer)


def test_process_transitions_stores_interventions_in_offline_buffer():
    """Intervention transitions land in offline buffer with done/next_state preserved as-is."""
    policy = _policy_stub()
    online_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )
    offline_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=False,
    )

    # Transitions: intv, intv, policy, intv, intv(done)
    transitions = [
        _transition(0, is_intervention=True),
        _transition(1, is_intervention=True),
        _transition(2, is_intervention=False),
        _transition(3, is_intervention=True),
        _transition(4, is_intervention=True, done=True),
    ]
    transition_queue = queue.Queue()
    transition_queue.put(transitions_to_bytes(transitions))

    process_transitions(
        transition_queue=transition_queue,
        replay_buffers={"insert": online_buffer},
        offline_replay_buffers={"insert": offline_buffer},
        device="cpu",
        shutdown_event=SimpleNamespace(is_set=lambda: False),
    )

    assert len(online_buffer) == 5
    assert len(offline_buffer) == 4  # 4 intervention transitions

    # done is preserved exactly — no forced True at segment boundaries
    assert torch.equal(
        offline_buffer.dones[:4].cpu(),
        torch.tensor([False, False, False, True]),
    )
    assert torch.equal(
        offline_buffer.truncateds[:4].cpu(),
        torch.tensor([False, False, False, False]),
    )

    # next_state at the last step of segment 1 (index 1) must be the real next frame (index=2)
    assert torch.equal(
        offline_buffer.next_states["observation.state"][1].cpu(),
        torch.tensor([2.0, 0.0]),
    )


def test_process_transitions_moves_data_to_replay_storage_device(monkeypatch):
    policy = _policy_stub()
    online_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )

    transition_queue = queue.Queue()
    transition_queue.put(transitions_to_bytes([_transition(0, is_intervention=False)]))

    seen_devices: list[str] = []

    def _record_move(*, transition, device):
        seen_devices.append(str(device))
        return transition

    monkeypatch.setattr(learner_server, "move_transition_to_device", _record_move)

    process_transitions(
        transition_queue=transition_queue,
        replay_buffers={"insert": online_buffer},
        offline_replay_buffers={"insert": None},
        device="cuda",
        shutdown_event=SimpleNamespace(is_set=lambda: False),
    )

    assert seen_devices == ["cpu"]
    assert len(online_buffer) == 1


def test_checkpoint_safe_runtime_config_clears_and_restores_shm_manager():
    robot_cfg = SimpleNamespace(shm_manager=object())
    cfg = SimpleNamespace(env=SimpleNamespace(robot={"main": robot_cfg}))
    original = robot_cfg.shm_manager

    with learner_server._checkpoint_safe_runtime_config(cfg):
        assert robot_cfg.shm_manager is None

    assert robot_cfg.shm_manager is original


def test_replay_buffer_export_writes_visual_feature_names_for_act_loading(tmp_path):
    buffer = ReplayBuffer(
        capacity=4,
        device="cpu",
        storage_device="cpu",
        state_keys=["observation.images.front", "observation.state"],
        optimize_memory=False,
    )

    for index in range(2):
        image = torch.full((1, 3, 4, 5), fill_value=float(index), dtype=torch.float32)
        state = {
            "observation.images.front": image,
            "observation.state": torch.tensor([[float(index), float(index + 1)]], dtype=torch.float32),
        }
        next_state = {
            "observation.images.front": image + 1.0,
            "observation.state": torch.tensor([[float(index + 1), float(index + 2)]], dtype=torch.float32),
        }
        buffer.add(
            state=state,
            action=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            reward=1.0,
            next_state=next_state,
            done=index == 1,
            truncated=False,
            complementary_info={"is_intervention": 1.0},
        )

    root = tmp_path / "dataset-offline"
    learner_server._save_replay_buffer_to_lerobot_dataset(
        buffer,
        repo_id="repo-insert",
        fps=10,
        root=str(root),
        task_name="insert",
    )

    dataset = LeRobotDataset(repo_id="repo-insert", root=str(root))
    assert dataset.meta.features["observation.images.front"]["names"] == ["channels", "height", "width"]
    assert dataset.meta.features["action"]["names"] is None

    if dataset_to_policy_features is None:
        pytest.skip("dataset_to_policy_features is unavailable in this LeRobot version")

    policy_features = dataset_to_policy_features(dataset.meta.features)
    assert policy_features["observation.images.front"].shape == (3, 4, 5)


def test_replay_buffer_export_works_without_image_writer_api(monkeypatch):
    buffer = ReplayBuffer(
        capacity=4,
        device="cpu",
        storage_device="cpu",
        state_keys=["observation.state"],
        optimize_memory=False,
    )
    _add_demo_transitions(buffer, count=2)

    saved_episodes: list[int] = []

    class _FakeDataset:
        def add_frame(self, frame: dict) -> None:
            pass

        def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
            saved_episodes.append(1)

        def finalize(self) -> None:
            pass

    monkeypatch.setattr(learner_server.LeRobotDataset, "create", lambda **kwargs: _FakeDataset())

    learner_server._replay_buffer_to_lerobot_dataset(
        buffer,
        repo_id="repo-insert",
        fps=10,
        root="/tmp/fake-dataset",
        task_name="insert",
    )

    assert saved_episodes == [1]


def test_offline_buffer_checkpoint_roundtrip_multi_intervention_edge_cases(tmp_path):
    """
    Full roundtrip: online buffer → lerobot dataset → initialize_offline_replay_buffers.

    Episode A (ep=1): [policy, intv, intv, policy, intv, policy(done)]
      Two intervention segments; episode ends on a policy step.
      - intv@step1: next_state must be (1,2)  [within segment]
      - intv@step2: next_state must be (1,3)  [crosses into policy — not a dummy]
      - intv@step4: next_state must be (1,5)  [crosses into terminal policy — not a dummy]

    Episode B (ep=2): [policy, intv, intv(done)]
      One intervention block; episode ends on an intervention.
      - intv@step1: next_state must be (2,2)
      - intv@step2: done=True, next_state is dummy (fine — masked by (1-done) in TD target)
    """
    policy = _policy_stub()

    online_buffer = ReplayBuffer(
        capacity=16,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )

    def _add(ep: int, step: int, *, is_intv: bool, done: bool = False) -> None:
        online_buffer.add(
            state={"observation.state": torch.tensor([[float(ep), float(step)]])},
            action=torch.tensor([[float(step), 0.0]]),
            reward=float(step),
            next_state={"observation.state": torch.tensor([[float(ep), float(step + 1)]])},
            done=done,
            truncated=False,
            complementary_info={"is_intervention": float(is_intv)},
        )

    # Episode A: [policy, intv, intv, policy, intv, policy(done=True)]
    _add(1, 0, is_intv=False)
    _add(1, 1, is_intv=True)
    _add(1, 2, is_intv=True)
    _add(1, 3, is_intv=False)
    _add(1, 4, is_intv=True)
    _add(1, 5, is_intv=False, done=True)

    # Episode B: [policy, intv, intv(done=True)]
    _add(2, 0, is_intv=False)
    _add(2, 1, is_intv=True)
    _add(2, 2, is_intv=True, done=True)

    learner_server._save_replay_buffer_to_lerobot_dataset(
        online_buffer,
        repo_id="repo-insert",
        fps=10,
        root=str(tmp_path / "out" / "insert" / "dataset"),
        task_name="insert",
    )

    cfg = SimpleNamespace(
        resume=True,
        output_dir=tmp_path / "out",
        dataset=SimpleNamespace(repo_id="repo", root=None),
        env=SimpleNamespace(task="task"),
    )

    offline_buffers = initialize_offline_replay_buffers(
        cfg=cfg,
        policies={"insert": policy},
        device="cpu",
        storage_device="cpu",
    )

    offline = offline_buffers["insert"]
    assert len(offline) == 5  # (1,1),(1,2),(1,4) from A + (2,1),(2,2) from B

    # Build lookup: (ep, step) → {done, next, is_intervention}
    transitions: dict[tuple[int, int], dict] = {}
    for i in range(len(offline)):
        sv = offline.states["observation.state"][i].cpu()
        nv = offline.next_states["observation.state"][i].cpu()
        key = (int(sv[0].item()), int(sv[1].item()))
        transitions[key] = {
            "done": bool(offline.dones[i].item()),
            "next": (float(nv[0].item()), float(nv[1].item())),
            "is_intervention": bool(offline.complementary_info["is_intervention"][i].bool().item()),
        }

    assert set(transitions.keys()) == {(1, 1), (1, 2), (1, 4), (2, 1), (2, 2)}

    for key, t in transitions.items():
        assert t["is_intervention"], f"Transition {key} missing is_intervention flag"

    # Episode A — segment 1 (single step): next within segment
    assert not transitions[(1, 1)]["done"]
    assert transitions[(1, 1)]["next"] == pytest.approx((1.0, 2.0))

    # Episode A — segment 1 boundary: next crosses into policy (critical — old code gave dummy)
    assert not transitions[(1, 2)]["done"]
    assert transitions[(1, 2)]["next"] == pytest.approx((1.0, 3.0))

    # Episode A — segment 2 boundary: next crosses into terminal policy (critical — old code gave dummy)
    assert not transitions[(1, 4)]["done"]
    assert transitions[(1, 4)]["next"] == pytest.approx((1.0, 5.0))

    # Episode B — mid-segment: next within segment
    assert not transitions[(2, 1)]["done"]
    assert transitions[(2, 1)]["next"] == pytest.approx((2.0, 2.0))

    # Episode B — episode ends on intervention: done=True survives round-trip
    assert transitions[(2, 2)]["done"]
    # next_state is a dummy (lerobot sets next_obs = current_obs for done=True frames);
    # TD ignores it via (1 - done) * gamma * V(next_state) = 0, so we only check it's not the
    # real next observation (2,3) that was never stored in this episode
    assert transitions[(2, 2)]["next"] != pytest.approx((2.0, 3.0))


def test_save_training_checkpoint_allows_empty_online_buffer_for_offline_only_training(tmp_path, monkeypatch):
    policy = _policy_stub()
    online_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )

    monkeypatch.setattr(learner_server, "save_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(learner_server, "update_last_checkpoint", lambda checkpoint_dir: None)

    cfg = SimpleNamespace(
        output_dir=tmp_path / "out",
        dataset=SimpleNamespace(repo_id="repo"),
        env=SimpleNamespace(task="task"),
    )

    save_training_checkpoint(
        cfg=cfg,
        primitive_id="insert",
        optimization_step=100,
        online_steps=1000,
        interaction_message={"Interaction step": 0},
        policy=SimpleNamespace(),
        optimizers={},
        replay_buffer=online_buffer,
        fps=10,
    )

    # Online dataset not written when buffer is empty; no dataset-offline written at all
    assert not (tmp_path / "out" / "insert" / "dataset").exists()
    assert not (tmp_path / "out" / "insert" / "dataset-offline").exists()
