from __future__ import annotations

from types import SimpleNamespace
import queue

import torch
from lerobot.rl.buffer import ReplayBuffer
from lerobot.transport.utils import transitions_to_bytes

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
    policy = _policy_stub()
    saved_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=True,
    )
    _add_demo_transitions(saved_buffer)

    primitive_root = tmp_path / "out" / "insert"
    saved_buffer.to_lerobot_dataset(repo_id="repo-insert", fps=10, root=str(primitive_root / "dataset"))
    saved_buffer.to_lerobot_dataset(repo_id="repo-insert", fps=10, root=str(primitive_root / "dataset-offline"))

    cfg = SimpleNamespace(
        resume=True,
        output_dir=tmp_path / "out",
        dataset=SimpleNamespace(repo_id="repo"),
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
    assert len(offline_buffers["insert"]) == len(saved_buffer)


def test_process_transitions_splits_offline_intervention_segments():
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
    assert len(offline_buffer) == 4
    assert torch.equal(
        offline_buffer.dones[:4].cpu(),
        torch.tensor([False, True, False, True]),
    )
    assert torch.equal(
        offline_buffer.truncateds[:4].cpu(),
        torch.tensor([False, False, False, False]),
    )
    assert torch.equal(
        offline_buffer.complementary_info[learner_server.SEGMENT_END_KEY][:4].cpu().bool(),
        torch.tensor([False, True, False, True]),
    )
    assert torch.equal(
        offline_buffer.next_states["observation.state"][1].cpu(),
        torch.tensor([[2.0, 0.0]]),
    )


def test_td_valid_offline_iterator_skips_nonterminal_segment_boundaries():
    policy = _policy_stub()
    offline_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=False,
    )

    for index, segment_end, done in [
        (0, False, False),
        (1, True, False),
        (2, False, False),
        (3, True, True),
    ]:
        transition = _transition(index, is_intervention=True, done=done)
        transition["complementary_info"][learner_server.SEGMENT_END_KEY] = segment_end
        offline_buffer.add(**{key: value for key, value in transition.items() if key != "id"})

    assert learner_server._num_td_valid_offline_samples(offline_buffer) == 3

    batch = next(learner_server._td_valid_offline_iterator(offline_buffer, batch_size=16))
    segment_end = batch["complementary_info"][learner_server.SEGMENT_END_KEY].bool()
    done = batch["done"].bool()
    truncated = batch["truncated"].bool()
    assert not bool((segment_end & ~done & ~truncated).any())


def test_segment_split_export_marks_segment_boundaries_done_without_mutating_buffer(tmp_path):
    policy = _policy_stub()
    offline_buffer = ReplayBuffer(
        capacity=8,
        device="cpu",
        storage_device="cpu",
        state_keys=policy.config.input_features.keys(),
        optimize_memory=False,
    )

    for index, segment_end, done in [
        (0, False, False),
        (1, True, False),
        (2, False, False),
        (3, True, True),
    ]:
        transition = _transition(index, is_intervention=True, done=done)
        transition["complementary_info"][learner_server.SEGMENT_END_KEY] = segment_end
        offline_buffer.add(**{key: value for key, value in transition.items() if key != "id"})

    original_dones = offline_buffer.dones[:4].clone()
    learner_server._save_replay_buffer_to_lerobot_dataset(
        offline_buffer,
        repo_id="repo-insert",
        fps=10,
        root=str(tmp_path / "dataset-offline"),
        split_on_segment_end=True,
    )

    assert torch.equal(offline_buffer.dones[:4].cpu(), original_dones.cpu())
    assert torch.equal(
        offline_buffer.truncateds[:4].cpu(),
        torch.tensor([False, False, False, False]),
    )


def test_save_training_checkpoint_allows_empty_online_buffer_for_offline_only_training(tmp_path, monkeypatch):
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
        optimize_memory=True,
    )
    _add_demo_transitions(offline_buffer)

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
        offline_replay_buffer=offline_buffer,
        fps=10,
    )

    assert not (tmp_path / "out" / "insert" / "dataset").exists()
    assert (tmp_path / "out" / "insert" / "dataset-offline").exists()
