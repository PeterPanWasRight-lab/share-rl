import torch
from lerobot.rl.buffer import ReplayBuffer

from share.rl.buffer_metrics import build_replay_metrics, summarize_replay_buffer


def _add(buffer, *, reward=0.0, done=False, truncated=False, intervention=False):
    index = len(buffer)
    state = {"observation.state": torch.tensor([[float(index)]])}
    buffer.add(
        state=state,
        action=torch.tensor([[0.1]]),
        reward=reward,
        next_state={"observation.state": torch.tensor([[float(index + 1)]])},
        done=done,
        truncated=truncated,
        complementary_info={"is_intervention": torch.tensor([intervention])},
    )


def test_replay_buffer_summary_counts_transitions_episodes_and_interventions():
    buffer = ReplayBuffer(capacity=8, device="cpu", storage_device="cpu", use_drq=False)
    _add(buffer)
    _add(buffer, reward=1.0, done=True)
    _add(buffer, intervention=True)
    _add(buffer, truncated=True, intervention=True)

    assert summarize_replay_buffer(buffer) == {
        "transitions": 4,
        "capacity": 8,
        "fill_percent": 50.0,
        "completed_trajectories": 2,
        "partial_trajectory_fragments": 0,
        "interventions": 2,
        "successes": 1,
        "terminal_failures": 1,
        "positive_reward_transitions": 1,
        "write_position": 4,
        "is_full": False,
    }


def test_replay_metrics_payload_contains_both_buffers():
    buffer = ReplayBuffer(capacity=4, device="cpu", storage_device="cpu", use_drq=False)
    _add(buffer)

    payload = build_replay_metrics(
        online_buffers={"insert": buffer},
        offline_buffers={"insert": None},
        optimization_steps={"insert": 7},
    )

    assert payload["primitives"]["insert"]["optimization_step"] == 7
    assert payload["primitives"]["insert"]["online"]["transitions"] == 1
    assert payload["primitives"]["insert"]["offline"]["transitions"] == 0
