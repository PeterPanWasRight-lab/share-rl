"""Lightweight in-memory replay-buffer metrics for live monitoring."""

from __future__ import annotations

import time
from typing import Any

import torch


def summarize_replay_buffer(buffer: Any | None) -> dict[str, Any]:
    if buffer is None:
        return _empty_summary(capacity=0)

    size = int(len(buffer))
    capacity = int(getattr(buffer, "capacity", 0))
    if size == 0 or not getattr(buffer, "initialized", False):
        return _empty_summary(capacity=capacity)

    valid = slice(0, size)
    dones = buffer.dones[valid].detach().bool().cpu()
    truncateds = buffer.truncateds[valid].detach().bool().cpu()
    rewards = buffer.rewards[valid].detach().cpu()
    terminal = dones | truncateds

    intervention_count = 0
    complementary = getattr(buffer, "complementary_info", {})
    intervention = complementary.get("is_intervention")
    if isinstance(intervention, torch.Tensor):
        intervention_count = int(intervention[valid].detach().bool().sum().cpu().item())

    latest_index = (int(buffer.position) - 1) % capacity
    latest_is_terminal = bool(buffer.dones[latest_index] or buffer.truncateds[latest_index])
    completed = int(terminal.sum().item())
    successes = int(((rewards > 0) & terminal).sum().item())
    return {
        "transitions": size,
        "capacity": capacity,
        "fill_percent": round(100.0 * size / capacity, 1) if capacity else 0.0,
        "completed_trajectories": completed,
        "partial_trajectory_fragments": 0 if latest_is_terminal else 1,
        "interventions": intervention_count,
        "successes": successes,
        "terminal_failures": completed - successes,
        "positive_reward_transitions": int((rewards > 0).sum().item()),
        "write_position": int(buffer.position),
        "is_full": size >= capacity if capacity else False,
    }


def build_replay_metrics(
    *,
    online_buffers: dict[str, Any],
    offline_buffers: dict[str, Any | None],
    optimization_steps: dict[str, int],
) -> dict[str, Any]:
    return {
        "updated_at_unix": time.time(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "primitives": {
            primitive_id: {
                "optimization_step": int(optimization_steps.get(primitive_id, 0)),
                "online": summarize_replay_buffer(online_buffers.get(primitive_id)),
                "offline": summarize_replay_buffer(offline_buffers.get(primitive_id)),
            }
            for primitive_id in online_buffers
        },
    }
def _empty_summary(*, capacity: int) -> dict[str, Any]:
    return {
        "transitions": 0,
        "capacity": capacity,
        "fill_percent": 0.0,
        "completed_trajectories": 0,
        "partial_trajectory_fragments": 0,
        "interventions": 0,
        "successes": 0,
        "terminal_failures": 0,
        "positive_reward_transitions": 0,
        "write_position": 0,
        "is_full": False,
    }
