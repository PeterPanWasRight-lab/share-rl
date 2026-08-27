"""Flat JSON adapter shared by WebConfig and its runnable examples."""

from __future__ import annotations

import copy
import dataclasses
import json
from enum import Enum
from pathlib import Path
from typing import Any

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    HookConfig,
    ImagePreprocessingConfig,
    KinematicsConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.workspace.mpnet import _decode_mpnet, _encode_mpnet, validate_mpnet_config


def _restore_bounds(frame: dict[str, Any]) -> None:
    min_pose = frame.get("min_pose")
    max_pose = frame.get("max_pose")
    if isinstance(min_pose, list):
        frame["min_pose"] = [float("-inf") if value is None else value for value in min_pose]
    if isinstance(max_pose, list):
        frame["max_pose"] = [float("inf") if value is None else value for value in max_pose]


def _restore_task_frames(primitive: dict[str, Any]) -> None:
    task_frame = primitive.get("task_frame", {})
    if not isinstance(task_frame, dict):
        return
    if "target" in task_frame:
        _restore_bounds(task_frame)
        return
    for frame in task_frame.values():
        if isinstance(frame, dict):
            _restore_bounds(frame)


def _restore_processor(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    image_payload = payload.get("image_preprocessing")
    image_preprocessing = None
    if isinstance(image_payload, dict):
        image_payload = copy.deepcopy(image_payload)
        if image_payload.get("resize_size") is not None:
            image_payload["resize_size"] = tuple(image_payload["resize_size"])
        image_preprocessing = ImagePreprocessingConfig(**image_payload)
    events_payload = copy.deepcopy(payload.get("events", {}))
    if events_payload.get("pulse_events") is not None:
        events_payload["pulse_events"] = tuple(events_payload["pulse_events"])
    return ManipulationPrimitiveProcessorConfig(
        control_time_s=payload.get("control_time_s", 10.0),
        fps=payload.get("fps", 10.0),
        image_preprocessing=image_preprocessing,
        events=EventConfig(**events_payload),
        hooks=HookConfig(**copy.deepcopy(payload.get("hooks", {}))),
        observation=ObservationConfig(**copy.deepcopy(payload.get("observation", {}))),
        gripper=GripperConfig(**copy.deepcopy(payload.get("gripper", {}))),
        kinematics=KinematicsConfig(**copy.deepcopy(payload.get("kinematics", {}))),
    )


def decode_flat_mpnet(payload: Any):
    """Decode WebConfig JSON into runtime-ready MP-Net dataclasses."""
    if not isinstance(payload, dict):
        raise ValueError("MP-Net config must be a JSON object")
    normalized = copy.deepcopy(payload)
    primitives = normalized.get("primitives", {})
    if not isinstance(primitives, dict):
        raise ValueError("primitives must be a JSON object")
    for primitive in primitives.values():
        if not isinstance(primitive, dict):
            raise ValueError("each primitive must be a JSON object")
        _restore_task_frames(primitive)
        primitive["processor"] = _restore_processor(primitive.get("processor", {}))
    return validate_mpnet_config(_decode_mpnet(normalized))


def load_flat_mpnet(path: str | Path):
    return decode_flat_mpnet(json.loads(Path(path).read_text(encoding="utf-8")))


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def encode_flat_mpnet(config) -> dict[str, Any]:
    """Encode a runtime config without leaking dataclass instances into JSON."""
    return _plain(_encode_mpnet(config))
