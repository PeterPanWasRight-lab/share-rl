#!/usr/bin/env python
"""Convert a 4-D XYZ+gripper checkpoint to the current 3-D XYZ interface."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def _trim_stats(stats: dict, key: str, expected_size: int) -> None:
    feature_stats = stats.get(key)
    if not isinstance(feature_stats, dict):
        return
    for stat_name, values in feature_stats.items():
        if isinstance(values, list) and len(values) == expected_size:
            feature_stats[stat_name] = values[:-1]


def _normalized_feature_mean(config: dict, feature_key: str, index: int) -> float:
    """Return the representative normalized value folded into a removed input bias."""
    stats = config.get("dataset_stats", {}).get(feature_key, {})
    mean = float(stats["mean"][index])
    mode = config.get("normalization_mapping", {}).get("STATE", "IDENTITY")
    if mode == "MEAN_STD":
        return 0.0
    if mode == "MIN_MAX":
        minimum = float(stats["min"][index])
        maximum = float(stats["max"][index])
        if maximum == minimum:
            return -1.0
        return 2.0 * (mean - minimum) / (maximum - minimum) - 1.0
    return mean


def _convert_processor(path: Path) -> None:
    tensors = load_file(str(path))
    converted = {}
    for key, value in tensors.items():
        if key.startswith("action.") and value.ndim == 1 and value.shape[0] == 4:
            value = value[:3]
        elif key.startswith("observation.state.") and value.ndim == 1 and value.shape[0] == 31:
            value = value[:30]
        converted[key] = value.contiguous()
    save_file(converted, str(path))


def convert(
    source: Path,
    destination: Path,
    online_steps: int,
    overwrite: bool,
    policy_type: str = "sac",
    initial_std: float = 0.1,
    initial_temperature: float = 0.01,
) -> None:
    if initial_std <= 0:
        raise ValueError("initial_std must be positive")
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if not (source / "config.json").is_file() or not (source / "model.safetensors").is_file():
        raise FileNotFoundError(f"Not a pretrained-model checkpoint: {source}")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}; pass --overwrite to replace it")
        shutil.rmtree(destination)
    shutil.copytree(source, destination)

    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    input_shape = config["input_features"]["observation.state"]["shape"]
    action_shape = config["output_features"]["action"]["shape"]
    is_legacy = input_shape == [31] and action_shape == [4]
    is_xyz = input_shape == [30] and action_shape == [3]
    if not (is_legacy or is_xyz):
        raise ValueError(
            "Expected either 31-D state + 4-D action or 30-D state + 3-D action, "
            f"got state={input_shape}, action={action_shape}"
        )

    preserve_hilserl_architecture = is_xyz and config.get("type") == "sac_dagger_bc"
    if policy_type == "sac" and preserve_hilserl_architecture:
        # The HIL-SERL checkpoint uses a custom observation encoder. Keep its
        # registered policy class and switch only the learner/action semantics.
        config["training_mode"] = "sac"
        config["bc_lr"] = None
        config["temperature_init"] = initial_temperature
    else:
        config["type"] = policy_type
    config["online_steps"] = online_steps
    if is_legacy:
        config["input_features"]["observation.state"]["shape"] = [30]
        config["output_features"]["action"]["shape"] = [3]
    if policy_type == "sac" and not preserve_hilserl_architecture:
        config.pop("bc_lr", None)
        config.pop("bc_loss_type", None)
    gripper_reference = _normalized_feature_mean(config, "observation.state", -1) if is_legacy else 0.0
    if is_legacy:
        _trim_stats(config.get("dataset_stats", {}), "observation.state", 31)
        _trim_stats(config.get("dataset_stats", {}), "action", 4)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    model_path = destination / "model.safetensors"
    tensors = load_file(str(model_path))
    if policy_type == "sac" and preserve_hilserl_architecture:
        # MSE behavior cloning does not train the actor's std head. Initialize
        # narrow, trainable exploration around the cloned mean instead of
        # exposing the robot to the random std weights from BC construction.
        tensors["actor.std_layer.weight"] = torch.zeros_like(tensors["actor.std_layer.weight"])
        tensors["actor.std_layer.bias"] = torch.full_like(
            tensors["actor.std_layer.bias"],
            math.log(initial_std),
        )
        if "log_alpha" in tensors:
            tensors["log_alpha"] = torch.full_like(tensors["log_alpha"], math.log(initial_temperature))
    state_weight_key = "actor.encoder.state_encoder.0.weight"
    state_bias_key = "actor.encoder.state_encoder.0.bias"
    state_weight = tensors.get(state_weight_key)
    if state_weight is not None and state_weight.shape[1] == 31:
        tensors[state_bias_key] = tensors[state_bias_key] + state_weight[:, -1] * gripper_reference
    converted = {}
    for key, value in tensors.items():
        if is_legacy and key == "actor.encoder.state_encoder.0.weight" and tuple(value.shape)[1:] == (31,):
            value = value[:, :30]
        elif key in {"actor.mean_layer.weight", "actor.std_layer.weight"} and value.shape[0] == 4:
            value = value[:3]
        elif key in {"actor.mean_layer.bias", "actor.std_layer.bias"} and value.shape[0] == 4:
            value = value[:3]
        elif (key.startswith("critic_ensemble.") or key.startswith("critic_target.")) and key.endswith(
            "net.net.0.weight"
        ) and value.shape[1] == 772:
            # Encoded observation occupies the first 768 columns; the old gripper
            # action is the final column. Critics were not trained during BC.
            value = value[:, :771]
        converted[key] = value.contiguous()
    save_file(converted, str(model_path))

    if is_legacy:
        for processor_path in destination.glob("*processor*.safetensors"):
            _convert_processor(processor_path)

    print(f"XYZ {policy_type} checkpoint: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--online-steps", type=int, default=40_000)
    parser.add_argument("--policy-type", choices=("sac", "sac_dagger_bc"), default="sac")
    parser.add_argument("--initial-std", type=float, default=0.1)
    parser.add_argument("--initial-temperature", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(
        args.source,
        args.destination,
        args.online_steps,
        args.overwrite,
        args.policy_type,
        args.initial_std,
        args.initial_temperature,
    )


if __name__ == "__main__":
    main()
