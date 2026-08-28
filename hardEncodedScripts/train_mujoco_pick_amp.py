#!/usr/bin/env python
"""Run standalone Pick AMP offline BC, online SAC, or SERL-aligned training."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from examples.demo_pick_amp import build_pick_amp_config
from share.configs.mpnet import DatasetRecordConfig
from share.configs.rl import MPNetTrainRLServerPipelineConfig
from share.rl.force_backoff import ForceBackoffConfig
from share.scripts.actor_server import run_actor
from share.scripts.learner_server import run_learner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/mujoco/pickAMPDemos100"),
    )
    common.add_argument("--batch-size", type=int, default=128)
    common.add_argument("--seed", type=int, default=20260827)
    common.add_argument("--save-freq", type=int, default=1000)
    common.add_argument("--learner-port", type=int, default=50071)
    common.add_argument("--online-buffer-capacity", type=int, default=10_000)

    offline = subparsers.add_parser("offline", parents=[common])
    offline.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco/pickAMPOfflineBC5K"),
    )
    offline.add_argument("--updates", type=int, default=5000)
    offline.add_argument("--bc-lr", type=float, default=3e-4)

    online = subparsers.add_parser("online", parents=[common])
    online.add_argument("--checkpoint", type=Path, required=True)
    online.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco/pickAMPOnlineSAC1K"),
    )
    online.add_argument("--updates", type=int, default=1000)
    online.add_argument("--critic-warmup-updates", type=int, default=700)
    online.add_argument("--online-warmup-steps", type=int, default=100)
    online.add_argument("--actor-lr", type=float, default=1e-5)
    online.add_argument("--actor-update-freq", type=int, default=20)
    online.add_argument("--bc-anchor-weight", type=float, default=1.0)
    online.add_argument("--exploration-std", type=float, default=0.01)
    online.add_argument(
        "--viewer",
        action="store_true",
        help="Show the interactive MuJoCo viewer in the actor process.",
    )

    hil_serl = subparsers.add_parser(
        "hil-serl",
        parents=[common],
        help="Paper-faithful RLPD/SAC training without intervention.",
    )
    hil_serl.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco/pickAMPHILSERLNoIntervention10K"),
    )
    hil_serl.add_argument("--updates", type=int, default=10_000)
    hil_serl.add_argument("--online-warmup-steps", type=int, default=100)
    hil_serl.add_argument(
        "--viewer",
        action="store_true",
        help="Show the interactive MuJoCo viewer in the actor process.",
    )
    hil_serl.set_defaults(
        batch_size=256,
        save_freq=5_000,
        online_buffer_capacity=10_000,
    )

    serl = subparsers.add_parser(
        "serl",
        parents=[common],
        help="SERL PickCube/RLPD-aligned training with 20 demonstrations.",
    )
    serl.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco/pickAMPSERLAligned10K"),
    )
    serl.add_argument("--updates", type=int, default=10_000)
    serl.add_argument("--online-warmup-steps", type=int, default=1_000)
    serl.add_argument("--random-action-steps", type=int, default=1_000)
    serl.add_argument(
        "--viewer",
        action="store_true",
        help="Show the interactive MuJoCo viewer in the actor process.",
    )
    serl.set_defaults(
        batch_size=256,
        save_freq=2_000,
        online_buffer_capacity=200_000,
    )
    return parser


def _build_pipeline(
    *,
    args: argparse.Namespace,
    training_mode: str,
    pretrained_path: Path | None,
) -> MPNetTrainRLServerPipelineConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_pick_amp_config(
        device=device,
        episode_steps=500,
        peg_xy_randomization_m=0.025,
        viewer=bool(getattr(args, "viewer", False)),
    )
    # Actor runtime expects this optional safety surface; Pick has no contact
    # insertion phase, so it is explicitly disabled for this standalone config.
    config.force_backoff = ForceBackoffConfig(enabled=False)
    policy = config.primitives["pick"].policy
    policy.training_mode = training_mode
    policy.online_steps = args.updates
    policy.online_buffer_capacity = args.online_buffer_capacity
    policy.actor_learner_config.learner_port = args.learner_port
    policy.pretrained_path = pretrained_path
    if training_mode == "bc":
        policy.offline_buffer_capacity = 50_000
        policy.bc_lr = args.bc_lr
    elif args.phase in {"hil-serl", "serl"}:
        if pretrained_path is not None:
            raise ValueError("HIL-SERL mode must start from a randomly initialized policy.")
        policy.offline_buffer_capacity = args.online_buffer_capacity
        policy.online_step_before_learning = args.online_warmup_steps
        policy.actor_update_after = 0
        policy.actor_lr = 3e-4
        policy.critic_lr = 3e-4
        policy.temperature_lr = 3e-4
        policy.discount = 0.96 if args.phase == "serl" else 0.97
        policy.temperature_init = 1e-2
        policy.use_backup_entropy = False
        policy.num_critics = 10 if args.phase == "serl" else 2
        policy.num_subsample_critics = 2 if args.phase == "serl" else None
        policy.utd_ratio = 4 if args.phase == "serl" else 2
        policy.policy_update_freq = 4 if args.phase == "serl" else 2
        policy.sac_bc_loss_weight = 0.0
        policy.freeze_shared_encoder_during_sac = False
        policy.stream_transitions_immediately = True
        policy.random_action_steps = getattr(args, "random_action_steps", 0)
        policy.limit_updates_to_online_transitions = args.phase != "serl"
        policy.proprio_latent_dim = None
        policy.policy_kwargs.std_min = 1e-5
        policy.policy_kwargs.std_max = 5.0
    else:
        policy.offline_buffer_capacity = 50_000
        policy.online_step_before_learning = args.online_warmup_steps
        policy.actor_update_after = args.critic_warmup_updates
        policy.actor_lr = args.actor_lr
        policy.policy_update_freq = args.actor_update_freq
        policy.sac_bc_loss_weight = args.bc_anchor_weight
        policy.freeze_shared_encoder_during_sac = True
        policy.policy_kwargs.std_min = args.exploration_std
        policy.policy_kwargs.std_max = args.exploration_std

    dataset = DatasetRecordConfig(
        repo_id="local/mujoco-pick-amp",
        root=str(args.dataset_root.resolve()),
        video_backend="pyav",
    )
    pipeline = MPNetTrainRLServerPipelineConfig(
        env=config,
        dataset=dataset,
        output_dir=args.output_dir.resolve(),
        job_name=f"mujoco-pick-amp-{training_mode}",
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=0,
        save_freq=args.save_freq,
        log_freq=100,
        replay_dashboard_enable=False,
    )
    pipeline.wandb.enable = False
    pipeline.validate(output_role="learner")
    return pipeline


def _require_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a fresh experiment into non-empty output: {path}"
        )


def run_offline(args: argparse.Namespace) -> dict[str, Any]:
    _require_fresh_output(args.output_dir)
    cfg = _build_pipeline(args=args, training_mode="bc", pretrained_path=None)
    result = run_learner(cfg)
    summary = {
        "phase": "offline_bc",
        "dataset_root": str(args.dataset_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "result": result,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def _learner_process(
    cfg: MPNetTrainRLServerPipelineConfig,
    shutdown_event: Any,
    result_queue: Any,
) -> None:
    try:
        result_queue.put({"status": "ok", "result": run_learner(cfg, shutdown_event)})
    except BaseException:  # noqa: BLE001
        result_queue.put({"status": "error", "traceback": traceback.format_exc()})
        shutdown_event.set()


def run_online(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = None
    if args.phase == "online":
        checkpoint = args.checkpoint.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
    _require_fresh_output(args.output_dir)
    learner_cfg = _build_pipeline(
        args=args,
        training_mode="sac",
        pretrained_path=checkpoint,
    )
    actor_cfg = _build_pipeline(
        args=args,
        training_mode="sac",
        pretrained_path=checkpoint,
    )
    context = mp.get_context("spawn")
    shutdown_event = context.Event()
    learner_results = context.Queue()
    learner_process = context.Process(
        target=_learner_process,
        args=(learner_cfg, shutdown_event, learner_results),
        daemon=True,
    )
    learner_process.start()
    try:
        actor_result = run_actor(actor_cfg, shutdown_event)
    finally:
        shutdown_event.set()
        learner_process.join()

    learner_message = learner_results.get(timeout=5.0)
    learner_results.close()
    if learner_message["status"] == "error":
        raise RuntimeError(learner_message["traceback"])
    learner_result = learner_message["result"]
    summary = {
        "phase": (
            "serl_aligned_rlpd"
            if args.phase == "serl"
            else "hil_serl_no_intervention"
            if args.phase == "hil-serl"
            else "online_sac"
        ),
        "pretrained_checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "critic_warmup_updates": getattr(args, "critic_warmup_updates", 0),
        "actor_update_freq": getattr(args, "actor_update_freq", 1),
        "bc_anchor_weight": getattr(args, "bc_anchor_weight", 0.0),
        "exploration_std": getattr(args, "exploration_std", None),
        "actor": actor_result,
        "learner": learner_result,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    args = _parser().parse_args()
    result = run_offline(args) if args.phase == "offline" else run_online(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
