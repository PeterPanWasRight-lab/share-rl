"""Safe local runtime controls for the SHaRe-RL web console."""

from __future__ import annotations

import copy
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PROFILE_PATH = THIS_DIR / "runtime_profile.json"
SERVICE_ROLES = ("actor", "learner")
ENV_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
ACTOR_TIMING_PATTERN = re.compile(
    r"^(?:DEBUG|INFO) (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"\[ACTOR\] primitive=(?P<primitive>\S+) "
    r"loop=\s*(?P<loop_ms>[\d.]+)ms \((?P<loop_hz>[\d.]+)hz\) "
    r"(?:policy|step)=\s*(?P<work_ms>[\d.]+)ms \((?P<work_hz>[\d.]+)hz\)"
)
VIEWER_EXAMPLES = {
    "pick_insert": {
        "label": "MuJoCo Pick & Insert",
        "script": REPO_ROOT / "examples" / "demo_pick_insert.py",
    },
}

DEFAULT_PROFILE: dict[str, Any] = {
    "name": "MuJoCo insertion",
    "env_type": "mujoco_ur5e_insertion",
    "job_name": "web-console-insertion",
    "output_root": "outputs/web-console/insertion",
    "dataset_root": "",
    "learner_checkpoint": "",
    "actor_checkpoint": "",
    "seed": 20260827,
    "device": "cuda",
    "batch_size": 256,
    "save_freq": 1000,
    "log_freq": 100,
    "online_steps": 20000,
    "online_warmup_steps": 100,
    "actor_lr": 0.0003,
    "policy_update_freq": 1,
    "actor_update_after": 0,
    "frame_stack": 1,
    "image_size": 64,
    "vision_encoder": "helper2424/resnet10",
    "freeze_shared_encoder": False,
    "learner_host": "127.0.0.1",
    "learner_port": 50051,
    "replay_host": "127.0.0.1",
    "replay_port": 8000,
    "viewer": False,
    "teleop_mode": "none",
}


def _number(value: Any, name: str, *, minimum: float, maximum: float, integer: bool = False):
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def validate_profile(payload: Any) -> dict[str, Any]:
    """Validate and normalize the allowlisted launch profile fields."""
    if not isinstance(payload, dict):
        raise ValueError("profile must be a JSON object")
    unknown = sorted(set(payload) - set(DEFAULT_PROFILE))
    if unknown:
        raise ValueError(f"unsupported profile fields: {', '.join(unknown)}")
    profile = copy.deepcopy(DEFAULT_PROFILE)
    profile.update(payload)
    for name in (
        "name",
        "job_name",
        "output_root",
        "dataset_root",
        "learner_checkpoint",
        "actor_checkpoint",
        "vision_encoder",
    ):
        if not isinstance(profile[name], str):
            raise ValueError(f"{name} must be a string")
        profile[name] = profile[name].strip()
    if not profile["name"] or not profile["job_name"] or not profile["output_root"]:
        raise ValueError("name, job_name, and output_root are required")
    if not ENV_TYPE_PATTERN.fullmatch(str(profile["env_type"])):
        raise ValueError("env_type may only contain lowercase letters, numbers, and underscores")
    if profile["device"] not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    if profile["teleop_mode"] not in {"none", "keyboard"}:
        raise ValueError("teleop_mode must be none or keyboard")
    for name in ("viewer", "freeze_shared_encoder"):
        if not isinstance(profile[name], bool):
            raise ValueError(f"{name} must be true or false")
    profile["seed"] = _number(profile["seed"], "seed", minimum=0, maximum=2**31 - 1, integer=True)
    profile["batch_size"] = _number(profile["batch_size"], "batch_size", minimum=1, maximum=65536, integer=True)
    profile["save_freq"] = _number(profile["save_freq"], "save_freq", minimum=1, maximum=10**9, integer=True)
    profile["log_freq"] = _number(profile["log_freq"], "log_freq", minimum=1, maximum=10**9, integer=True)
    profile["online_steps"] = _number(profile["online_steps"], "online_steps", minimum=1, maximum=10**9, integer=True)
    profile["online_warmup_steps"] = _number(profile["online_warmup_steps"], "online_warmup_steps", minimum=0, maximum=10**9, integer=True)
    profile["policy_update_freq"] = _number(profile["policy_update_freq"], "policy_update_freq", minimum=1, maximum=10**6, integer=True)
    profile["actor_update_after"] = _number(profile["actor_update_after"], "actor_update_after", minimum=0, maximum=10**9, integer=True)
    profile["frame_stack"] = _number(profile["frame_stack"], "frame_stack", minimum=1, maximum=32, integer=True)
    profile["image_size"] = _number(profile["image_size"], "image_size", minimum=32, maximum=2048, integer=True)
    profile["learner_port"] = _number(profile["learner_port"], "learner_port", minimum=1, maximum=65535, integer=True)
    profile["replay_port"] = _number(profile["replay_port"], "replay_port", minimum=1, maximum=65535, integer=True)
    profile["actor_lr"] = _number(profile["actor_lr"], "actor_lr", minimum=1e-8, maximum=1.0)
    for name in ("learner_host", "replay_host"):
        if str(profile[name]) not in {"127.0.0.1", "localhost"}:
            raise ValueError(f"{name} is restricted to localhost")
    for name in ("dataset_root", "learner_checkpoint", "actor_checkpoint"):
        if profile[name]:
            _resolve_project_path(profile[name], name)
    return profile


def _resolve_project_path(value: str, name: str) -> Path:
    configured = Path(value)
    if configured.is_absolute():
        raise ValueError(f"{name} must be relative to the repository")
    resolved = (REPO_ROOT / configured).resolve()
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        raise ValueError(f"{name} must stay inside the repository")
    return resolved


def resolve_output_root(profile: dict[str, Any]) -> Path:
    return _resolve_project_path(profile["output_root"], "output_root")


def _asset_record(path: Path, *, label: str, timestamp_path: Path) -> dict[str, Any]:
    modified = timestamp_path.stat().st_mtime
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "label": label,
        "modified_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified)),
        "mtime": modified,
    }


def discover_project_assets() -> dict[str, list[dict[str, Any]]]:
    """Find valid local demo roots and policy checkpoints, newest first."""
    outputs = REPO_ROOT / "outputs"
    if not outputs.exists():
        return {"datasets": [], "checkpoints": []}

    datasets: dict[Path, dict[str, Any]] = {}
    for info_path in outputs.glob("**/meta/info.json"):
        stats_path = info_path.with_name("stats.json")
        if not stats_path.is_file():
            continue
        dataset_dir = info_path.parent.parent
        # Online replay exports use <primitive>/dataset/meta and are not demos.
        if dataset_dir.name == "dataset":
            continue
        if dataset_dir.parent.name == "offline-demos":
            root = dataset_dir.parent.parent
        else:
            root = dataset_dir.parent
        try:
            record = _asset_record(
                root,
                label=f"{root.relative_to(REPO_ROOT).as_posix()} · {dataset_dir.name}",
                timestamp_path=max((info_path, stats_path), key=lambda path: path.stat().st_mtime),
            )
        except (OSError, ValueError):
            continue
        previous = datasets.get(root)
        if previous is None or record["mtime"] > previous["mtime"]:
            datasets[root] = record

    checkpoints: dict[Path, dict[str, Any]] = {}
    for config_path in outputs.glob("**/checkpoints/*/pretrained_model/config.json"):
        policy_path = config_path.parent
        step_dir = policy_path.parent
        if step_dir.name == "last":
            continue
        try:
            relative = policy_path.relative_to(REPO_ROOT).as_posix()
            record = _asset_record(
                policy_path,
                label=f"{relative} · step {step_dir.name}",
                timestamp_path=config_path,
            )
        except (OSError, ValueError):
            continue
        checkpoints[policy_path] = record

    return {
        "datasets": sorted(datasets.values(), key=lambda item: item["mtime"], reverse=True),
        "checkpoints": sorted(checkpoints.values(), key=lambda item: item["mtime"], reverse=True),
    }


def load_profile() -> dict[str, Any]:
    if not PROFILE_PATH.exists():
        return copy.deepcopy(DEFAULT_PROFILE)
    return validate_profile(json.loads(PROFILE_PATH.read_text(encoding="utf-8")))


def save_profile(payload: Any) -> dict[str, Any]:
    profile = validate_profile(payload)
    resolve_output_root(profile)
    temporary = PROFILE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(PROFILE_PATH)
    return profile


def build_service_command(role: str, profile: dict[str, Any]) -> list[str]:
    """Build an argv-only command; browser input never becomes shell text."""
    if role not in SERVICE_ROLES:
        raise ValueError(f"unknown service role: {role}")
    profile = validate_profile(profile)
    script = REPO_ROOT / "src" / "share" / "scripts" / f"{role}_server.py"
    output_dir = resolve_output_root(profile)
    command = [
        sys.executable,
        str(script),
        f"--env.type={profile['env_type']}",
        f"--env.learner_host={profile['learner_host']}",
        f"--env.learner_port={profile['learner_port']}",
        f"--env.policy_device={profile['device']}",
        f"--env.online_steps={profile['online_steps']}",
        f"--env.online_step_before_learning={profile['online_warmup_steps']}",
        f"--env.policy_actor_lr={profile['actor_lr']}",
        f"--env.policy_update_freq={profile['policy_update_freq']}",
        f"--env.policy_actor_update_after={profile['actor_update_after']}",
        f"--env.policy_frame_stack={profile['frame_stack']}",
        f"--env.policy_image_size={profile['image_size']}",
        f"--env.policy_freeze_shared_encoder_during_sac={str(profile['freeze_shared_encoder']).lower()}",
        f"--env.viewer={str(profile['viewer'] and role == 'actor').lower()}",
        f"--env.teleop_mode={profile['teleop_mode'] if role == 'actor' else 'none'}",
        f"--output_dir={output_dir}",
        f"--job_name={profile['job_name']}",
        f"--seed={profile['seed']}",
        f"--batch_size={profile['batch_size']}",
        f"--save_freq={profile['save_freq']}",
        f"--log_freq={profile['log_freq']}",
        f"--replay_dashboard_enable={str(role == 'learner').lower()}",
        f"--replay_dashboard_host={profile['replay_host']}",
        f"--replay_dashboard_port={profile['replay_port']}",
        "--wandb.enable=false",
    ]
    if profile["vision_encoder"]:
        command.append(f"--env.policy_vision_encoder_name={profile['vision_encoder']}")
    if role == "learner" and profile["dataset_root"]:
        dataset_root = _resolve_project_path(profile["dataset_root"], "dataset_root")
        command.extend([
            "--dataset.repo_id=local/web-console",
            f"--dataset.root={dataset_root}",
        ])
    checkpoint = profile[f"{role}_checkpoint"]
    if checkpoint:
        checkpoint_path = _resolve_project_path(checkpoint, f"{role}_checkpoint")
        if not (checkpoint_path / "config.json").is_file():
            raise ValueError(
                f"{role}_checkpoint must point to a pretrained_model directory containing config.json"
            )
        command.append(f"--policy.path={checkpoint_path}")
    return command


class ServiceManager:
    """Own Actor/Learner child processes launched by this web server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, dict[str, Any]] = {}

    def _status_unlocked(self, role: str) -> dict[str, Any]:
        record = self._records.get(role)
        if record is None:
            return {"role": role, "state": "stopped", "pid": None, "started_at": None, "exit_code": None, "log_path": None}
        process: subprocess.Popen = record["process"]
        exit_code = process.poll()
        state = "running" if exit_code is None else "exited"
        if exit_code is None:
            elapsed_until = time.time()
        else:
            if "stopped_at_unix" not in record:
                record["stopped_at_unix"] = time.time()
                record["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed_until = record["stopped_at_unix"]
            log_handle = record.get("log_handle")
            if log_handle is not None and not log_handle.closed:
                log_handle.close()
        return {
            "role": role,
            "state": state,
            "pid": process.pid,
            "started_at": record["started_at"],
            "uptime_s": round(max(0.0, elapsed_until - record["started_at_unix"]), 1),
            "stopped_at": record.get("stopped_at"),
            "exit_code": exit_code,
            "log_path": str(record["log_path"]),
        }

    def status(self, role: str | None = None) -> Any:
        with self._lock:
            if role is not None:
                if role not in SERVICE_ROLES:
                    raise ValueError(f"unknown service role: {role}")
                return self._status_unlocked(role)
            return {name: self._status_unlocked(name) for name in SERVICE_ROLES}

    def start(self, role: str, profile: dict[str, Any]) -> dict[str, Any]:
        command = build_service_command(role, profile)
        with self._lock:
            if self._status_unlocked(role)["state"] == "running":
                raise ValueError(f"{role} is already running")
            if role == "actor" and self._status_unlocked("learner")["state"] != "running":
                raise ValueError("start the learner before starting the actor")
            previous = self._records.get(role)
            if previous is not None and not previous["log_handle"].closed:
                previous["log_handle"].close()
            log_dir = resolve_output_root(profile) / "console-logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{role}.log"
            log_handle = log_path.open("ab", buffering=0)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._records[role] = {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_at_unix": time.time(),
            }
            return self._status_unlocked(role)

    def stop(self, role: str) -> dict[str, Any]:
        with self._lock:
            if role not in SERVICE_ROLES:
                raise ValueError(f"unknown service role: {role}")
            if role == "learner" and self._status_unlocked("actor")["state"] == "running":
                raise ValueError("stop the actor before stopping the learner")
            record = self._records.get(role)
            if record is None or record["process"].poll() is not None:
                return self._status_unlocked(role)
            process: subprocess.Popen = record["process"]
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        with self._lock:
            record["log_handle"].close()
            return self._status_unlocked(role)

    def log_tail(self, role: str, lines: int = 120) -> str:
        status = self.status(role)
        if not status["log_path"]:
            return ""
        path = Path(status["log_path"])
        if not path.exists():
            return ""
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(lines, 5000)):])

    def clear_log(self, role: str) -> None:
        """Truncate one console-owned service log without stopping its process."""
        with self._lock:
            if role not in SERVICE_ROLES:
                raise ValueError(f"unknown service role: {role}")
            record = self._records.get(role)
            if record is None:
                return
            Path(record["log_path"]).write_text("", encoding="utf-8")


def fetch_replay_metrics(profile: dict[str, Any]) -> dict[str, Any]:
    profile = validate_profile(profile)
    url = f"http://{profile['replay_host']}:{profile['replay_port']}/api/metrics"
    try:
        with urllib.request.urlopen(url, timeout=0.7) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"connected": True, "url": url, "metrics": payload}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"connected": False, "url": url, "error": str(exc), "metrics": {"primitives": {}}}


def fetch_actor_timing(profile: dict[str, Any]) -> dict[str, Any]:
    """Read the latest Actor loop timing without exposing per-step lines in the UI log."""
    profile = validate_profile(profile)
    path = resolve_output_root(profile) / "logs" / f"actor_{profile['job_name']}.log"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 128 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    for line in reversed(lines):
        match = ACTOR_TIMING_PATTERN.search(line)
        if match is None:
            continue
        values = match.groupdict()
        return {
            "available": True,
            "primitive": values["primitive"],
            "loop_hz": float(values["loop_hz"]),
            "loop_ms": float(values["loop_ms"]),
            "policy_hz": float(values["work_hz"]),
            "policy_ms": float(values["work_ms"]),
            "timestamp": values["timestamp"],
        }
    return {"available": False, "path": str(path)}


class ViewerExampleRunner:
    """Run one allowlisted MuJoCo example with a saved MP-Net config."""

    def __init__(self, log_root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._record: dict[str, Any] | None = None
        self._log_root = log_root or REPO_ROOT / "outputs" / "web-console" / "example-runs"

    @staticmethod
    def examples() -> list[dict[str, str]]:
        return [
            {"id": example_id, "label": spec["label"], "script": spec["script"].name}
            for example_id, spec in VIEWER_EXAMPLES.items()
        ]

    def _status_unlocked(self) -> dict[str, Any]:
        if self._record is None:
            return {
                "state": "stopped",
                "pid": None,
                "example_id": None,
                "config_name": None,
                "exit_code": None,
                "log_path": None,
            }
        record = self._record
        process: subprocess.Popen = record["process"]
        exit_code = process.poll()
        if exit_code is None:
            state = "running"
            elapsed_until = time.time()
        else:
            state = "exited"
            if "stopped_at_unix" not in record:
                record["stopped_at_unix"] = time.time()
                record["stopped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            elapsed_until = record["stopped_at_unix"]
            if not record["log_handle"].closed:
                record["log_handle"].close()
        return {
            "state": state,
            "pid": process.pid,
            "example_id": record["example_id"],
            "config_name": record["config_name"],
            "started_at": record["started_at"],
            "stopped_at": record.get("stopped_at"),
            "uptime_s": round(max(0.0, elapsed_until - record["started_at_unix"]), 1),
            "exit_code": exit_code,
            "log_path": str(record["log_path"]),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_unlocked()

    def start(
        self,
        *,
        example_id: str,
        config_name: str,
        config_path: Path,
        steps: int,
    ) -> dict[str, Any]:
        spec = VIEWER_EXAMPLES.get(str(example_id))
        if spec is None:
            raise ValueError("unsupported viewer example")
        steps = _number(steps, "steps", minimum=1, maximum=100000, integer=True)
        config_path = config_path.resolve()
        if not config_path.is_file() or config_path.suffix != ".json":
            raise ValueError("saved MP-Net config does not exist")
        command = [
            sys.executable,
            str(spec["script"]),
            "--viewer",
            f"--steps={steps}",
            f"--config={config_path}",
        ]
        with self._lock:
            if self._status_unlocked()["state"] == "running":
                raise ValueError("a viewer example is already running")
            if self._record is not None and not self._record["log_handle"].closed:
                self._record["log_handle"].close()
            self._log_root.mkdir(parents=True, exist_ok=True)
            log_path = self._log_root / f"{example_id}.log"
            log_handle = log_path.open("ab", buffering=0)
            child_env = os.environ.copy()
            child_env["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._record = {
                "process": process,
                "log_handle": log_handle,
                "log_path": log_path,
                "example_id": example_id,
                "config_name": config_name,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_at_unix": time.time(),
            }
            return self._status_unlocked()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._record is None or self._record["process"].poll() is not None:
                return self._status_unlocked()
            process: subprocess.Popen = self._record["process"]
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
        with self._lock:
            return self._status_unlocked()

    def log_tail(self, lines: int = 160) -> str:
        status = self.status()
        if not status["log_path"]:
            return ""
        path = Path(status["log_path"])
        if not path.exists():
            return ""
        return "\n".join(
            path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -max(1, min(lines, 5000)):
            ]
        )

    def clear_log(self) -> None:
        """Truncate the current viewer log without stopping its process."""
        with self._lock:
            if self._record is None:
                return
            Path(self._record["log_path"]).write_text("", encoding="utf-8")


service_manager = ServiceManager()
example_runner = ViewerExampleRunner()
