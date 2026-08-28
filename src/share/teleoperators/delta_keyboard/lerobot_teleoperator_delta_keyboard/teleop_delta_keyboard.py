#!/usr/bin/env python

import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from queue import Queue
from typing import Any

from lerobot.processor import RobotAction
from lerobot.processor.hil_processor import HasTeleopEvents
from lerobot.teleoperators import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from share.teleoperators.utils import TeleopEvents

from .config_delta_keyboard import (
    KeyboardAxisBinding,
    KeyboardVelocityTeleopConfig,
)

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


logger = logging.getLogger(__name__)


class KeyboardVelocityTeleop(Teleoperator, HasTeleopEvents):
    """
    Configurable keyboard teleoperator returning velocity commands, e.g.:

        {
            "x.vel": 0.1,
            "y.vel": 0.0,
            "z.vel": -0.1,
            "rx.vel": 0.0,
            "ry.vel": 0.5,
            "rz.vel": 0.0,
        }

    Key bindings and axis scales are defined in the config.
    """

    config_class = KeyboardVelocityTeleopConfig
    name = "keyboard_velocity"

    AXES = ("x", "y", "z", "rx", "ry", "rz")

    def __init__(self, config: KeyboardVelocityTeleopConfig):
        super().__init__(config)
        self.config = config

        self.event_queue = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None
        self._remote_socket: socket.socket | None = None
        self._remote_socket_path = Path(
            os.environ.get("SHARE_KEYBOARD_TELEOP_SOCKET", "/tmp/share_keyboard_teleop.sock")
        )
        self._connected = False
        self.logs: dict[str, float] = {}
        self._gripper_position = float(self.config.initial_gripper_position)

        self._event_states: dict[str, bool] = {
            event_name: False for event_name in self.config.event_bindings
        }
        self._prev_event_key_state: dict[str, bool] = {
            event_name: False for event_name in self.config.event_bindings
        }
        self._pressed_since_event_read: set[str] = set()
        self._remote_key_values_pending: dict[str, float] | None = None
        self._remote_key_values_active: dict[str, float] = {}

    @property
    def action_features(self) -> dict[str, type]:
        features = {f"{axis}.vel": float for axis in self.AXES}
        if self.config.gripper_enabled:
            features["gripper.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        listener_class = getattr(keyboard, "Listener", None) if keyboard is not None else None
        if listener_class is None:
            return self._connected
        return (
            isinstance(self.listener, listener_class)
            and self.listener.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        listener_class = getattr(keyboard, "Listener", None) if keyboard is not None else None
        if PYNPUT_AVAILABLE and listener_class is not None:
            logger.info("pynput is available - enabling local keyboard listener.")
            self.listener = listener_class(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logger.info("pynput listener not available - skipping local keyboard listener.")
            self.listener = None
        self._connect_remote_keyboard()
        self._connected = True

    def _connect_remote_keyboard(self) -> None:
        """Accept optional local key tokens from automation without X11 injection."""
        path = self._remote_socket_path.expanduser().resolve()
        if path.exists():
            proc_net_unix = Path("/proc/net/unix")
            active = proc_net_unix.exists() and str(path) in proc_net_unix.read_text(errors="replace")
            if active:
                logger.warning("Remote keyboard socket already active at %s; automation disabled", path)
                return
            path.unlink()
        remote_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        remote_socket.bind(str(path))
        remote_socket.setblocking(False)
        os.chmod(path, 0o600)
        self._remote_socket = remote_socket
        self._remote_socket_path = path
        logger.info("Remote keyboard automation socket: %s", path)

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _normalize_key(self, key) -> str | None:
        """
        Convert pynput key objects into stable config-friendly string tokens.
        """
        if key is None:
            return None

        # Character key
        if hasattr(key, "char") and key.char is not None:
            return str(key.char).lower()

        # Special key
        if keyboard is not None and isinstance(key, keyboard.Key):
            token = key.name.lower()
            # pynput aliases left Ctrl/Alt to the unsuffixed enum names on
            # Linux, while right Alt may be exposed as AltGr.
            return {
                "ctrl": "ctrl_l",
                "alt": "alt_l",
                "alt_gr": "alt_r",
            }.get(token, token)

        # Fallback
        try:
            return str(key).lower()
        except Exception:
            return None

    def _on_press(self, key) -> None:
        token = self._normalize_key(key)
        if token is not None:
            self.event_queue.put((token, True))

    def _on_release(self, key) -> None:
        token = self._normalize_key(key)
        if token is not None:
            self.event_queue.put((token, False))

            if self.config.escape_disconnects and token == "esc":
                logger.info("ESC pressed, disconnecting.")
                self.disconnect()

    def _drain_pressed_keys(self) -> None:
        self._drain_remote_keys()
        while not self.event_queue.empty():
            token, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[token] = True
                self._pressed_since_event_read.add(token)
            else:
                self.current_pressed.pop(token, None)

    def _drain_remote_keys(self) -> None:
        if self._remote_socket is None:
            return
        while True:
            try:
                message = json.loads(self._remote_socket.recv(4096))
            except BlockingIOError:
                return
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
                logger.warning("Ignoring malformed remote keyboard event: %s", exc)
                continue
            if isinstance(message, dict) and isinstance(message.get("pulse"), list):
                pulse_values: dict[str, float] = {}
                for item in message["pulse"]:
                    if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                        continue
                    value = item.get("value", 1.0)
                    if isinstance(value, (int, float)):
                        pulse_values[item["key"].lower()] = max(0.0, min(1.0, float(value)))
                self._remote_key_values_pending = pulse_values
                continue
            token = message.get("key") if isinstance(message, dict) else None
            pressed = message.get("pressed") if isinstance(message, dict) else None
            if not isinstance(token, str) or not isinstance(pressed, bool):
                logger.warning("Ignoring malformed remote keyboard event: %r", message)
                continue
            self.event_queue.put((token.lower(), pressed))

    def _axis_value(self, binding: KeyboardAxisBinding) -> float:
        if not binding.enabled or not self._all_keys_pressed(binding.required_keys):
            return 0.0

        value = 0.0
        if binding.pos_key is not None:
            value += (
                1.0
                if self.current_pressed.get(binding.pos_key, False)
                else self._remote_key_values_active.get(binding.pos_key, 0.0)
            )
        if binding.neg_key is not None:
            value -= (
                1.0
                if self.current_pressed.get(binding.neg_key, False)
                else self._remote_key_values_active.get(binding.neg_key, 0.0)
            )

        return value * binding.scale

    def _all_keys_pressed(self, keys: tuple[str, ...]) -> bool:
        return all(self.current_pressed.get(key, False) for key in keys)

    def _gripper_key_pressed(self, key: str) -> bool:
        return self._all_keys_pressed(self.config.gripper_required_keys) and self.current_pressed.get(
            key, False
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        start = time.perf_counter()

        self._drain_pressed_keys()

        action = {
            "x.vel": self._axis_value(self.config.x),
            "y.vel": self._axis_value(self.config.y),
            "z.vel": self._axis_value(self.config.z),
            "rx.vel": self._axis_value(self.config.rx),
            "ry.vel": self._axis_value(self.config.ry),
            "rz.vel": self._axis_value(self.config.rz),
        }
        if self.config.gripper_enabled:
            if self._gripper_key_pressed(self.config.gripper_open_key):
                self._gripper_position = 0.0
            if self._gripper_key_pressed(self.config.gripper_close_key):
                self._gripper_position = 1.0
            action["gripper.pos"] = self._gripper_position

        self.logs["read_pos_dt_s"] = time.perf_counter() - start
        return action

    def set_gripper_position(self, position: float) -> None:
        """Set the persistent gripper target used by scripted demos."""
        self._gripper_position = max(0.0, min(1.0, float(position)))

    def reset_episode(self) -> None:
        """Clear stale keys and restore the configured gripper target."""
        self.current_pressed.clear()
        while not self.event_queue.empty():
            self.event_queue.get_nowait()
        self._gripper_position = float(self.config.initial_gripper_position)
        self._event_states = {
            event_name: False for event_name in self.config.event_bindings
        }
        self._prev_event_key_state = {
            event_name: False for event_name in self.config.event_bindings
        }
        self._pressed_since_event_read.clear()
        self._remote_key_values_pending = None
        self._remote_key_values_active.clear()

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Optional event interface.

        Configurable events are defined in config.event_bindings.
        If include_intervention_event=True, IS_INTERVENTION is True whenever any motion axis is active.
        """
        self._drain_pressed_keys()
        self._remote_key_values_active = self._remote_key_values_pending or {}
        self._remote_key_values_pending = None

        events: dict[str, Any] = {}

        for event_name, binding in self.config.event_bindings.items():
            # Preserve short taps whose press and release both arrive between
            # two control cycles, especially Enter and slash episode markers.
            current_pressed = (
                self.current_pressed.get(binding.key, False)
                or binding.key in self._pressed_since_event_read
            )
            prev_pressed = self._prev_event_key_state.get(event_name, False)

            if binding.toggle:
                if current_pressed and not prev_pressed:
                    self._event_states[event_name] = not self._event_states[event_name]
                events[event_name] = self._event_states[event_name]
            else:
                events[event_name] = current_pressed

            self._prev_event_key_state[event_name] = current_pressed

        self._pressed_since_event_read.clear()

        if self.config.include_intervention_event:
            action = self.get_action()
            motion_active = any(abs(action[f"{axis}.vel"]) > 0.0 for axis in self.AXES)
            gripper_active = self.config.gripper_enabled and (
                self._gripper_key_pressed(self.config.gripper_open_key)
                or self._gripper_key_pressed(self.config.gripper_close_key)
            )
            events[TeleopEvents.IS_INTERVENTION] = motion_active or gripper_active

        return events

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if self._remote_socket is not None:
            self._remote_socket.close()
            self._remote_socket = None
            if self._remote_socket_path.exists():
                self._remote_socket_path.unlink()
        self._connected = False
