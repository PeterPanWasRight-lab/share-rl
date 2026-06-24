#!/usr/bin/env python3
"""Copy-paste ZeroMQ client for the FoundationPose publisher."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import zmq


class FoundationPoseClientError(RuntimeError):
    """Raised when the FoundationPose service returns an error."""


class FoundationPoseClient:
    """
    Thin REQ client for the FoundationPose ZeroMQ publisher.

    Typical lifecycle:
        client = FoundationPoseClient("tcp://127.0.0.1:5555")
        client.start_estimation(object_model_path=mesh_path, camera_intrinsics=K)
        pose = client.first_frame(image=rgb, depth=depth, mask=mask)["pose"]
        pose = client.track_frame(image=rgb, depth=depth)["pose"]

    The same client instance can be reused to switch to a new mesh:
        client.start_estimation(object_model_path=new_mesh_path, camera_intrinsics=K)
        pose = client.first_frame(image=rgb, depth=depth, mask=mask)["pose"]
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 30000,
        linger_ms: int = 0,
        context: Optional[zmq.Context] = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self.linger_ms = int(linger_ms)
        self._owns_context = context is None
        self._context = context or zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, self.linger_ms)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(self.endpoint)

    def close(self) -> None:
        if getattr(self, "_socket", None) is not None:
            self._socket.close(self.linger_ms)
            self._socket = None
        if self._owns_context and getattr(self, "_context", None) is not None:
            self._context.term()
            self._context = None

    def __enter__(self) -> "FoundationPoseClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def normalize_camera_intrinsics(camera_intrinsics: Any) -> np.ndarray:
        if isinstance(camera_intrinsics, dict):
            if "K" in camera_intrinsics:
                return FoundationPoseClient.normalize_camera_intrinsics(camera_intrinsics["K"])
            if {"fx", "fy", "cx", "cy"}.issubset(camera_intrinsics):
                return np.array(
                    [
                        [float(camera_intrinsics["fx"]), 0.0, float(camera_intrinsics["cx"])],
                        [0.0, float(camera_intrinsics["fy"]), float(camera_intrinsics["cy"])],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )

        K = np.asarray(camera_intrinsics, dtype=np.float64)
        if K.shape == (3, 3):
            return K
        if K.shape == (4,):
            fx, fy, cx, cy = K.tolist()
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        raise ValueError(
            "camera_intrinsics must be a 3x3 matrix, a length-4 array, "
            "or a dict with K or fx/fy/cx/cy"
        )

    def call(self, command: str, **payload: Any) -> Dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("client is closed")

        request = {"command": command, **payload}
        try:
            self._socket.send_pyobj(request)
            reply = self._socket.recv_pyobj()
        except zmq.Again as exc:
            raise TimeoutError(
                f"FoundationPose request timed out after {self.timeout_ms} ms: {command}"
            ) from exc

        if not isinstance(reply, dict):
            raise FoundationPoseClientError(f"invalid reply for {command}: {type(reply)!r}")

        if reply.get("status") != "ok":
            error = reply.get("error", "unknown service error")
            traceback = reply.get("traceback")
            message = f"FoundationPose {command} failed: {error}"
            if traceback:
                message = f"{message}\n{traceback}"
            raise FoundationPoseClientError(message)

        return reply.get("result", {})

    def ping(self) -> Dict[str, Any]:
        return self.call("ping")

    def status(self) -> Dict[str, Any]:
        return self.ping()

    def set_camera_intrinsics(self, camera_intrinsics: Any) -> Dict[str, Any]:
        K = self.normalize_camera_intrinsics(camera_intrinsics)
        result = self.call("set_camera_intrinsics", camera_intrinsics=K)
        if "camera_intrinsics" in result:
            result["camera_intrinsics"] = np.asarray(result["camera_intrinsics"], dtype=np.float64)
        return result

    def start_estimation(
        self,
        object_model_path: Optional[str] = None,
        mesh_file: Optional[str] = None,
        camera_intrinsics: Any = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if object_model_path is not None:
            payload["object_model_path"] = object_model_path
        if mesh_file is not None:
            payload["mesh_file"] = mesh_file
        if camera_intrinsics is not None:
            payload["camera_intrinsics"] = self.normalize_camera_intrinsics(camera_intrinsics)

        result = self.call("start_estimation", **payload)
        camera = result.get("camera")
        if isinstance(camera, dict) and camera.get("camera_intrinsics") is not None:
            camera["camera_intrinsics"] = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
        return result

    def first_frame(
        self,
        image: Any,
        depth: Any,
        mask: Any,
        camera_intrinsics: Any = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image": np.asarray(image),
            "depth": np.asarray(depth, dtype=np.float32),
            "mask": np.asarray(mask),
        }
        if camera_intrinsics is not None:
            payload["camera_intrinsics"] = self.normalize_camera_intrinsics(camera_intrinsics)

        result = self.call("first_frame", **payload)
        result["pose"] = np.asarray(result["pose"], dtype=np.float32)
        return result

    def track_frame(
        self,
        image: Any,
        depth: Any,
        camera_intrinsics: Any = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image": np.asarray(image),
            "depth": np.asarray(depth, dtype=np.float32),
        }
        if camera_intrinsics is not None:
            payload["camera_intrinsics"] = self.normalize_camera_intrinsics(camera_intrinsics)

        result = self.call("track_frame", **payload)
        result["pose"] = np.asarray(result["pose"], dtype=np.float32)
        return result

    def shutdown(self) -> Dict[str, Any]:
        return self.call("shutdown")


__all__ = ["FoundationPoseClient", "FoundationPoseClientError"]
