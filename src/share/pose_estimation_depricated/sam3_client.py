#!/usr/bin/env python3
"""Copy-paste ZeroMQ client for the SAM3 publisher."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import zmq


class Sam3ClientError(RuntimeError):
    """Raised when the SAM3 service returns an error."""


class Sam3Client:
    """
    Thin REQ client for the SAM3 ZeroMQ publisher.

    Typical lifecycle:
        client = Sam3Client("tcp://127.0.0.1:5565")
        client.set_prompt("red mug")
        result = client.segment_image(image=frame, image_format="bgr")

    A prompt can also be provided per request:
        result = client.segment_image(image=frame, prompt="red mug")
    """

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:5565",
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

    def __enter__(self) -> "Sam3Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def call(self, command: str, **payload: Any) -> Dict[str, Any]:
        if self._socket is None:
            raise RuntimeError("client is closed")

        request = {"command": command, **payload}
        try:
            self._socket.send_pyobj(request)
            reply = self._socket.recv_pyobj()
        except zmq.Again as exc:
            raise TimeoutError(f"SAM3 request timed out after {self.timeout_ms} ms: {command}") from exc

        if not isinstance(reply, dict):
            raise Sam3ClientError(f"invalid reply for {command}: {type(reply)!r}")

        if reply.get("status") != "ok":
            error = reply.get("error", "unknown service error")
            traceback = reply.get("traceback")
            message = f"SAM3 {command} failed: {error}"
            if traceback:
                message = f"{message}\n{traceback}"
            raise Sam3ClientError(message)

        return reply.get("result", {})

    def ping(self) -> Dict[str, Any]:
        return self.call("ping")

    def status(self) -> Dict[str, Any]:
        return self.ping()

    def set_prompt(self, prompt: str) -> Dict[str, Any]:
        return self.call("set_prompt", prompt=prompt)

    def set_confidence_threshold(self, confidence_threshold: float) -> Dict[str, Any]:
        return self.call("set_confidence_threshold", confidence_threshold=float(confidence_threshold))

    def segment_image(
        self,
        image: Any,
        image_format: str = "bgr",
        prompt: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "image": np.asarray(image),
            "image_format": str(image_format).lower(),
        }
        if prompt is not None:
            payload["prompt"] = prompt
        if confidence_threshold is not None:
            payload["confidence_threshold"] = float(confidence_threshold)

        result = self.call("segment_image", **payload)
        result["mask"] = np.asarray(result["mask"], dtype=np.uint8)
        if result.get("box") is not None:
            result["box"] = np.asarray(result["box"], dtype=np.float32)
        return result

    def shutdown(self) -> Dict[str, Any]:
        return self.call("shutdown")


__all__ = ["Sam3Client", "Sam3ClientError"]
