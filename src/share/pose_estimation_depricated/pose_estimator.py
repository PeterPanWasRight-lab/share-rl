"""High-level pose estimation orchestration for SAM3 + FoundationPose."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from share.pose_estimation.foundationpose_client import FoundationPoseClient
from share.pose_estimation.sam3_client import Sam3Client


class PoseEstimatorError(RuntimeError):
    """Raised when the pose-estimation pipeline cannot produce a valid result."""


class PoseEstimator:
    """
    Simple pose-estimation wrapper around SAM3 and FoundationPose.

    The estimator segments the current RGB frame with SAM3 using a text prompt
    and passes the resulting mask into FoundationPose.
    """

    def __init__(
        self,
        sam3_endpoint: str = "tcp://127.0.0.1:5565",
        foundationpose_endpoint: str = "tcp://127.0.0.1:5555",
        timeout_ms: int = 30000,
        mesh_path: str | None = None,
        prompt: str | None = None,
        camera_intrinsics: Any = None,
        image_format: str = "bgr",
        confidence_threshold: float | None = None,
    ) -> None:
        self.sam3_client = Sam3Client(
            endpoint=sam3_endpoint,
            timeout_ms=timeout_ms,
        )
        self.foundationpose_client = FoundationPoseClient(
            endpoint=foundationpose_endpoint,
            timeout_ms=timeout_ms,
        )
        self._default_camera_intrinsics = self._normalize_camera_intrinsics(camera_intrinsics)
        self._default_image_format = str(image_format).lower()
        self._default_confidence_threshold = (
            None if confidence_threshold is None else float(confidence_threshold)
        )
        self._default_mesh_path = None if mesh_path is None else str(mesh_path)
        self._default_prompt = None if prompt is None else str(prompt)

        self.reset_configuration()
        self.restart_tracking()

    def estimate_pose(
        self,
        *,
        image: Any,
        depth: Any,
    ) -> dict[str, Any]:
        """
        Estimate the object pose for the given RGB-D frame.

        Returns a dict with at least:
            pose: 4x4 object pose from FoundationPose
            mask: segmentation mask returned by SAM3 on first frame, else None
            box: optional segmentation box from SAM3 on first frame, else None
            segmentation: full SAM3 response on first frame, else None
            foundationpose: full FoundationPose response
        """
        segmentation_result: dict[str, Any] | None = None
        mask: np.ndarray | None = None
        box: np.ndarray | None = None

        if self._is_first_frame:
            if not self.mesh_path:
                raise ValueError("mesh_path must be configured before calling estimate_pose()")
            if not self.prompt:
                raise ValueError("prompt must be configured before calling estimate_pose()")

            segmentation_result = self.sam3_client.segment_image(
                image=image,
                image_format=self.image_format,
                prompt=self.prompt,
                confidence_threshold=self.confidence_threshold,
            )
            raw_mask = segmentation_result.get("mask")
            if raw_mask is None:
                raise PoseEstimatorError(
                    f"SAM3 returned no mask for prompt {self.prompt!r}."
                )
            mask = np.asarray(raw_mask, dtype=np.uint8)
            if not self._mask_has_foreground(mask):
                raise PoseEstimatorError(
                    f"SAM3 returned an empty mask for prompt {self.prompt!r}."
                )
            if segmentation_result.get("box") is not None:
                box = np.asarray(segmentation_result["box"], dtype=np.float32)
            self.foundationpose_client.start_estimation(
                object_model_path=self.mesh_path,
                camera_intrinsics=self.camera_intrinsics,
            )
            foundationpose_result = self.foundationpose_client.first_frame(
                image=image,
                depth=depth,
                mask=mask,
                camera_intrinsics=self.camera_intrinsics,
            )
            is_first_frame = True
            self._is_first_frame = False
        else:
            foundationpose_result = self.foundationpose_client.track_frame(
                image=image,
                depth=depth,
                camera_intrinsics=self.camera_intrinsics,
            )
            is_first_frame = False

        return {
            "pose": np.asarray(foundationpose_result["pose"], dtype=np.float32),
            "mask": mask,
            "box": box,
            "segmentation": segmentation_result,
            "foundationpose": foundationpose_result,
            "is_first_frame": is_first_frame,
        }
    

    def close(self) -> None:
        if getattr(self, "sam3_client", None) is not None:
            self.sam3_client.close()
        if getattr(self, "foundationpose_client", None) is not None:
            self.foundationpose_client.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _normalize_camera_intrinsics(camera_intrinsics: Any) -> np.ndarray | None:
        if camera_intrinsics is None:
            return None
        return FoundationPoseClient.normalize_camera_intrinsics(camera_intrinsics)

    @staticmethod
    def _mask_has_foreground(mask: np.ndarray) -> bool:
        return bool(np.any(mask > 0))

    def configure(
        self,
        *,
        mesh_path: str | None = None,
        prompt: str | None = None,
        camera_intrinsics: Any = None,
        image_format: str | None = None,
        confidence_threshold: float | None = None,
    ) -> None:
        tracking_invalidated = False
        if mesh_path is not None:
            self.mesh_path = str(mesh_path)
            tracking_invalidated = True
        if prompt is not None:
            self.prompt = str(prompt)
            tracking_invalidated = True
        if camera_intrinsics is not None:
            self.camera_intrinsics = self._normalize_camera_intrinsics(camera_intrinsics)
            tracking_invalidated = True
        if image_format is not None:
            self.image_format = str(image_format).lower()
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)
        if tracking_invalidated:
            self.restart_tracking()

    def reset_configuration(self) -> None:
        self.mesh_path = self._default_mesh_path
        self.prompt = self._default_prompt
        self.camera_intrinsics = (
            None if self._default_camera_intrinsics is None else self._default_camera_intrinsics.copy()
        )
        self.image_format = self._default_image_format
        self.confidence_threshold = self._default_confidence_threshold
        self.restart_tracking()

    def clear_mesh_path(self) -> None:
        self.mesh_path = None
        self.restart_tracking()

    def clear_prompt(self) -> None:
        self.prompt = None
        self.restart_tracking()

    def clear_camera_intrinsics(self) -> None:
        self.camera_intrinsics = None
        self.restart_tracking()

    def reset_image_format(self) -> None:
        self.image_format = self._default_image_format

    def reset_confidence_threshold(self) -> None:
        self.confidence_threshold = self._default_confidence_threshold

    def restart_tracking(self) -> None:
        """Force the next estimate_pose() call to reinitialize FoundationPose."""
        self._is_first_frame = True





__all__ = ["PoseEstimator", "PoseEstimatorError"]
