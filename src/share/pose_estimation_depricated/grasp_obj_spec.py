import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraspObjectSpec:
    """Object descriptor needed for grasp pose estimation."""

    object_name: str
    mesh_path: str
    segmentation_prompt: str
    confidence_threshold: float=0.2

    def __post_init__(self) -> None:
        if not self.object_name:
            raise ValueError("object_name must be a non-empty string")
        if not self.mesh_path:
            raise ValueError("mesh_path must be a non-empty string")
        if not self.segmentation_prompt:
            raise ValueError("segmentation_prompt must be a non-empty string")
        confidence_threshold = float(self.confidence_threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0.0, 1.0]")
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_name": self.object_name,
            "mesh_path": self.mesh_path,
            "segmentation_prompt": self.segmentation_prompt,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraspObjectSpec":
        mesh_path = str(payload["mesh_path"])
        object_name = payload.get("object_name") or payload.get("name") or Path(mesh_path).stem
        return cls(
            object_name=str(object_name),
            mesh_path=mesh_path,
            segmentation_prompt=str(payload["segmentation_prompt"]),
            confidence_threshold=float(payload.get("confidence_threshold", 0.2)),
        )

    def to_json_file(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "GraspObjectSpec":
        json_path = Path(path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("GraspObjectSpec JSON must decode to an object")
        mesh_path_raw = payload.get("mesh_path")
        if mesh_path_raw is not None:
            mesh_path = Path(str(mesh_path_raw))
            if not mesh_path.is_absolute():
                payload["mesh_path"] = str((json_path.parent / mesh_path).resolve())
        return cls.from_dict(payload)

__all__ = ["GraspObjectSpec"]
