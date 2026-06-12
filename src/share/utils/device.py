try:
    from lerobot.utils.device_utils import get_safe_torch_device
except ImportError:  # pragma: no cover - compatibility with older lerobot layouts
    from lerobot.utils.utils import get_safe_torch_device

__all__ = ["get_safe_torch_device"]
