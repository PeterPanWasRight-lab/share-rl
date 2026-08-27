"""Standalone Flask backend for the MP-Net configuration editor."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from share.workspace.mpnet import (  # noqa: E402
    TRANSITION_TYPES,
    _decode_mpnet,
    _encode_mpnet,
    apply_edit,
    create_template_mpnet,
    summarize_mpnet_debug,
    validate_mpnet_config,
)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CONFIGS_DIR = Path(os.environ.get("MPNET_WEB_CONFIG_DIR", THIS_DIR / "configs")).resolve()
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PRIMITIVE_TYPES = {
    "static": "静态目标位姿原语",
    "move_delta": "增量运动原语",
    "open_loop_trajectory": "开环轨迹原语",
}


def _json_safe(value: Any) -> Any:
    """Return strict-JSON data; unbounded pose limits are represented by null."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _config_name(name: str) -> str:
    candidate = str(name or "")
    if candidate.endswith(".json"):
        candidate = candidate[:-5]
    if not NAME_PATTERN.fullmatch(candidate):
        raise ValueError("配置名只能包含字母、数字、下划线和连字符，长度为 1–64")
    return candidate


def _config_path(name: str) -> Path:
    return CONFIGS_DIR / f"{_config_name(name)}.json"


def _decode(payload: Any):
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return validate_mpnet_config(_decode_mpnet(payload))


def _load(path: Path):
    return _decode(json.loads(path.read_text(encoding="utf-8")))


def _save(config, path: Path) -> None:
    """Validate and atomically persist the browser serialization format."""
    validate_mpnet_config(config)
    payload = _json_safe(_encode_mpnet(config))
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _response(config, name: str):
    return jsonify(
        {
            "name": name,
            "summary": _json_safe(summarize_mpnet_debug(config)),
            "raw": _json_safe(_encode_mpnet(config)),
        }
    )


@app.errorhandler(404)
def _not_found(_error):
    return jsonify({"error": "资源不存在"}), 404


@app.errorhandler(405)
def _method_not_allowed(_error):
    return jsonify({"error": "不支持该请求方法"}), 405


@app.route("/")
def index():
    return send_from_directory(THIS_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.get("/api/configs")
def list_configs():
    return jsonify(sorted(path.stem for path in CONFIGS_DIR.glob("*.json")))


@app.post("/api/configs")
def create_config():
    try:
        body = request.get_json(silent=True) or {}
        name = _config_name(body.get("name", "new_config"))
        path = _config_path(name)
        if path.exists():
            return jsonify({"error": f"配置 '{name}' 已存在"}), 409
        primitive_name = _config_name(body.get("primitive_name", "main"))
        config = create_template_mpnet(primitive_name, notes=body.get("notes"))
        _save(config, path)
        return _response(config, name), 201
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/configs/<name>")
def get_config(name: str):
    try:
        clean_name = _config_name(name)
        path = _config_path(clean_name)
        if not path.exists():
            return jsonify({"error": f"配置 '{clean_name}' 不存在"}), 404
        return _response(_load(path), clean_name)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.put("/api/configs/<name>")
def save_config(name: str):
    try:
        clean_name = _config_name(name)
        path = _config_path(clean_name)
        if not path.exists():
            return jsonify({"error": f"配置 '{clean_name}' 不存在"}), 404
        config = _decode(request.get_json(silent=True))
        _save(config, path)
        return _response(config, clean_name)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.delete("/api/configs/<name>")
def delete_config(name: str):
    try:
        clean_name = _config_name(name)
        path = _config_path(clean_name)
        if not path.exists():
            return jsonify({"error": f"配置 '{clean_name}' 不存在"}), 404
        path.unlink()
        return jsonify({"name": clean_name, "message": "已删除"})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/configs/<name>/edit")
def edit_config(name: str):
    try:
        clean_name = _config_name(name)
        path = _config_path(clean_name)
        if not path.exists():
            return jsonify({"error": f"配置 '{clean_name}' 不存在"}), 404
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not body.get("operation"):
            raise ValueError("缺少 operation 字段")
        arguments = body.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("arguments 必须是 JSON 对象")
        config = apply_edit(_load(path), body["operation"], arguments)
        _save(config, path)
        return _response(config, clean_name)
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/validate")
def validate_config():
    try:
        _decode(request.get_json(silent=True))
        return jsonify({"valid": True, "message": "配置验证通过"})
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"valid": False, "message": str(exc)}), 400


def _field_default(field: dataclasses.Field) -> Any:
    if field.default is not dataclasses.MISSING:
        return _json_safe(field.default)
    if field.default_factory is not dataclasses.MISSING:
        return _json_safe(field.default_factory())
    return None


@app.get("/api/transition-types")
def transition_types():
    result = {}
    hidden = {"source", "target", "additional_reward", "reason"}
    for type_name, transition_class in TRANSITION_TYPES.items():
        fields = {
            field.name: {"type": str(field.type), "default": _field_default(field)}
            for field in dataclasses.fields(transition_class)
            if field.name not in hidden
        }
        result[type_name] = {
            "doc": (transition_class.__doc__ or type_name).strip().splitlines()[0],
            "fields": fields,
        }
    return jsonify(result)


@app.get("/api/primitive-types")
def primitive_types():
    return jsonify({name: {"doc": doc} for name, doc in PRIMITIVE_TYPES.items()})


def _seed_example_config() -> None:
    if any(CONFIGS_DIR.glob("*.json")):
        return
    config = create_template_mpnet("main", notes="自动生成的 MP-Net 模板")
    _save(config, _config_path("template"))


if __name__ == "__main__":
    _seed_example_config()
    host = os.environ.get("MPNET_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("MPNET_WEB_PORT", "5050"))
    debug = os.environ.get("MPNET_WEB_DEBUG", "0") == "1"
    print(f"MP-Net 编辑器: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
