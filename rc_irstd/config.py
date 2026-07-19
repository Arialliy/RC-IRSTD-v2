from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    loaded["_config_path"] = str(config_path)
    loaded["_config_dir"] = str(config_path.parent)
    return loaded


def _parse_override_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must use key=value syntax: {override}")
        dotted_key, raw_value = override.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"Invalid override key: {override}")
        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            current = cursor.get(key)
            if current is None:
                current = {}
                cursor[key] = current
            if not isinstance(current, dict):
                raise ValueError(f"Cannot descend into non-mapping key: {dotted_key}")
            cursor = current
        cursor[keys[-1]] = _parse_override_value(raw_value)
    return result


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cursor: Any = config
    for key in dotted_key.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value = deep_get(config, dotted_key, None)
    if value is None:
        raise KeyError(f"Missing required configuration key: {dotted_key}")
    return value


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(config.get("_config_dir", "."))
    return (base / path).resolve()


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}
