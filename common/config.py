"""Configuration loading for config-driven execution."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """Dict with attribute access, so ``cfg.lr`` and ``cfg["lr"]`` both work."""

    _RESERVED = ("_RESERVED",)

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith("__") and name not in Config._RESERVED:
            try:
                value = dict.__getitem__(self, name)
            except KeyError:
                pass
            else:
                return Config(value) if isinstance(value, dict) else value
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"No config key or attribute {name!r}. Available keys: {sorted(self)}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str) -> Config:
    """Load a YAML config, resolving a ``defaults:`` parent relative to it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent_ref = raw.pop("defaults", None)
    if parent_ref:
        parent = load_config((path.parent / parent_ref).resolve())
        raw = _deep_merge(parent, raw)

    raw.setdefault("_config_path", str(path))
    return Config(raw)


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    """Apply ``key.subkey=value`` CLI overrides, parsed as YAML scalars."""
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        target: dict[str, Any] = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    cfg.setdefault("_overrides", []).extend(overrides)
    return cfg


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    """Standard parser shared by every task entry point."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config.")
    parser.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Override config entries, e.g. --set train.epochs=1 method.lambda_mmd=10",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast reduced-scale sanity check instead of the full protocol.",
    )
    return parser
