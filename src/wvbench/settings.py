"""Configuration loading and resolution.

Sources, later wins:
  config/profiles.toml [defaults] -> [profiles.<name>] -> [overrides."<dataset>"]
  -> WVB_PARAMS env ("k=v,k=v") -> --param k=v (CLI)

Keys without a dot are index parameters; keys with a dot address another
section of profiles.toml ("ingest.mode=server_side").
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]


def _path_env(name: str, container_default: str, local_default: Path) -> Path:
    """Use the env var if set, else the baseline container path if it exists, else a repo path."""
    if os.getenv(name):
        return Path(os.environ[name])
    if Path(container_default).is_dir():
        return Path(container_default)
    return local_default


CONFIG_DIR = Path(os.getenv("WVB_CONFIG_DIR", ROOT_DIR / "config"))
DATA_DIR = _path_env("WVB_DATA_DIR", "/data", ROOT_DIR / "data")
DB_DIR = _path_env("WVB_DB_DIR", "/db", ROOT_DIR / "db-data")
RESULTS_DIR = _path_env("WVB_RESULTS_DIR", "/results", ROOT_DIR / "results")

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_METRICS_PORT = int(os.getenv("WEAVIATE_METRICS_PORT", "2112"))
WEAVIATE_CONTAINER = os.getenv("WEAVIATE_CONTAINER", "wvbench-weaviate")

# Sections of profiles.toml that dotted keys may address.
SECTIONS = ("estimator", "oversize", "schema", "ingest", "timing", "metrics", "check", "run")


class ConfigError(Exception):
    """Invalid or inconsistent configuration (exit code 2)."""


def load_toml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"config file not found: {path}") from e


def coerce(value: str) -> Any:
    """Turn a command-line string into bool / int / float / list / str."""
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [coerce(x) for x in inner.split(";")] if inner else []
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_params(items: list[str] | None) -> dict[str, Any]:
    """Parse ["k=v", ...]. Lists use ';' as separator: sweep_values=[16;32;64]."""
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ConfigError(f"parameter must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = coerce(value)
    return out


def env_params() -> dict[str, Any]:
    raw = os.getenv("WVB_PARAMS", "").strip()
    return parse_params([p for p in raw.split(",") if p.strip()]) if raw else {}


@dataclass
class Layer:
    """One source of settings, kept for the run log."""

    source: str
    values: dict[str, Any]


@dataclass
class Settings:
    """Everything a run needs, after merging all sources."""

    dataset: str
    profile: str
    profile_reason: str
    index: dict[str, Any]
    sections: dict[str, dict[str, Any]]
    layers: list[Layer] = field(default_factory=list)

    def section(self, name: str) -> dict[str, Any]:
        return self.sections[name]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "profile": self.profile,
            "profile_reason": self.profile_reason,
            "index": self.index,
            "sections": self.sections,
            "layers": [{"source": l.source, "values": l.values} for l in self.layers],
        }


def _apply(index: dict, sections: dict, values: dict[str, Any], source: str) -> None:
    for key, value in values.items():
        if key == "profile":
            continue
        if "." in key:
            sec, sub = key.split(".", 1)
            if sec not in SECTIONS:
                raise ConfigError(f"{source}: unknown section {sec!r} in {key!r}")
            if sub not in sections[sec]:
                raise ConfigError(f"{source}: unknown key {key!r}")
            sections[sec][sub] = value
        else:
            if key not in index:
                raise ConfigError(f"{source}: unknown index parameter {key!r}")
            index[key] = value


class ProfileConfig:
    """Parsed profiles.toml with helpers to build a profile's index settings."""

    def __init__(self, raw: dict[str, Any] | None = None):
        self.raw = raw if raw is not None else load_toml("profiles.toml")
        for sec in SECTIONS:
            self.raw.setdefault(sec, {})
        self.defaults: dict[str, Any] = self.raw.get("defaults", {})
        self.profiles: dict[str, dict] = self.raw.get("profiles", {})
        self.rules: list[dict] = self.raw.get("selection", {}).get("rules", [])
        self.overrides: dict[str, dict] = self.raw.get("overrides", {})

    def base_sections(self) -> dict[str, dict[str, Any]]:
        return {sec: copy.deepcopy(self.raw.get(sec, {})) for sec in SECTIONS}

    def profile_index(self, name: str) -> dict[str, Any]:
        if name not in self.profiles:
            raise ConfigError(f"unknown profile {name!r}; available: {', '.join(self.profiles)}")
        index = copy.deepcopy(self.defaults)
        for key, value in self.profiles[name].items():
            if key not in index:
                raise ConfigError(f"profiles.{name}: unknown index parameter {key!r}")
            index[key] = value
        return index


def merge(
    pc: ProfileConfig,
    dataset: str,
    profile: str,
    profile_reason: str,
    cli_params: dict[str, Any],
) -> Settings:
    """Merge defaults -> profile -> dataset overrides -> env -> CLI."""
    index = pc.profile_index(profile)
    sections = pc.base_sections()
    layers = [Layer("defaults", copy.deepcopy(pc.defaults)), Layer(f"profiles.{profile}", pc.profiles[profile])]
    for source, values in (
        (f"overrides.{dataset}", pc.overrides.get(dataset, {})),
        ("WVB_PARAMS", env_params()),
        ("--param", cli_params),
    ):
        if values:
            _apply(index, sections, values, source)
            layers.append(Layer(source, values))
    return Settings(dataset, profile, profile_reason, index, sections, layers)


def manual_profile(pc: ProfileConfig, dataset: str, cli_profile: str | None) -> tuple[str, str] | None:
    """Manual profile choice, if any: CLI > WVB_PROFILE > overrides.<dataset>.profile."""
    if cli_profile:
        return cli_profile, "manual (--profile)"
    if os.getenv("WVB_PROFILE"):
        return os.environ["WVB_PROFILE"], "manual (WVB_PROFILE)"
    pinned = pc.overrides.get(dataset, {}).get("profile")
    if pinned:
        return pinned, f"manual (overrides.{dataset}.profile)"
    return None


def default_dataset() -> str | None:
    return os.getenv("WVB_DATASET") or None
