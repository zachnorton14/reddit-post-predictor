"""Environment profiles (YAML) and token resolution."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

PACKAGED_ENV_FILE = Path(__file__).with_name("environments.yaml")
LOCAL_ENV_FILE = Path("vcl_environments.yaml")


class ConfigError(Exception):
    """Bad or missing configuration (exit code 2)."""


@dataclass(frozen=True)
class Environment:
    name: str
    image: str
    length_minutes: int
    description: str = ""
    ssh_user: str | None = None
    wait_timeout_minutes: int = 30

    def to_dict(self) -> dict:
        return asdict(self)


def _positive_int(env_name: str, field: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"environment {env_name!r}: {field} must be a positive integer, got {value!r}")
    return value


def env_file_path(path: str | os.PathLike | None = None) -> Path:
    """Pick the profile file: explicit path, $VCL_ENV_FILE, ./vcl_environments.yaml, packaged default."""
    if path:
        return Path(path)
    if os.environ.get("VCL_ENV_FILE"):
        return Path(os.environ["VCL_ENV_FILE"])
    if LOCAL_ENV_FILE.exists():
        return LOCAL_ENV_FILE
    return PACKAGED_ENV_FILE


def load_environments(path: str | os.PathLike | None = None) -> dict[str, Environment]:
    file = env_file_path(path)
    try:
        raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"environment file not found: {file}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {file}: {exc}") from exc

    envs_raw = raw.get("environments") if isinstance(raw, dict) else None
    if not isinstance(envs_raw, dict) or not envs_raw:
        raise ConfigError(f"{file}: expected a top-level 'environments:' mapping")

    known = set(Environment.__dataclass_fields__) - {"name"}
    envs: dict[str, Environment] = {}
    for name, spec in envs_raw.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"environment {name!r}: expected a mapping")
        unknown = set(spec) - known
        if unknown:
            raise ConfigError(f"environment {name!r}: unknown keys {sorted(unknown)}")
        if spec.get("image") in (None, ""):
            raise ConfigError(f"environment {name!r}: 'image' is required")
        if "length_minutes" not in spec:
            raise ConfigError(f"environment {name!r}: 'length_minutes' is required")
        envs[name] = Environment(
            name=name,
            image=str(spec["image"]),
            length_minutes=_positive_int(name, "length_minutes", spec["length_minutes"]),
            description=str(spec.get("description", "")),
            ssh_user=spec.get("ssh_user"),
            wait_timeout_minutes=_positive_int(
                name, "wait_timeout_minutes", spec.get("wait_timeout_minutes", 30)
            ),
        )
    return envs


def get_environment(name: str, path: str | os.PathLike | None = None) -> Environment:
    envs = load_environments(path)
    if name not in envs:
        raise ConfigError(f"unknown environment {name!r}; available: {', '.join(sorted(envs))}")
    return envs[name]


def resolve_token(token_file: str | os.PathLike | None = None) -> str:
    """Token from $VCL_TOKEN, else from a token file. Never logged."""
    token = os.environ.get("VCL_TOKEN", "").strip()
    if token:
        return token
    if token_file:
        try:
            token = Path(token_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ConfigError(f"cannot read token file {token_file}: {exc}") from exc
        if token:
            return token
    raise ConfigError(
        "no VCL API token: set $VCL_TOKEN or pass --token-file "
        "(create one at vcl.ncsu.edu > Manage > User Preferences > Manage Tokens)"
    )
