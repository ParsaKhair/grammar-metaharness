import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONSTRAINT_TYPES = ("json_schema", "regex")


@dataclass
class Constraint:
    name: str
    type: str
    description: str = ""
    schema: dict | None = None
    pattern: str | None = None


@dataclass
class Config:
    path: Path | None
    constraints: dict[str, Constraint] = field(default_factory=dict)
    timeout: float = 900
    backend_args: dict[str, list[str]] = field(default_factory=dict)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


def find_config() -> Path | None:
    override = os.environ.get("GRAMMAR_METAHARNESS_CONFIG")
    if override:
        return Path(override).expanduser()
    for name in ("config.local.toml", "config.toml"):
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    return None


def make_constraint(name: str, raw: dict, base: Path) -> Constraint:
    kind = raw.get("type", "json_schema")
    if kind not in CONSTRAINT_TYPES:
        raise ValueError(f"constraint {name!r}: unsupported type {kind!r}, expected one of {CONSTRAINT_TYPES}")
    constraint = Constraint(name=name, type=kind, description=raw.get("description", ""))
    if kind == "json_schema":
        schema = raw.get("schema")
        if "schema_file" in raw:
            schema = (base / raw["schema_file"]).read_text(encoding="utf-8")
        if isinstance(schema, str):
            schema = json.loads(schema)
        if not isinstance(schema, dict):
            raise ValueError(f"constraint {name!r}: json_schema needs a 'schema' table, JSON string, or 'schema_file'")
        constraint.schema = schema
    else:
        if not isinstance(raw.get("pattern"), str):
            raise ValueError(f"constraint {name!r}: regex needs a 'pattern' string")
        constraint.pattern = raw["pattern"]
    return constraint


def load_config() -> Config:
    load_dotenv(Path.cwd() / ".env")
    path = find_config()
    if path is None:
        return Config(path=None)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    backends = raw.get("backends", {})
    return Config(
        path=path,
        constraints={
            name: make_constraint(name, body, path.parent)
            for name, body in raw.get("constraints", {}).items()
        },
        timeout=float(defaults.get("timeout", 900)),
        backend_args={name: [str(a) for a in body.get("args", [])] for name, body in backends.items()},
    )
