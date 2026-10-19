import json
import re

from jsonschema import Draft202012Validator, SchemaError

from .config import Constraint


def check_schema(schema: dict) -> list[str]:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return [f"invalid JSON Schema: {exc.message}"]
    return []


def validate(constraint: Constraint, output: str, parsed=None) -> tuple[bool, list[str], object]:
    if constraint.type == "regex":
        try:
            matched = re.fullmatch(constraint.pattern, output.strip()) is not None
        except re.error as exc:
            return False, [f"invalid regex: {exc}"], None
        return matched, [] if matched else [f"output does not fully match regex {constraint.pattern!r}"], None
    if parsed is None:
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            return False, [f"output is not valid JSON: {exc}"], None
    errors = [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in Draft202012Validator(constraint.schema).iter_errors(parsed)
    ]
    return not errors, errors, parsed
