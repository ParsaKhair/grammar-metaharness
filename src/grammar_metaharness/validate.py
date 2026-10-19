import json
import re

from jsonschema import Draft202012Validator, SchemaError

from .config import Constraint


def check_constraint(constraint: Constraint) -> list[str]:
    try:
        if constraint.type == "regex":
            re.compile(constraint.pattern)
        else:
            Draft202012Validator.check_schema(constraint.schema)
    except SchemaError as exc:
        return [f"constraint {constraint.name!r}: invalid JSON Schema: {exc.message}"]
    except re.error as exc:
        return [f"constraint {constraint.name!r}: invalid regex: {exc}"]
    return []


def validate(constraint: Constraint, output: str, parsed=None) -> tuple[bool, list[str], object]:
    if constraint.type == "regex":
        matched = re.fullmatch(constraint.pattern, output) is not None
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
