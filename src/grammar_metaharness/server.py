from pathlib import Path
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from .backends import BACKENDS, enforcement, generate
from .config import Constraint, load_config
from .validate import check_constraint, validate

INSTRUCTIONS = """\
Dispatch Claude Code or Codex as headless subagents whose final answer must satisfy an output constraint \
(a JSON Schema or a regex). Call list_constraints to see the named constraints configured for this \
project and how strongly each backend can enforce them, then call run_constrained. Every result reports \
valid, errors, and an enforcement record: only codex with a JSON Schema uses provider-side constrained \
decoding; claude_code with a JSON Schema uses Claude Code's structured-output option; regex constraints \
are validated after generation only. Pass cwd as the subagent's working directory (for example its worktree). \
Always check valid before trusting output."""

mcp = MCPServer("grammar-metaharness", instructions=INSTRUCTIONS)


def describe(constraint: Constraint, backend_args: dict) -> dict:
    body = {"name": constraint.name, "type": constraint.type, "description": constraint.description}
    if constraint.schema is not None:
        body["schema"] = constraint.schema
    if constraint.pattern is not None:
        body["pattern"] = constraint.pattern
    body["enforcement"] = {
        backend: enforcement(backend, constraint, backend_args.get(backend))["level"] for backend in BACKENDS if backend != "mock"
    }
    return body


@mcp.tool(description="List the named output constraints from config.toml, with the enforcement level each backend provides.")
def list_constraints() -> dict:
    config = load_config()
    return {
        "config_path": str(config.path) if config.path else None,
        "constraints": [describe(c, config.backend_args) for c in config.constraints.values()],
    }


def resolve(config, constraint, schema, pattern) -> Constraint:
    given = [x for x in (constraint, schema, pattern) if x is not None]
    if len(given) != 1:
        raise ValueError("provide exactly one of constraint, schema, or pattern")
    if constraint is not None:
        if constraint not in config.constraints:
            raise ValueError(f"unknown constraint {constraint!r}; known: {sorted(config.constraints)}")
        chosen = config.constraints[constraint]
    elif schema is not None:
        chosen = Constraint(name="<inline>", type="json_schema", schema=schema)
    else:
        chosen = Constraint(name="<inline>", type="regex", pattern=pattern)
    errors = check_constraint(chosen)
    if errors:
        raise ValueError(errors[0])
    return chosen


@mcp.tool(
    description=(
        "Run a Claude Code or Codex subagent headlessly on a prompt and constrain its final answer to a named "
        "constraint, an inline JSON Schema, or an inline regex. Uses the CLI's native structured-output flag "
        "for JSON Schemas, validates every result post-hoc, and returns {output, parsed, valid, errors, "
        "enforcement, attempts, backend_error}. The mock backend echoes the prompt as output for testing."
    )
)
async def run_constrained(
    backend: Annotated[Literal["claude_code", "codex", "mock"], Field(description="Subagent CLI to run")],
    prompt: Annotated[str, Field(description="Task for the subagent")],
    constraint: Annotated[str | None, Field(description="Name of a constraint from list_constraints")] = None,
    schema: Annotated[dict | None, Field(description="Inline JSON Schema, instead of a named constraint")] = None,
    pattern: Annotated[str | None, Field(description="Inline regex the whole answer must match")] = None,
    cwd: Annotated[str | None, Field(description="Working directory for the subagent")] = None,
    model: Annotated[str | None, Field(description="Optional model override passed to the CLI")] = None,
    retries: Annotated[int, Field(ge=0, le=1, description="Re-run once with validation errors fed back")] = 0,
) -> dict:
    try:
        config = load_config()
        chosen = resolve(config, constraint, schema, pattern)
    except (ValueError, OSError) as exc:
        return {"output": None, "parsed": None, "valid": False, "errors": [str(exc)], "enforcement": None}
    workdir = str(Path(cwd or Path.cwd()).expanduser().resolve())
    if not Path(workdir).is_dir():
        return {"output": None, "parsed": None, "valid": False, "errors": [f"cwd does not exist: {workdir}"],
                "enforcement": None}
    args = config.backend_args.get(backend, [])
    feedback = None
    for attempt in range(1, retries + 2):
        result = await generate(backend, prompt, chosen, workdir, model, args, config.timeout, feedback)
        valid, errors, parsed = validate(chosen, result.text, result.parsed)
        if result.error:
            valid, errors = False, [result.error] + errors
        if valid or result.error:
            break
        feedback = errors
    return {
        "output": result.text,
        "parsed": parsed,
        "valid": valid,
        "errors": errors,
        "enforcement": enforcement(backend, chosen, args),
        "constraint": chosen.name,
        "backend": backend,
        "attempts": attempt,
        "backend_error": result.error,
    }


def main() -> None:
    mcp.run()
