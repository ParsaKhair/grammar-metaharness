import asyncio
import contextlib
import json
import os
import signal
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import Constraint

BACKENDS = ("claude_code", "codex", "mock")

POST_ONLY_NOTE = (
    "Neither Claude Code nor Codex exposes regex, GBNF, or Lark token-level grammars. "
    "The format is requested in the prompt and checked only after generation."
)


@dataclass
class Generation:
    text: str
    parsed: object = None
    error: str | None = None


PROVIDER_MARKERS = ("model_provider", "profile", "--oss", "--local-provider")


def codex_uses_openai(args: list[str]) -> bool:
    if os.environ.get("OPENAI_BASE_URL") or any(a.startswith("-p") or any(m in a for m in PROVIDER_MARKERS) for a in args):
        return False
    path = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser() / "config.toml"
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, tomllib.TOMLDecodeError):
        return False
    profile = cfg.get("profiles", {}).get(cfg.get("profile"), {}) if cfg.get("profile") else {}
    return profile.get("model_provider", cfg.get("model_provider", "openai")) == "openai"


def enforcement(backend: str, constraint: Constraint, args: list[str] | None = None) -> dict:
    if backend == "mock":
        return {
            "level": "post_validation_only",
            "token_level_guarantee": False,
            "mechanism": "mock backend echoes the prompt; output is validated post-hoc",
            "post_validated": True,
        }
    if constraint.type != "json_schema":
        return {
            "level": "post_validation_only",
            "token_level_guarantee": False,
            "mechanism": "format instruction in prompt + post-hoc re.fullmatch",
            "post_validated": True,
            "note": POST_ONLY_NOTE,
        }
    if backend == "codex" and not codex_uses_openai(args or []):
        return {
            "level": "provider_structured_output",
            "token_level_guarantee": False,
            "mechanism": "codex exec --output-schema",
            "post_validated": True,
            "note": (
                "Codex is configured for a non-default model provider or endpoint, so OpenAI strict "
                "constrained decoding cannot be assumed. The schema is still passed to the provider."
            ),
        }
    if backend == "codex":
        result = {
            "level": "provider_constrained_decoding",
            "token_level_guarantee": True,
            "mechanism": "codex exec --output-schema",
            "post_validated": True,
            "note": (
                "Codex sends the schema to the OpenAI Responses API as a strict json_schema text format, "
                "which OpenAI documents as constrained decoding for schemas in the strict-mode subset. "
                "Holds only when Codex uses the OpenAI provider, and applies to the final message only."
            ),
        }
        issues = strict_issues(constraint.schema)
        if issues:
            result["warnings"] = issues
        return result
    return {
        "level": "provider_structured_output",
        "token_level_guarantee": False,
        "mechanism": "claude -p --output-format json --json-schema",
        "post_validated": True,
        "note": (
            "Claude Code documents --json-schema as 'JSON Schema for structured output validation'. "
            "No token-level decoding guarantee is claimed beyond that."
        ),
    }


def strict_issues(schema, path="<root>") -> list[str]:
    issues = []
    if not isinstance(schema, dict):
        return issues
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            issues.append(f"{path}: OpenAI strict mode requires additionalProperties: false")
        missing = sorted(set(props) - set(schema.get("required", [])))
        if missing:
            issues.append(f"{path}: OpenAI strict mode requires all properties in 'required', missing {missing}")
        for key, sub in props.items():
            issues += strict_issues(sub, f"{path}/{key}")
    if "items" in schema:
        issues += strict_issues(schema["items"], f"{path}/items")
    for keyword in ("anyOf", "allOf", "oneOf"):
        for i, sub in enumerate(schema.get(keyword, [])):
            issues += strict_issues(sub, f"{path}/{keyword}/{i}")
    for keyword in ("$defs", "definitions"):
        for key, sub in schema.get(keyword, {}).items():
            issues += strict_issues(sub, f"{path}/{keyword}/{key}")
    return issues


def compose(prompt: str, constraint: Constraint, feedback: list[str] | None) -> str:
    parts = [prompt]
    if constraint.type == "regex":
        parts.append(
            "Your final response must consist only of text that fully matches this regular expression, "
            f"with no extra prose or code fences:\n{constraint.pattern}"
        )
    if feedback:
        parts.append(
            "Your previous final response failed validation with these errors:\n"
            + "\n".join(f"- {e}" for e in feedback)
            + "\nReturn a corrected final response."
        )
    return "\n\n".join(parts)


async def stop(proc) -> None:
    for sig, grace in ((signal.SIGTERM, 3), (signal.SIGKILL, 5)):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, sig)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), grace)


async def execute(cmd: list[str], stdin: str, cwd: str, timeout: float) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin.encode()), timeout)
    except BaseException:
        await asyncio.shield(stop(proc))
        raise
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def run_claude_code(prompt, constraint, cwd, model, args, timeout) -> Generation:
    cmd = [os.environ.get("CLAUDE_CODE_BIN", "claude"), "-p", "--output-format", "json"]
    if constraint.type == "json_schema":
        cmd += ["--json-schema", json.dumps(constraint.schema)]
    if model:
        cmd += ["--model", model]
    code, out, err = await execute(cmd + args, prompt, cwd, timeout)
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return Generation(text=out, error=f"claude exited {code}: {err.strip()[-2000:] or 'non-JSON output'}")
    structured = data.get("structured_output")
    text = json.dumps(structured) if structured is not None else str(data.get("result", ""))
    error = None
    if code != 0 or data.get("is_error"):
        error = f"claude exited {code}, subtype={data.get('subtype')}: {str(data.get('result', ''))[-2000:]}"
    return Generation(text=text, parsed=structured, error=error)


async def run_codex(prompt, constraint, cwd, model, args, timeout) -> Generation:
    with tempfile.TemporaryDirectory(prefix="grammar-mcp-") as tmp:
        last = Path(tmp) / "last_message.txt"
        cmd = [
            os.environ.get("CODEX_BIN", "codex"), "exec", "--color", "never",
            "--skip-git-repo-check", "-C", cwd, "-o", str(last),
        ]
        if constraint.type == "json_schema":
            schema_path = Path(tmp) / "schema.json"
            schema_path.write_text(json.dumps(constraint.schema), encoding="utf-8")
            cmd += ["--output-schema", str(schema_path)]
        if model:
            cmd += ["-m", model]
        code, out, err = await execute(cmd + args + ["-"], prompt, cwd, timeout)
        text = last.read_text(encoding="utf-8") if last.is_file() else ""
    error = None if code == 0 else f"codex exited {code}: {(err.strip() or out.strip())[-2000:]}"
    return Generation(text=text, error=error)


async def generate(backend, prompt, constraint, cwd, model, args, timeout, feedback=None) -> Generation:
    if backend == "mock":
        return Generation(text=prompt)
    runner = run_claude_code if backend == "claude_code" else run_codex
    full_prompt = compose(prompt, constraint, feedback)
    try:
        return await runner(full_prompt, constraint, cwd, model, args, timeout)
    except FileNotFoundError as exc:
        return Generation(text="", error=f"{backend} binary not found: {exc.filename}")
    except asyncio.TimeoutError:
        return Generation(text="", error=f"{backend} timed out after {timeout}s")
