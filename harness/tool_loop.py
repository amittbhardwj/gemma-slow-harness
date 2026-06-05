from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .config import HarnessConfig
from .llm import LocalLLMClient
from .repo_context import load_repo_instructions
from .tools import ToolRegistry, ToolResult

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(slots=True)
class LoopStep:
    thought: str
    tool_calls: list[dict]
    results: list[ToolResult] = field(default_factory=list)
    final: str | None = None


@dataclass(slots=True)
class LoopRun:
    steps: list[LoopStep]
    final: str


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Tool-loop response must be a JSON object.")
    return data


class ToolLoopAgent:
    """A simple structured tool-call loop.

    The loop is intentionally JSON-based so it can work with local models that do
    not support native function calling. It is conservative and bounded.
    """

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.llm = LocalLLMClient(cfg)
        self.tools = ToolRegistry(cfg)

    def run(self, task: str, *, max_steps: int = 6, allowed_tools: list[str] | None = None) -> LoopRun:
        allowed_tools = allowed_tools or [
            "read_file",
            "list_files",
            "search_files",
            "run_python",
            "run_shell",
            "write_file",
        ]
        repo_ctx = load_repo_instructions(self.cfg.workspace).as_context(self.cfg.workspace)
        transcript = ""
        steps: list[LoopStep] = []
        system = "You are a cautious local coding agent. Use tools only when useful. Return JSON only."
        base = f"""
TASK:
{task}

REPOSITORY CONTEXT:
{repo_ctx or 'No AGENTS.md-style instructions found.'}

TOOLS:
{self.tools.tool_manifest()}

Allowed tools for this run: {', '.join(allowed_tools)}

At each step return exactly one JSON object:
{{
  "thought": "brief reasoning",
  "tool_calls": [{{"tool": "read_file", "args": {{"path": "..."}}}}],
  "final": null
}}

When done, return:
{{"thought": "done", "tool_calls": [], "final": "final answer"}}

Rules:
- Do not call tools outside the allowed list.
- Prefer reading/searching before writing.
- Do not write files unless GEMMA_ALLOW_WRITE is enabled.
- Do not run shell commands unless GEMMA_ALLOW_SHELL is enabled.
- Keep each step small.
"""
        final = ""
        for _ in range(max_steps):
            prompt = base + "\n\nPREVIOUS TOOL TRANSCRIPT:\n" + (transcript or "<none>")
            resp = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1400,
            )
            data = _parse_json_object(resp.content)
            calls = data.get("tool_calls") or []
            if not isinstance(calls, list):
                calls = []
            step = LoopStep(thought=str(data.get("thought", "")), tool_calls=calls, final=data.get("final"))
            if step.final:
                final = str(step.final)
                steps.append(step)
                break
            for call in calls[:4]:
                if not isinstance(call, dict):
                    continue
                name = call.get("tool")
                kwargs = call.get("args") or {}
                if name not in allowed_tools:
                    result = ToolResult(name=str(name), ok=False, output=f"Tool not allowed in this loop: {name}")
                else:
                    result = self.tools.call(str(name), **kwargs)
                step.results.append(result)
                transcript += f"\n\nTHOUGHT: {step.thought}\nTOOL CALL: {name} {json.dumps(kwargs)}\n{result.as_context()}"
            steps.append(step)
            if not calls:
                final = "No final answer produced; loop stopped because no tool calls were requested."
                break
        if not final:
            final = "Reached max tool-loop steps before final answer."
        return LoopRun(steps=steps, final=final)
