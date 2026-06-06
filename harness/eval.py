from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agent import SlowHarnessAgent
from .config import HarnessConfig
from .llm import LocalLLMClient


@dataclass(slots=True)
class EvalTask:
    name: str
    prompt: str
    success_criteria: str
    tags: list[str]


@dataclass(slots=True)
class EvalAnswer:
    name: str
    answer: str
    elapsed_sec: float | None = None
    score: int | None = None
    judge: str = ""


@dataclass(slots=True)
class EvalResult:
    task: EvalTask
    baseline: EvalAnswer
    harness: EvalAnswer
    reference: EvalAnswer | None
    verdict: str
    created_at: str


def load_eval_tasks(path: Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: expected JSON object") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object")
        name = str(raw.get("name") or "").strip()
        prompt = str(raw.get("prompt") or "").strip()
        success_criteria = str(raw.get("success_criteria") or "").strip()
        tags_raw = raw.get("tags") or []
        if not name or not prompt or not success_criteria:
            raise ValueError(f"{path}:{lineno}: name, prompt, and success_criteria are required")
        if not isinstance(tags_raw, list):
            raise ValueError(f"{path}:{lineno}: tags must be a JSON array")
        tasks.append(EvalTask(name=name, prompt=prompt, success_criteria=success_criteria, tags=[str(t) for t in tags_raw]))
    if not tasks:
        raise ValueError(f"No eval tasks found in {path}")
    return tasks


def write_eval_results(path: Path, results: list[EvalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(_result_to_json(result), ensure_ascii=False) + "\n")


def format_eval_summary(results: list[EvalResult]) -> str:
    lines = ["# Frontier Approximation Eval", ""]
    for result in results:
        ref_score = result.reference.score if result.reference else None
        lines.append(f"## {result.task.name}")
        lines.append(f"- baseline: {result.baseline.score}")
        lines.append(f"- harness: {result.harness.score}")
        if result.reference:
            lines.append(f"- reference: {ref_score}")
        lines.append(f"- verdict: {result.verdict}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class FrontierEvalRunner:
    """Compare one-shot local answers with harness answers and an optional reference."""

    def __init__(self, cfg: HarnessConfig, *, reference_cfg: HarnessConfig | None = None) -> None:
        self.cfg = cfg
        self.reference_cfg = reference_cfg
        self.local_client = LocalLLMClient(cfg)
        self.harness_agent = SlowHarnessAgent(cfg)
        self.reference_client = LocalLLMClient(reference_cfg) if reference_cfg else None

    def run_task(self, task: EvalTask, *, use_rag: bool = True, use_learning: bool = False) -> EvalResult:
        baseline_resp = self.local_client.chat(
            [
                {"role": "system", "content": "Answer directly in one pass. Be concise and practical."},
                {"role": "user", "content": task.prompt},
            ],
            temperature=0.2,
        )
        baseline = EvalAnswer(name="baseline", answer=baseline_resp.content, elapsed_sec=baseline_resp.elapsed_sec)

        harness_run = self.harness_agent.run(
            task.prompt,
            use_rag=use_rag,
            use_learning=use_learning,
            use_skills=use_learning,
            save_learning=False,
        )
        harness = EvalAnswer(name="harness", answer=harness_run.answer, score=harness_run.score, judge=harness_run.judge)

        reference = None
        if self.reference_client is not None:
            reference_resp = self.reference_client.chat(
                [
                    {"role": "system", "content": "You are the reference frontier coding assistant. Answer with the best practical solution."},
                    {"role": "user", "content": task.prompt},
                ],
                temperature=0.1,
            )
            reference = EvalAnswer(name="reference", answer=reference_resp.content, elapsed_sec=reference_resp.elapsed_sec)

        self._score_answers(task, baseline, harness, reference)
        verdict = self._verdict(baseline, harness, reference)
        return EvalResult(
            task=task,
            baseline=baseline,
            harness=harness,
            reference=reference,
            verdict=verdict,
            created_at=datetime.now(UTC).isoformat(),
        )

    def run(self, tasks: list[EvalTask], *, use_rag: bool = True, use_learning: bool = False) -> list[EvalResult]:
        return [self.run_task(task, use_rag=use_rag, use_learning=use_learning) for task in tasks]

    def _score_answers(self, task: EvalTask, baseline: EvalAnswer, harness: EvalAnswer, reference: EvalAnswer | None) -> None:
        answers = [baseline, harness] + ([reference] if reference else [])
        for answer in answers:
            if answer is None:
                continue
            if answer.score is not None and answer.judge:
                continue
            judge_text, score = self._judge_answer(task, answer.answer)
            answer.judge = judge_text
            answer.score = score

    def _judge_answer(self, task: EvalTask, answer: str) -> tuple[str, int | None]:
        prompt = f"""
Evaluate this answer against the task and success criteria.

TASK:
{task.prompt}

SUCCESS CRITERIA:
{task.success_criteria}

ANSWER:
{answer}

Score from 0 to 100 for correctness, usefulness, verification discipline, safety, and specificity.
Return exactly:
SCORE: <integer>
REASON: <one paragraph>
"""
        text = self.local_client.chat(
            [{"role": "system", "content": "You are a strict local coding eval judge."}, {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=700,
        ).content
        return text, _parse_score(text)

    @staticmethod
    def _verdict(baseline: EvalAnswer, harness: EvalAnswer, reference: EvalAnswer | None) -> str:
        if baseline.score is None or harness.score is None:
            return "inconclusive"
        delta = harness.score - baseline.score
        if reference and reference.score is not None:
            gap = reference.score - harness.score
            if gap <= 5:
                return f"harness near reference; +{delta} over baseline"
            return f"harness +{delta} over baseline; {gap} behind reference"
        if delta >= 10:
            return f"harness improves baseline by {delta}"
        if delta <= -10:
            return f"harness regresses baseline by {-delta}"
        return f"harness roughly matches baseline; delta {delta}"


def _parse_score(text: str) -> int | None:
    for line in text.splitlines():
        if line.upper().startswith("SCORE:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return max(0, min(100, int(raw)))
            except ValueError:
                return None
    return None


def _result_to_json(result: EvalResult) -> dict[str, Any]:
    data = asdict(result)
    return data
