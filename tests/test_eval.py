from pathlib import Path

import pytest

from harness.eval import EvalAnswer, EvalResult, EvalTask, format_eval_summary, load_eval_tasks


def test_load_eval_tasks_jsonl(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '# comments are allowed\n'
        '{"name":"task","prompt":"do it","success_criteria":"must work","tags":["repo"]}\n',
        encoding="utf-8",
    )

    tasks = load_eval_tasks(path)

    assert tasks == [EvalTask(name="task", prompt="do it", success_criteria="must work", tags=["repo"])]


def test_load_eval_tasks_requires_fields(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"name":"missing prompt"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="name, prompt, and success_criteria"):
        load_eval_tasks(path)


def test_format_eval_summary_includes_scores():
    result = EvalResult(
        task=EvalTask(name="repo-task", prompt="p", success_criteria="s", tags=[]),
        baseline=EvalAnswer(name="baseline", answer="a", score=60),
        harness=EvalAnswer(name="harness", answer="h", score=82),
        reference=EvalAnswer(name="reference", answer="r", score=90),
        verdict="harness +22 over baseline; 8 behind reference",
        created_at="now",
    )

    text = format_eval_summary([result])

    assert "repo-task" in text
    assert "baseline: 60" in text
    assert "harness: 82" in text
    assert "reference: 90" in text
