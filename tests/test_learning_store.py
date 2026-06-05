from pathlib import Path

from harness.config import HarnessConfig
from harness.learning import LearningStore


def cfg_for(tmp_path: Path) -> HarnessConfig:
    return HarnessConfig(
        workspace=tmp_path,
        rag_db_path=tmp_path / ".gemma_harness/rag.sqlite3",
        learning_db_path=tmp_path / ".gemma_harness/learning.sqlite3",
    )


def test_save_and_search_run(tmp_path: Path):
    store = LearningStore(cfg_for(tmp_path))
    run_id = store.save_run(
        task="debug flask import error",
        plan="inspect imports",
        answer="fix package imports",
        judge="SCORE: 90",
        score=90,
        lessons="Always check cwd and package roots.",
        candidates=["a"],
        critiques=["b"],
    )
    assert run_id == 1
    hits = store.search_runs("flask package root")
    assert hits
    assert hits[0].id == run_id
    assert hits[0].success is True


def test_skill_upsert_and_promote(tmp_path: Path):
    store = LearningStore(cfg_for(tmp_path))
    run_id = store.save_run(
        task="create casting cost ML pipeline",
        plan="plan",
        answer="answer",
        judge="SCORE: 88",
        score=88,
        lessons="Use train/test split and leakage checks.",
        candidates=[],
        critiques=[],
        tags=["ml", "casting"],
    )
    skill_id = store.promote_run_to_skill(run_id=run_id, name="casting-cost-ml")
    assert skill_id == 1
    skill = store.get_skill("casting-cost-ml")
    assert skill is not None
    assert "leakage" in skill.workflow.lower()
    assert "casting" in skill.tags
    assert store.search_skills("cost pipeline")
