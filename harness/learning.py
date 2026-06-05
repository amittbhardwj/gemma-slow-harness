from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import HarnessConfig


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


@dataclass(slots=True)
class LearnedRun:
    id: int
    created_at: str
    task: str
    score: int | None
    success: bool
    lessons: str
    answer: str
    judge: str
    tags: list[str]


@dataclass(slots=True)
class Skill:
    id: int
    name: str
    description: str
    tags: list[str]
    trigger: str
    workflow: str
    prompt_template: str
    verification: str
    source_run_id: int | None
    uses: int
    created_at: str
    updated_at: str


class LearningStore:
    """Persistent memory for self-learning runs and reusable skills.

    Uses SQLite only, so the harness remains dependency-free and small enough for
    a 16 GB Mac. This is not automatic fine-tuning. It is retrieval memory:
    successful patterns, mistakes, verification checks, and saved workflows are
    injected into future prompts as local context.
    """

    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.db_path = cfg.learning_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS runs(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    task TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    judge TEXT NOT NULL,
                    score INTEGER,
                    success INTEGER NOT NULL,
                    lessons TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    critiques_json TEXT NOT NULL,
                    used_skills_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS skills(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    workflow TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    verification TEXT NOT NULL,
                    source_run_id INTEGER,
                    uses INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_runs_score ON runs(score)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")

    def save_run(
        self,
        *,
        task: str,
        plan: str,
        answer: str,
        judge: str,
        score: int | None,
        lessons: str,
        candidates: list[str],
        critiques: list[str],
        used_skills: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> int:
        success = score is not None and score >= self.cfg.auto_learn_min_score
        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO runs(
                    created_at, task, plan, answer, judge, score, success, lessons,
                    candidates_json, critiques_json, used_skills_json, tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    task,
                    plan,
                    answer,
                    judge,
                    score,
                    1 if success else 0,
                    lessons,
                    _json_dumps(candidates),
                    _json_dumps(critiques),
                    _json_dumps(used_skills or []),
                    _json_dumps(tags or []),
                ),
            )
            return int(cur.lastrowid)

    def get_run(self, run_id: int) -> LearnedRun | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def list_runs(self, *, limit: int = 20, successful_only: bool = False) -> list[LearnedRun]:
        where = "WHERE success = 1" if successful_only else ""
        with self._connect() as con:
            rows = con.execute(f"SELECT * FROM runs {where} ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_run(r) for r in rows]

    def search_runs(self, query: str, *, top_k: int | None = None, successful_only: bool = False) -> list[LearnedRun]:
        top_k = top_k or self.cfg.memory_top_k
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return self.list_runs(limit=top_k, successful_only=successful_only)
        clauses = []
        params: list[Any] = []
        for term in terms[:8]:
            like = f"%{term}%"
            clauses.append("(lower(task) LIKE ? OR lower(answer) LIKE ? OR lower(lessons) LIKE ? OR lower(judge) LIKE ?)")
            params.extend([like, like, like, like])
        sql = "SELECT * FROM runs WHERE (" + " OR ".join(clauses) + ")"
        if successful_only:
            sql += " AND success = 1"
        sql += " ORDER BY success DESC, score DESC, id DESC LIMIT ?"
        params.append(top_k)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_run(r) for r in rows]

    def upsert_skill(
        self,
        *,
        name: str,
        description: str,
        trigger: str,
        workflow: str,
        prompt_template: str = "",
        verification: str = "",
        tags: list[str] | None = None,
        source_run_id: int | None = None,
    ) -> int:
        now = _now()
        tags_json = _json_dumps(tags or [])
        with self._connect() as con:
            existing = con.execute("SELECT id, created_at FROM skills WHERE name = ?", (name,)).fetchone()
            if existing:
                con.execute(
                    """
                    UPDATE skills
                    SET description = ?, tags_json = ?, trigger = ?, workflow = ?,
                        prompt_template = ?, verification = ?, source_run_id = ?, updated_at = ?
                    WHERE name = ?
                    """,
                    (
                        description,
                        tags_json,
                        trigger,
                        workflow,
                        prompt_template,
                        verification,
                        source_run_id,
                        now,
                        name,
                    ),
                )
                return int(existing["id"])
            cur = con.execute(
                """
                INSERT INTO skills(
                    name, description, tags_json, trigger, workflow, prompt_template,
                    verification, source_run_id, uses, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (name, description, tags_json, trigger, workflow, prompt_template, verification, source_run_id, now, now),
            )
            return int(cur.lastrowid)

    def get_skill(self, name_or_id: str | int) -> Skill | None:
        with self._connect() as con:
            if isinstance(name_or_id, int) or str(name_or_id).isdigit():
                row = con.execute("SELECT * FROM skills WHERE id = ?", (int(name_or_id),)).fetchone()
            else:
                row = con.execute("SELECT * FROM skills WHERE name = ?", (str(name_or_id),)).fetchone()
        return self._row_to_skill(row) if row else None

    def list_skills(self, *, limit: int = 50) -> list[Skill]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM skills ORDER BY uses DESC, updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def search_skills(self, query: str, *, top_k: int | None = None) -> list[Skill]:
        top_k = top_k or self.cfg.skill_top_k
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return self.list_skills(limit=top_k)
        clauses = []
        params: list[Any] = []
        for term in terms[:8]:
            like = f"%{term}%"
            clauses.append(
                "(lower(name) LIKE ? OR lower(description) LIKE ? OR lower(trigger) LIKE ? OR lower(workflow) LIKE ? OR lower(verification) LIKE ?)"
            )
            params.extend([like, like, like, like, like])
        sql = "SELECT * FROM skills WHERE " + " OR ".join(clauses) + " ORDER BY uses DESC, updated_at DESC LIMIT ?"
        params.append(top_k)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [self._row_to_skill(r) for r in rows]

    def delete_skill(self, name_or_id: str | int) -> bool:
        with self._connect() as con:
            if isinstance(name_or_id, int) or str(name_or_id).isdigit():
                cur = con.execute("DELETE FROM skills WHERE id = ?", (int(name_or_id),))
            else:
                cur = con.execute("DELETE FROM skills WHERE name = ?", (str(name_or_id),))
            return cur.rowcount > 0

    def mark_skill_used(self, name_or_id: str | int) -> None:
        with self._connect() as con:
            if isinstance(name_or_id, int) or str(name_or_id).isdigit():
                con.execute("UPDATE skills SET uses = uses + 1, updated_at = ? WHERE id = ?", (_now(), int(name_or_id)))
            else:
                con.execute("UPDATE skills SET uses = uses + 1, updated_at = ? WHERE name = ?", (_now(), str(name_or_id)))

    def promote_run_to_skill(
        self,
        *,
        run_id: int,
        name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> int:
        run = self.get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        desc = description or f"Reusable workflow learned from run {run_id}."
        trigger = run.task[:600]
        workflow = run.lessons.strip() or run.answer[:4000]
        prompt_template = (
            "Use this saved workflow when the new task is similar.\n\n"
            "Original task pattern:\n{task}\n\n"
            "Required output:\n- assumptions\n- steps\n- verification\n- final recommendation\n"
        )
        verification = run.judge[:2000]
        return self.upsert_skill(
            name=name,
            description=desc,
            trigger=trigger,
            workflow=workflow,
            prompt_template=prompt_template,
            verification=verification,
            tags=tags or run.tags,
            source_run_id=run_id,
        )

    def memory_context_block(self, query: str) -> str:
        runs = self.search_runs(query, top_k=self.cfg.memory_top_k, successful_only=False)
        if not runs:
            return "[LEARNING MEMORY] No previous similar runs found."
        blocks = []
        for r in runs:
            status = "success" if r.success else "needs caution"
            blocks.append(
                f"[MEMORY RUN id={r.id} score={r.score} status={status} created={r.created_at}]\n"
                f"Task: {r.task[:700]}\n"
                f"Lessons:\n{r.lessons[:1600]}\n"
                f"Judge:\n{r.judge[:900]}"
            )
        return "\n\n---\n\n".join(blocks)

    def skills_context_block(self, query: str, *, forced_skill: str | None = None) -> tuple[str, list[str]]:
        if forced_skill:
            skill = self.get_skill(forced_skill)
            if skill is None:
                return f"[SKILLS] Requested skill not found: {forced_skill}", []
            skills = [skill]
        else:
            skills = self.search_skills(query, top_k=self.cfg.skill_top_k)
        if not skills:
            return "[SKILLS] No matching saved skills found.", []
        blocks = []
        names = []
        for s in skills:
            names.append(s.name)
            blocks.append(self.render_skill(s))
        return "\n\n---\n\n".join(blocks), names

    def render_skill(self, skill: Skill) -> str:
        return (
            f"[SKILL id={skill.id} name={skill.name!r} uses={skill.uses}]\n"
            f"Description: {skill.description}\n"
            f"Tags: {', '.join(skill.tags) if skill.tags else 'none'}\n"
            f"Use when: {skill.trigger}\n\n"
            f"Workflow:\n{skill.workflow}\n\n"
            f"Prompt template:\n{skill.prompt_template or 'None'}\n\n"
            f"Verification:\n{skill.verification or 'None'}"
        )

    def stats(self) -> dict[str, int]:
        with self._connect() as con:
            runs = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            successful = con.execute("SELECT COUNT(*) FROM runs WHERE success = 1").fetchone()[0]
            skills = con.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        return {"runs": int(runs), "successful_runs": int(successful), "skills": int(skills)}

    def _row_to_run(self, row: sqlite3.Row) -> LearnedRun:
        return LearnedRun(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            task=str(row["task"]),
            score=int(row["score"]) if row["score"] is not None else None,
            success=bool(row["success"]),
            lessons=str(row["lessons"]),
            answer=str(row["answer"]),
            judge=str(row["judge"]),
            tags=_json_loads(row["tags_json"], []),
        )

    def _row_to_skill(self, row: sqlite3.Row) -> Skill:
        return Skill(
            id=int(row["id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            tags=_json_loads(row["tags_json"], []),
            trigger=str(row["trigger"]),
            workflow=str(row["workflow"]),
            prompt_template=str(row["prompt_template"]),
            verification=str(row["verification"]),
            source_run_id=int(row["source_run_id"]) if row["source_run_id"] is not None else None,
            uses=int(row["uses"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )


