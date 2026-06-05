from __future__ import annotations

import re
from dataclasses import dataclass

from .config import HarnessConfig
from .learning import LearningStore
from .llm import LocalLLMClient, Message
from .memory_policy import compact_context_block
from .prompts import CANDIDATE, CRITIC, JUDGE, LESSON_EXTRACTOR, PLANNER, REFINER, SLOW_SYSTEM
from .rag import RagStore
from .repo_context import load_repo_instructions
from .tools import ToolRegistry


_SCORE_RE = re.compile(r"SCORE:\s*(\d{1,3})", re.IGNORECASE)


@dataclass(slots=True)
class AgentRun:
    answer: str
    plan: str
    candidates: list[str]
    critiques: list[str]
    judge: str
    score: int | None
    lessons: str = ""
    run_id: int | None = None
    used_skills: list[str] | None = None


class SlowHarnessAgent:
    def __init__(self, cfg: HarnessConfig) -> None:
        self.cfg = cfg
        self.llm = LocalLLMClient(cfg)
        self.tools = ToolRegistry(cfg)
        self.rag = RagStore(cfg)
        self.learning = LearningStore(cfg)

    def _messages(self, instruction: str, user: str, extra: str = "") -> list[Message]:
        content = f"{instruction}\n\nUSER TASK:\n{user}"
        if extra.strip():
            content += f"\n\nCONTEXT:\n{extra}"
        return [
            {"role": "system", "content": SLOW_SYSTEM},
            {"role": "user", "content": content},
        ]

    def plan(self, task: str, context: str = "") -> str:
        return self.llm.chat(self._messages(PLANNER, task, context), temperature=0.2).content

    def generate_candidate(self, task: str, plan: str, context: str, idx: int) -> str:
        extra = f"PLAN:\n{plan}\n\nAVAILABLE TOOLS:\n{self.tools.tool_manifest()}\n\n{context}\n\nCandidate number: {idx}"
        return self.llm.chat(self._messages(CANDIDATE, task, extra), temperature=min(0.9, self.cfg.temperature + 0.12 * idx)).content

    def critique(self, task: str, plan: str, candidate: str, context: str) -> str:
        extra = f"PLAN:\n{plan}\n\nCANDIDATE ANSWER:\n{candidate}\n\nEVIDENCE/CONTEXT:\n{context}"
        return self.llm.chat(self._messages(CRITIC, task, extra), temperature=0.15).content

    def refine(self, task: str, plan: str, candidate: str, critiques: list[str], context: str) -> str:
        extra = (
            f"PLAN:\n{plan}\n\nCANDIDATE ANSWER:\n{candidate}\n\n"
            f"CRITIQUES:\n" + "\n\n---\n\n".join(critiques) + f"\n\nEVIDENCE/CONTEXT:\n{context}"
        )
        return self.llm.chat(self._messages(REFINER, task, extra), temperature=0.2).content

    def judge(self, task: str, answer: str, context: str) -> tuple[str, int | None]:
        extra = f"FINAL ANSWER:\n{answer}\n\nEVIDENCE/CONTEXT:\n{context}"
        text = self.llm.chat(self._messages(JUDGE, task, extra), temperature=0.0, max_tokens=800).content
        m = _SCORE_RE.search(text)
        score = int(m.group(1)) if m else None
        return text, score

    def extract_lessons(
        self,
        *,
        task: str,
        plan: str,
        answer: str,
        judge: str,
        score: int | None,
        critiques: list[str],
        used_skills: list[str],
    ) -> str:
        extra = (
            f"PLAN:\n{plan}\n\nFINAL ANSWER:\n{answer}\n\nJUDGE SCORE: {score}\n\n"
            f"JUDGE FEEDBACK:\n{judge}\n\nUSED SKILLS: {', '.join(used_skills) if used_skills else 'None'}\n\n"
            f"CRITIQUES:\n" + "\n\n---\n\n".join(critiques[-6:])
        )
        return self.llm.chat(self._messages(LESSON_EXTRACTOR, task, extra), temperature=0.1, max_tokens=1200).content

    def run(
        self,
        task: str,
        *,
        use_rag: bool = True,
        use_learning: bool | None = None,
        use_skills: bool = True,
        forced_skill: str | None = None,
        extra_context: str = "",
        candidates: int | None = None,
        debate_rounds: int | None = None,
        save_learning: bool | None = None,
    ) -> AgentRun:
        candidates = candidates or self.cfg.candidates
        debate_rounds = debate_rounds or self.cfg.debate_rounds
        use_learning = self.cfg.learning_enabled if use_learning is None else use_learning
        save_learning = self.cfg.learning_enabled if save_learning is None else save_learning

        context_parts = []
        repo_instructions = load_repo_instructions(self.cfg.workspace).as_context(self.cfg.workspace)
        if repo_instructions.strip():
            context_parts.append(repo_instructions)
        if extra_context.strip():
            context_parts.append(extra_context)
        if use_rag:
            context_parts.append(self.rag.context_block(task))

        used_skills: list[str] = []
        if use_learning:
            if use_skills:
                skills_context, used_skills = self.learning.skills_context_block(task, forced_skill=forced_skill)
                context_parts.append(skills_context)
                for skill_name in used_skills:
                    self.learning.mark_skill_used(skill_name)
            context_parts.append(self.learning.memory_context_block(task))

        context = "\n\n".join(x for x in context_parts if x.strip())
        context = compact_context_block(context, self.cfg)

        plan = self.plan(task, context)
        candidate_texts = [self.generate_candidate(task, plan, context, i + 1) for i in range(candidates)]

        # Critique all candidates once, pick/refine the candidate with strongest self-consistency.
        critiques: list[str] = []
        refined_answers: list[str] = []
        for cand in candidate_texts:
            local_critiques = []
            current = cand
            for _ in range(debate_rounds):
                crit = self.critique(task, plan, current, context)
                critiques.append(crit)
                local_critiques.append(crit)
                current = self.refine(task, plan, current, local_critiques, context)
            refined_answers.append(current)

        synthesis_context = context + "\n\nREFINED ANSWERS:\n" + "\n\n===\n\n".join(refined_answers)
        final = self.llm.chat(
            self._messages(
                "Synthesize the best final answer from the refined candidates. Keep only defensible, useful content.",
                task,
                synthesis_context,
            ),
            temperature=0.15,
        ).content
        judge_text, score = self.judge(task, final, context)

        # One repair pass if judge says weak.
        if score is not None and score < self.cfg.require_final_judge_score:
            final = self.llm.chat(
                self._messages(
                    "Repair the final answer based on the judge feedback. Return the corrected final answer only.",
                    task,
                    f"PREVIOUS FINAL:\n{final}\n\nJUDGE FEEDBACK:\n{judge_text}\n\nCONTEXT:\n{context}",
                ),
                temperature=0.1,
            ).content
            judge_text, score = self.judge(task, final, context)

        lessons = ""
        run_id: int | None = None
        if save_learning:
            lessons = self.extract_lessons(
                task=task,
                plan=plan,
                answer=final,
                judge=judge_text,
                score=score,
                critiques=critiques,
                used_skills=used_skills,
            )
            run_id = self.learning.save_run(
                task=task,
                plan=plan,
                answer=final,
                judge=judge_text,
                score=score,
                lessons=lessons,
                candidates=candidate_texts,
                critiques=critiques,
                used_skills=used_skills,
            )

        return AgentRun(
            answer=final,
            plan=plan,
            candidates=candidate_texts,
            critiques=critiques,
            judge=judge_text,
            score=score,
            lessons=lessons,
            run_id=run_id,
            used_skills=used_skills,
        )
