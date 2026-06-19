"""
Оркестратор: главный цикл Планировщик-Исполнитель-Критик.

На семинаре нужно:
- реализовать topological_sort (TODO 1),
- реализовать replan/rework-ветки цикла (TODO 2),
- написать synthesize для финального ответа (TODO 3).

Важно: max_iter защищает от бесконечного цикла, если Критик
постоянно говорит «переделай».
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker

VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}


def _invalid_plan_response(
    question: str,
    plan: Plan,
    errors: list[str],
    trace: list[dict[str, Any]],
    *,
    iter_num: int,
) -> dict[str, Any]:
    """Вернуть честный отказ, если план остаётся невалидным после feedback."""
    clean_plan = Plan(
        reasoning=(
            "Такой вопрос не решить текущим набором инструментов без выдуманных "
            f"tools. Ошибки плана: {errors}"
        ),
        subquestions=[],
    )
    trace.append(
        {
            "iter": iter_num,
            "kind": "schema_validator_reject",
            "question": question,
            "errors": errors,
            "invalid_plan": [sq.model_dump() for sq in plan.subquestions],
        }
    )
    return {
        "answer": clean_plan.reasoning,
        "plan": clean_plan,
        "answers": {},
        "trace": trace,
        "iterations": iter_num,
    }


def validate_plan(plan: Plan) -> list[str]:
    """Вернуть список ошибок плана (пустой список означает валидный план)."""
    errors: list[str] = []
    seen: set[int] = set()
    ids = {sq.id for sq in plan.subquestions}

    for sq in plan.subquestions:
        if sq.id in seen:
            errors.append(f"duplicate_subquestion_id:{sq.id}")
        seen.add(sq.id)

        unknown = sorted(set(sq.expected_tools) - VALID_TOOLS)
        if unknown:
            errors.append(f"sq{sq.id}:unknown_tools:{','.join(unknown)}")
        if not sq.expected_tools and sq.question.strip():
            errors.append(f"sq{sq.id}:empty_expected_tools")
        if sq.id in sq.depends_on:
            errors.append(f"sq{sq.id}:self_dependency")
        missing_deps = sorted(dep for dep in sq.depends_on if dep not in ids)
        if missing_deps:
            errors.append(f"sq{sq.id}:missing_dependencies:{missing_deps}")

    try:
        _topological_levels(plan.subquestions)
    except ValueError as e:
        errors.append(f"dependency_error:{e}")

    return errors


def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Разбить подвопросы на уровни: внутри уровня зависимостей нет."""
    by_id = {s.id: s for s in subqs}
    indegree: dict[int, int] = {sq.id: 0 for sq in subqs}
    children: dict[int, list[int]] = {sq.id: [] for sq in subqs}

    for sq in subqs:
        for dep in sq.depends_on:
            if dep not in by_id:
                continue
            indegree[sq.id] += 1
            children[dep].append(sq.id)

    ready = [sq_id for sq_id, degree in indegree.items() if degree == 0]
    levels: list[list[SubQuestion]] = []
    visited = 0

    while ready:
        current_ids = sorted(ready)
        ready = []
        levels.append([by_id[sq_id] for sq_id in current_ids])
        visited += len(current_ids)
        for sq_id in current_ids:
            for child_id in children[sq_id]:
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)

    if visited != len(subqs):
        unresolved = sorted(sq_id for sq_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"цикл в depends_on: {unresolved}")

    return levels


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    """Совместимость со старым API: плоский порядок из уровней."""
    return [sq for level in _topological_levels(subqs) for sq in level]


def execute_level(
    level: list[SubQuestion],
    prev_answers: dict[int, WorkerAnswer],
    *,
    parallel: bool = True,
) -> dict[int, WorkerAnswer]:
    """Прогнать все независимые подвопросы одного уровня."""
    if not level:
        return {}
    if not parallel or len(level) == 1:
        return {sq.id: worker(sq, prev_answers=prev_answers) for sq in level}

    with ThreadPoolExecutor(max_workers=min(4, len(level))) as ex:
        results = list(ex.map(lambda sq: worker(sq, prev_answers=prev_answers), level))
    return {ans.subquestion_id: ans for ans in results}


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    """Собрать финальный ответ одним LLM-вызовом без tools."""
    parts = [f"{i}. {answers[i].answer}" for i in sorted(answers)]
    joined = "\n".join(parts)
    fallback = " · ".join(answers[i].answer for i in sorted(answers))
    try:
        client = make_raw_client()
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Собери ответы исполнителей в 1-2 ясные фразы для пользователя. "
                        "Не добавляй новых чисел и не пересчитывай арифметику."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Исходный вопрос: {question}\n\nОтветы:\n{joined}",
                },
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip() or fallback
    except Exception:
        return fallback


def run_pwc(
    question: str,
    *,
    max_iter: int = 3,
    verbose: bool = True,
    use_validator: bool = True,
    parallel: bool = True,
    critic_temperature: float = 0.7,
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []

    plan = planner(question)
    trace.append(
        {
            "iter": 0,
            "kind": "plan",
            "reasoning": plan.reasoning,
            "subquestions": [sq.model_dump() for sq in plan.subquestions],
        }
    )
    if use_validator:
        errors = validate_plan(plan)
        trace.append({"iter": 0, "kind": "schema_validator", "errors": errors})
        if errors:
            feedback = f"Инструменты не существуют / ошибки плана: {errors}"
            plan = planner(question, feedback=feedback)
            trace.append(
                {
                    "iter": 0,
                    "kind": "replan_after_schema_validator",
                    "feedback": feedback,
                    "reasoning": plan.reasoning,
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            errors = validate_plan(plan)
            trace.append(
                {
                    "iter": 0,
                    "kind": "schema_validator_after_replan",
                    "errors": errors,
                }
            )
            if errors:
                return _invalid_plan_response(
                    question, plan, errors, trace, iter_num=0
                )

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    if not plan.subquestions:
        return {
            "answer": plan.reasoning,
            "plan": plan,
            "answers": {},
            "trace": trace,
            "iterations": 0,
        }

    for iter_num in range(1, max_iter + 1):
        answers: dict[int, WorkerAnswer] = {}
        try:
            levels = _topological_levels(plan.subquestions)
        except ValueError as e:
            plan = planner(question, feedback=f"Ошибка зависимостей в плане: {e}")
            trace.append(
                {
                    "iter": iter_num,
                    "kind": "replan",
                    "reason": str(e),
                    "subquestions": [sq.model_dump() for sq in plan.subquestions],
                }
            )
            continue

        for level_index, level in enumerate(levels, start=1):
            level_answers = execute_level(level, answers, parallel=parallel)
            answers.update(level_answers)
            for sq in level:
                ans = level_answers[sq.id]
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "worker",
                        "level": level_index,
                        "sq_id": sq.id,
                        "used_tools": ans.used_tools,
                        "answer": ans.answer,
                    }
                )
                if verbose:
                    print(f"  [{sq.id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(
            question,
            plan,
            answers,
            temperature=critic_temperature,
        )
        trace.append(
            {
                "iter": iter_num,
                "kind": "verdict",
                "ok": verdict.ok,
                "action": verdict.action,
                "reason": verdict.reason,
                "rework_ids": verdict.rework_ids,
            }
        )

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            feedback = f"Критик отклонил план: {verdict.reason}"
        elif verdict.action == "rework":
            feedback = (
                f"Критик просит переделать подвопросы {verdict.rework_ids}: "
                f"{verdict.reason}"
            )
        else:
            feedback = f"Критик не принял ответ: {verdict.reason}"

        plan = planner(question, feedback=feedback)
        trace.append(
            {
                "iter": iter_num,
                "kind": "replan",
                "feedback": feedback,
                "reasoning": plan.reasoning,
                "subquestions": [sq.model_dump() for sq in plan.subquestions],
            }
        )

        if use_validator:
            errors = validate_plan(plan)
            trace.append(
                {"iter": iter_num, "kind": "schema_validator", "errors": errors}
            )
            if errors:
                plan = planner(
                    question,
                    feedback=f"Инструменты не существуют / ошибки плана: {errors}",
                )
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "replan_after_schema_validator",
                        "errors": errors,
                        "reasoning": plan.reasoning,
                        "subquestions": [sq.model_dump() for sq in plan.subquestions],
                    }
                )
                errors = validate_plan(plan)
                trace.append(
                    {
                        "iter": iter_num,
                        "kind": "schema_validator_after_replan",
                        "errors": errors,
                    }
                )
                if errors:
                    return _invalid_plan_response(
                        question, plan, errors, trace, iter_num=iter_num
                    )
        if not plan.subquestions:
            return {
                "answer": plan.reasoning,
                "plan": plan,
                "answers": {},
                "trace": trace,
                "iterations": iter_num,
            }

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }


def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "Во сколько раз USD подорожал с 1 января 2022 по сегодня?"

    res = run_pwc(q)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")


if __name__ == "__main__":
    main()
