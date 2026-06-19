"""
Eval мульти-агента для домашки С6.

Прогоняет 6 вопросов в трёх конфигурациях:
1. одиночный агент С5;
2. PWC без валидатора схемы;
3. PWC с валидатором схемы.

По умолчанию N=5, как в задании. Для быстрой проверки:
    python eval_pwc.py --single
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_s5 import run_agent
from orchestrator import VALID_TOOLS, run_pwc, validate_plan

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass


CASES = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
        "comment": "Арифметика должна идти через calculate.",
        "must_have_keywords": ["usd"],
        "arith_required": True,
        "allow_empty_plan_ok": False,
    },
    {
        "id": "Q2",
        "query": (
            "Какая сейчас реальная ключевая ставка, если инфляцию брать "
            "по последнему доступному месяцу, а не по году?"
        ),
        "comment": "Нужно собрать ставку, инфляцию и посчитать разность.",
        "must_have_keywords": ["%"],
        "arith_required": True,
        "allow_empty_plan_ok": False,
    },
    {
        "id": "Q3",
        "query": (
            "Какова накопленная инфляция с января 2022 по март 2026? "
            "Рассчитай как произведение всех (1 + ипц_м/100) по месяцам."
        ),
        "comment": "Граница паттерна: нельзя выдумывать get_cumulative_inflation.",
        "must_have_keywords": [],
        "arith_required": True,
        "allow_empty_plan_ok": True,
    },
    {
        "id": "Q4",
        "query": (
            "Посчитай среднее значение ключевой ставки между январем 2022 "
            "и апрелем 2026."
        ),
        "comment": (
            "Кейс для валидатора: планировщик склонен выдумывать "
            "get_average_key_rate; валидатор должен заставить перепланировать "
            "или честно отказаться."
        ),
        "must_have_keywords": [],
        "arith_required": True,
        "allow_empty_plan_ok": True,
    },
    {
        "id": "Q5",
        "query": (
            "Сравни одновременно курс USD, EUR и CNY на 1 января 2022 "
            "и на 1 апреля 2026, затем скажи, какая валюта выросла сильнее."
        ),
        "comment": "Естественная параллельность: 3+ независимых валютных подвопроса.",
        "must_have_keywords": [],
        "arith_required": True,
        "allow_empty_plan_ok": False,
    },
    {
        "id": "Q6",
        "query": (
            "Насколько реальная ключевая ставка в апреле 2026 отличается "
            "от реальной ставки в январе 2022?"
        ),
        "comment": "Реальный макро-вопрос: две реальные ставки и разница между ними.",
        "must_have_keywords": ["%"],
        "arith_required": True,
        "allow_empty_plan_ok": False,
    },
]


def _plan_tools(plan) -> set[str]:
    tools: set[str] = set()
    if plan is None:
        return tools
    for sq in plan.subquestions:
        tools.update(sq.expected_tools)
    return tools


def _answer_text(result: dict) -> str:
    return (result.get("answer") or "").lower()


def _empty_plan_is_ok(case: dict, result: dict) -> bool:
    if not case.get("allow_empty_plan_ok"):
        return False
    plan = result.get("plan")
    answer = _answer_text(result)
    if plan is None or plan.subquestions:
        return False
    rejection_markers = ["не решить", "невозможно", "нельзя", "не хватает", "не решается"]
    return any(marker in answer for marker in rejection_markers)


def _check_single(case: dict, result: dict) -> dict:
    used = {e["call"] for e in result.get("trace", []) if "call" in e}
    ans = _answer_text(result)
    hallucinated = used - VALID_TOOLS
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    arith_without_calc = case["arith_required"] and bool(ans) and "calculate" not in used
    ok = bool(ans) and not hallucinated and must and not arith_without_calc
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "hallucinated": sorted(hallucinated),
        "must_have_ok": must,
        "arith_without_calc": arith_without_calc,
        "answer_preview": (result.get("answer") or "")[:220],
    }


def _check_pwc(case: dict, result: dict) -> dict:
    used = set()
    for t in result.get("trace", []):
        if t.get("kind") == "worker":
            used.update(t.get("used_tools") or [])
    plan = result.get("plan")
    plan_tools = _plan_tools(plan)
    hallucinated = used - VALID_TOOLS
    plan_hallucinated = plan_tools - VALID_TOOLS
    ans = _answer_text(result)
    must = all(kw.lower() in ans for kw in case["must_have_keywords"])
    schema_errors = validate_plan(plan) if plan is not None else ["missing_plan"]
    empty_ok = _empty_plan_is_ok(case, result)
    arith_without_calc = (
        case["arith_required"]
        and bool(ans)
        and bool(used)
        and "calculate" not in used
    )
    ok = (
        empty_ok
        or (
            bool(result.get("answer"))
            and not hallucinated
            and not plan_hallucinated
            and not schema_errors
            and must
            and not arith_without_calc
        )
    )
    return {
        "ok": ok,
        "used_tools": sorted(used),
        "plan_tools": sorted(plan_tools),
        "hallucinated_in_workers": sorted(hallucinated),
        "hallucinated_in_plan": sorted(plan_hallucinated),
        "schema_errors": schema_errors,
        "must_have_ok": must,
        "empty_plan_ok": empty_ok,
        "arith_without_calc": arith_without_calc,
        "iterations": result.get("iterations", -1),
        "answer_preview": (result.get("answer") or "")[:220],
    }


def _timed_call(fn):
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as e:
        result = {"answer": None, "error": f"{type(e).__name__}: {e}", "trace": []}
    return result, round(time.perf_counter() - start, 3)


def run_case(case: dict, *, n: int = 5) -> dict:
    configs = {
        "single": {"runs": [], "pass": 0},
        "pwc_no_validator": {"runs": [], "pass": 0},
        "pwc_validator": {"runs": [], "pass": 0},
    }

    for _ in range(n):
        r1, dt1 = _timed_call(lambda: run_agent(case["query"], max_iter=8, verbose=False))
        c1 = _check_single(case, r1)
        c1["elapsed_sec"] = dt1
        configs["single"]["runs"].append(c1)
        configs["single"]["pass"] += int(c1["ok"])

        r2, dt2 = _timed_call(
            lambda: run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                use_validator=False,
                parallel=True,
            )
        )
        c2 = _check_pwc(case, r2)
        c2["elapsed_sec"] = dt2
        configs["pwc_no_validator"]["runs"].append(c2)
        configs["pwc_no_validator"]["pass"] += int(c2["ok"])

        r3, dt3 = _timed_call(
            lambda: run_pwc(
                case["query"],
                max_iter=3,
                verbose=False,
                use_validator=True,
                parallel=True,
            )
        )
        c3 = _check_pwc(case, r3)
        c3["elapsed_sec"] = dt3
        configs["pwc_validator"]["runs"].append(c3)
        configs["pwc_validator"]["pass"] += int(c3["ok"])

    return {
        "id": case["id"],
        "query": case["query"],
        "comment": case["comment"],
        "n": n,
        **configs,
    }


def run_eval(*, n: int = 5) -> list[dict]:
    print(f"Eval С6: {len(CASES)} кейсов x 3 конфигурации x {n} прогонов\n")
    results = []
    for case in CASES:
        print(f"=== {case['id']}: {case['query'][:80]}...")
        r = run_case(case, n=n)
        results.append(r)
        print(
            f"   single {r['single']['pass']}/{n} | "
            f"pwc {r['pwc_no_validator']['pass']}/{n} | "
            f"pwc+validator {r['pwc_validator']['pass']}/{n}"
        )
        print()

    print("=" * 72)
    print("ИТОГО:")
    for r in results:
        print(
            f"  {r['id']}: single {r['single']['pass']}/{n}; "
            f"pwc {r['pwc_no_validator']['pass']}/{n}; "
            f"pwc+validator {r['pwc_validator']['pass']}/{n}"
        )

    out = Path(__file__).parent / "eval_pwc_results.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nРезультаты: {out}")
    return results


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Eval PWC: 6 кейсов x 3 конфигурации.")
    ap.add_argument("--single", action="store_true", help="Быстрый smoke: N=1.")
    ap.add_argument("--n", type=int, default=None, help="Число прогонов на конфиг.")
    args = ap.parse_args()

    n = 1 if args.single else (args.n if args.n is not None else 5)
    run_eval(n=n)


if __name__ == "__main__":
    main()
