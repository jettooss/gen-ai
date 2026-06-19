"""
Мультиагент: Planner → Workers (параллельно) → Critic.

Агент принимает запрос кандидата и профиль, использует tools:
  - search_vacancies(query, top_k)
  - score_match(profile, vacancy_id)
  - get_skills_gap(profile, vacancy_ids)

Возвращает финальный ответ с топ-вакансиями и анализом пробелов навыков.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from llm_client import get_model, make_client, make_raw_client
from rag import build_index, load_vacancies, search
from schemas import AgentVerdict, CandidateProfile, MatchResult

_VACANCIES: list[dict] | None = None
_BM25 = None


def _get_corpus():
    global _VACANCIES, _BM25
    if _VACANCIES is None:
        _VACANCIES = load_vacancies()
        _BM25 = build_index(_VACANCIES)
    return _VACANCIES, _BM25


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def search_vacancies(query: str, top_k: int = 5) -> list[dict]:
    vacancies, bm25 = _get_corpus()
    hits = search(query, vacancies, bm25, top_k=top_k)
    return [
        {
            "id": v["id"],
            "title": v["title"],
            "company": v["company"],
            "salary_min": v.get("salary_min"),
            "salary_max": v.get("salary_max"),
            "currency": v.get("currency", "RUR"),
            "experience_years": v.get("experience_years", 0),
            "required_skills": v.get("required_skills", [])[:10],
        }
        for v in hits
    ]


def score_match(profile: dict, vacancy_id: str) -> dict:
    vacancies, _ = _get_corpus()
    vac = next((v for v in vacancies if str(v["id"]) == str(vacancy_id)), None)
    if not vac:
        return {"error": f"vacancy {vacancy_id} not found"}

    client = make_client()
    skills_str = ", ".join(profile.get("skills", []))
    req_str = ", ".join(vac.get("required_skills", [])[:15])

    try:
        result = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты оцениваешь соответствие кандидата вакансии. "
                        "score от 1 (совсем не подходит) до 10 (идеально). "
                        "matched_skills — навыки из профиля, которые требует вакансия. "
                        "missing_skills — что требует вакансия, но нет у кандидата."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Кандидат: навыки: {skills_str}, "
                        f"опыт: {profile.get('experience_years', 0)} лет\n"
                        f"Вакансия: {vac['title']} ({vac['company']})\n"
                        f"Требования: {req_str}"
                    ),
                },
            ],
            response_model=MatchResult,
            temperature=0.0,
        )
        return result.model_dump()
    except Exception as e:
        return {"error": str(e), "vacancy_id": vacancy_id, "score": 1}


def get_skills_gap(profile: dict, vacancy_ids: list[str]) -> dict:
    vacancies, _ = _get_corpus()
    id_set = {str(v) for v in vacancy_ids}
    matched = [v for v in vacancies if str(v["id"]) in id_set]

    required: set[str] = set()
    for v in matched:
        for s in v.get("required_skills", []):
            required.add(s.lower())

    candidate_skills = {s.lower() for s in profile.get("skills", [])}
    gap = sorted(required - candidate_skills)
    return {"missing_skills": gap[:20], "total_required": len(required)}


TOOLS = {
    "search_vacancies": search_vacancies,
    "score_match": score_match,
    "get_skills_gap": get_skills_gap,
}

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """Ты планировщик задач. Разбей запрос на 2-4 шага.
Доступные инструменты:
  - search_vacancies(query, top_k=5) — найти вакансии по запросу
  - score_match(profile, vacancy_id) — оценить соответствие кандидата вакансии
  - get_skills_gap(profile, vacancy_ids) — найти недостающие навыки

Верни план как JSON: {"steps": [{"id":1,"tool":"...","args":{...},"depends_on":[],"description":"..."}]}"""


def plan(query: str, profile: CandidateProfile, client) -> list[dict]:
    profile_str = json.dumps(profile.model_dump(), ensure_ascii=False)
    resp = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Запрос: {query}\nПрофиль: {profile_str}"},
        ],
        response_model=list[dict],
        temperature=0.0,
    )
    return resp if isinstance(resp, list) else []


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------

def _run_step(step: dict, profile: CandidateProfile, prev_results: dict) -> dict:
    tool_name = step.get("tool", "")
    args = dict(step.get("args", {}))

    if "profile" in args:
        args["profile"] = profile.model_dump()

    if "vacancy_ids" in args and not args["vacancy_ids"]:
        all_ids = []
        for r in prev_results.values():
            if isinstance(r, list):
                all_ids.extend(item.get("id", "") for item in r if isinstance(item, dict))
        args["vacancy_ids"] = all_ids[:5]

    if tool_name not in TOOLS:
        return {"error": f"unknown tool: {tool_name}"}

    try:
        return TOOLS[tool_name](**args)
    except Exception as e:
        return {"error": str(e)}


def execute_steps(steps: list[dict], profile: CandidateProfile) -> dict[int, Any]:
    results: dict[int, Any] = {}
    by_id = {s["id"]: s for s in steps}

    while len(results) < len(steps):
        ready = [
            s for s in steps
            if s["id"] not in results
            and all(dep in results for dep in s.get("depends_on", []))
        ]
        if not ready:
            break
        if len(ready) == 1:
            step = ready[0]
            results[step["id"]] = _run_step(step, profile, results)
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(ready))) as ex:
                futures = {ex.submit(_run_step, s, profile, results): s["id"] for s in ready}
                for fut, sid in futures.items():
                    results[sid] = fut.result()

    return results


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """Ты критик качества ответов на вопросы о вакансиях.
Проверь: отвечает ли финальный ответ на исходный вопрос,
есть ли конкретные вакансии с пояснением, указаны ли пробелы навыков.
score: 1–10. ok=true если score >= 7."""


def criticize(query: str, answer: str, client) -> AgentVerdict:
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": f"Вопрос: {query}\n\nОтвет агента:\n{answer}"},
        ],
        response_model=AgentVerdict,
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------

def synthesize(query: str, step_results: dict, profile: CandidateProfile) -> str:
    parts = []
    for sid, result in sorted(step_results.items()):
        if isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "title" in item:
                    skills = ", ".join(item.get("required_skills", [])[:5])
                    parts.append(f"- [{item['id']}] {item['title']} ({item['company']}): {skills}")
        elif isinstance(result, dict):
            if "score" in result:
                parts.append(
                    f"- Вакансия {result.get('vacancy_id','?')}: оценка {result['score']}/10, "
                    f"совпадает: {', '.join(result.get('matched_skills',[])[:4])}, "
                    f"не хватает: {', '.join(result.get('missing_skills',[])[:4])}"
                )
            elif "missing_skills" in result:
                gaps = result["missing_skills"][:10]
                parts.append(f"- Пробелы в навыках: {', '.join(gaps)}")

    context = "\n".join(parts) or "(нет данных)"
    client = make_raw_client()
    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Напиши краткий структурированный ответ (3-5 предложений) "
                        "на основе результатов поиска вакансий. "
                        "Укажи топ-3 вакансии, оценку совпадения и чему стоит научиться."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Запрос: {query}\n\nРезультаты:\n{context}",
                },
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return context


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_agent(
    query: str,
    profile: CandidateProfile,
    *,
    max_iter: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    client = make_client()

    if verbose:
        print(f"\n[agent] запрос: {query}")

    steps_plan = [
        {"id": 1, "tool": "search_vacancies", "args": {"query": query, "top_k": 5}, "depends_on": [], "description": "Найти подходящие вакансии"},
        {"id": 2, "tool": "score_match", "args": {"profile": profile.model_dump(), "vacancy_id": "__top1__"}, "depends_on": [1], "description": "Оценить совпадение"},
        {"id": 3, "tool": "get_skills_gap", "args": {"profile": profile.model_dump(), "vacancy_ids": []}, "depends_on": [1], "description": "Найти пробелы навыков"},
    ]

    trace: list[dict] = []
    final_answer = ""
    verdict = None

    for iteration in range(1, max_iter + 1):
        step_results = execute_steps(steps_plan, profile)

        top_vacancies = step_results.get(1, [])
        if isinstance(top_vacancies, list) and top_vacancies:
            top_id = str(top_vacancies[0].get("id", ""))
            for step in steps_plan:
                if step["tool"] == "score_match":
                    step["args"]["vacancy_id"] = top_id

            step_results = execute_steps(steps_plan, profile)

        if verbose:
            for sid, res in sorted(step_results.items()):
                print(f"  [step {sid}] {str(res)[:120]}")

        final_answer = synthesize(query, step_results, profile)

        try:
            verdict = criticize(query, final_answer, client)
        except Exception as e:
            verdict = AgentVerdict(ok=True, score=7, reasoning=f"critic error: {e}")

        trace.append({
            "iteration": iteration,
            "verdict_ok": verdict.ok,
            "verdict_score": verdict.score,
            "verdict_reasoning": verdict.reasoning,
        })

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] score={verdict.score}: {verdict.reasoning[:80]}")

        if verdict.ok:
            break

    tools_used = [s["tool"] for s in steps_plan]

    return {
        "answer": final_answer,
        "verdict": verdict.model_dump() if verdict else {},
        "trace": trace,
        "iterations": len(trace),
        "tools_used": tools_used,
    }


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))

    profile = CandidateProfile(
        skills=["Python", "pandas", "scikit-learn", "SQL"],
        experience_years=2.0,
        desired_salary=180000,
        employment_type=["full"],
    )
    result = run_agent(
        "Ищу работу Data Scientist с зарплатой от 180 тысяч",
        profile,
        verbose=True,
    )
    print("\n=== ОТВЕТ ===")
    print(result["answer"])
