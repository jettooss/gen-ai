"""
Оценка системы: 15 тест-кейсов, LLM-as-judge (1-10), pass-rate ≥ 7.

Запуск: python eval.py
Выход:  output/eval_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from agent import run_agent
from llm_client import get_model, make_client
from rag import answer_with_citations, build_index, load_vacancies, search
from schemas import AgentVerdict, CandidateProfile

OUTPUT = Path(__file__).parent / "output" / "eval_results.json"

TEST_CASES = [
    {
        "case_id": 1,
        "query": "Ищу работу Python-разработчика с опытом ML",
        "profile": CandidateProfile(skills=["Python", "scikit-learn", "pandas", "SQL"], experience_years=2.0, desired_salary=150000),
        "mode": "agent",
    },
    {
        "case_id": 2,
        "query": "Data Scientist с опытом NLP и трансформерами",
        "profile": CandidateProfile(skills=["Python", "transformers", "PyTorch", "NLP", "BERT"], experience_years=3.0, desired_salary=200000),
        "mode": "agent",
    },
    {
        "case_id": 3,
        "query": "Junior Data Engineer — ETL, Spark, Airflow",
        "profile": CandidateProfile(skills=["Python", "SQL", "Airflow", "Spark"], experience_years=1.0, desired_salary=100000),
        "mode": "agent",
    },
    {
        "case_id": 4,
        "query": "ML Engineer, деплой моделей, MLflow, Docker",
        "profile": CandidateProfile(skills=["Python", "Docker", "MLflow", "FastAPI", "scikit-learn"], experience_years=2.5, desired_salary=180000),
        "mode": "agent",
    },
    {
        "case_id": 5,
        "query": "Аналитик данных, Excel/Power BI, SQL",
        "profile": CandidateProfile(skills=["SQL", "Excel", "Power BI", "Python"], experience_years=1.5, desired_salary=90000),
        "mode": "agent",
    },
    {
        "case_id": 6,
        "query": "Senior Python backend, FastAPI, PostgreSQL, микросервисы",
        "profile": CandidateProfile(skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"], experience_years=5.0, desired_salary=250000),
        "mode": "agent",
    },
    {
        "case_id": 7,
        "query": "CV-инженер, детекция объектов, YOLO",
        "profile": CandidateProfile(skills=["Python", "PyTorch", "OpenCV", "YOLO", "NumPy"], experience_years=2.0, desired_salary=170000),
        "mode": "agent",
    },
    {
        "case_id": 8,
        "query": "Research scientist, LLM fine-tuning, RL",
        "profile": CandidateProfile(skills=["Python", "PyTorch", "transformers", "RLHF", "CUDA"], experience_years=4.0, desired_salary=300000),
        "mode": "agent",
    },
    {
        "case_id": 9,
        "query": "Data Analyst для продуктовой компании",
        "profile": CandidateProfile(skills=["SQL", "Python", "Tableau", "A/B testing", "statistics"], experience_years=2.0, desired_salary=120000),
        "mode": "agent",
    },
    {
        "case_id": 10,
        "query": "Разработчик рекомендательных систем",
        "profile": CandidateProfile(skills=["Python", "collaborative filtering", "Spark", "SQL"], experience_years=3.0, desired_salary=200000),
        "mode": "agent",
    },
    {
        "case_id": 11,
        "query": "Какие вакансии требуют знание Kubernetes?",
        "profile": CandidateProfile(skills=["Python", "Kubernetes", "Docker", "CI/CD"], experience_years=3.0),
        "mode": "rag",
    },
    {
        "case_id": 12,
        "query": "Вакансии с зарплатой выше 200 тысяч для ML-инженера",
        "profile": CandidateProfile(skills=["Python", "ML", "PyTorch"], experience_years=3.0, desired_salary=200000),
        "mode": "rag",
    },
    {
        "case_id": 13,
        "query": "Где нужен опыт с облаками AWS или GCP?",
        "profile": CandidateProfile(skills=["Python", "AWS", "GCP", "Terraform"], experience_years=2.0),
        "mode": "rag",
    },
    {
        "case_id": 14,
        "query": "Junior позиции для выпускника с Python и ML",
        "profile": CandidateProfile(skills=["Python", "scikit-learn", "Jupyter"], experience_years=0.0, desired_salary=70000),
        "mode": "agent",
    },
    {
        "case_id": 15,
        "query": "Компании, которые ищут специалистов по NLP и обработке текста",
        "profile": CandidateProfile(skills=["Python", "NLP", "spaCy", "BERT"], experience_years=2.0, desired_salary=160000),
        "mode": "rag",
    },
]

EVAL_JUDGE_SYSTEM = """Ты оцениваешь качество ответа системы поиска вакансий.
score 1-10:
  10 — идеальный ответ: конкретные вакансии, объяснение совпадения, анализ навыков
  7-9 — хороший ответ: есть вакансии и объяснение
  4-6 — частичный ответ: есть вакансии, но без объяснения
  1-3 — плохой ответ: нет вакансий или не по теме
ok=true если score >= 7."""


def judge_answer(query: str, answer: str, client) -> AgentVerdict:
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": EVAL_JUDGE_SYSTEM},
            {"role": "user", "content": f"Вопрос: {query}\n\nОтвет системы:\n{answer}"},
        ],
        response_model=AgentVerdict,
        temperature=0.7,
    )


def run_eval() -> list[dict]:
    vacancies = load_vacancies()
    bm25 = build_index(vacancies)
    client = make_client()

    results = []
    passed = 0

    print(f"Запускаю eval: {len(TEST_CASES)} кейсов\n")

    for case in TEST_CASES:
        cid = case["case_id"]
        query = case["query"]
        profile = case["profile"]
        mode = case.get("mode", "agent")

        print(f"[{cid:>2}] {query[:60]}", end=" ")
        t0 = time.time()

        try:
            if mode == "rag":
                hits = search(query, vacancies, bm25, top_k=5)
                aw = answer_with_citations(query, hits, client)
                answer = aw.answer
                steps = 1
                tools_used = ["bm25_search", "llm_answer_gen"]
            else:
                res = run_agent(query, profile, verbose=False)
                answer = res["answer"]
                steps = res["iterations"]
                tools_used = res.get("tools_used", [])

            verdict = judge_answer(query, answer, client)
            elapsed = round(time.time() - t0, 1)

            passed_case = verdict.score >= 7
            if passed_case:
                passed += 1

            record = {
                "case_id": cid,
                "query": query,
                "mode": mode,
                "answer_preview": answer[:200],
                "judge_score": verdict.score,
                "judge_reasoning": verdict.reasoning,
                "pass": passed_case,
                "steps": steps,
                "tools_used": tools_used,
                "elapsed_sec": elapsed,
            }
            results.append(record)

            mark = "OK" if passed_case else "FAIL"
            print(f"-> score={verdict.score} {mark} ({elapsed}s)")

        except Exception as e:
            print(f"-> ERROR: {e}")
            results.append({
                "case_id": cid,
                "query": query,
                "mode": mode,
                "error": str(e),
                "judge_score": 0,
                "pass": False,
                "steps": 0,
                "elapsed_sec": 0,
            })

    total = len(results)
    pass_rate = passed / total if total > 0 else 0.0
    avg_score = sum(r.get("judge_score", 0) for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"Pass-rate: {passed}/{total} = {pass_rate:.0%}")
    print(f"Средний score: {avg_score:.1f}/10")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "summary": {
                    "total": total,
                    "passed": passed,
                    "pass_rate": round(pass_rate, 3),
                    "avg_score": round(avg_score, 2),
                },
                "cases": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Результаты -> {OUTPUT}")
    return results


if __name__ == "__main__":
    run_eval()
