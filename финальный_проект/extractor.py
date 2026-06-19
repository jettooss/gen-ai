"""
Структурированное извлечение полей из описаний вакансий через LLM.

Запуск: python extractor.py
Вход:   input/vacancies.json
Выход:  output/extracted_vacancies.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from llm_client import get_model, make_client
from schemas import Vacancy

INPUT = Path(__file__).parent / "input" / "vacancies.json"
OUTPUT = Path(__file__).parent / "output" / "extracted_vacancies.json"

SYSTEM = """Ты извлекаешь структурированные данные о вакансии.
Определи: минимальный/максимальный оклад (если указан), опыт в годах,
список технических навыков (только конкретные технологии/языки/инструменты),
тип занятости (full/part/contract/internship).
Возвращай ТОЛЬКО JSON."""


def _extract_one(raw: dict, client) -> Vacancy | None:
    vid = str(raw.get("id", ""))
    title = raw.get("name", "")
    company = (raw.get("employer") or {}).get("name", "")
    text = raw.get("_description_text", "") or ""

    salary = raw.get("salary") or {}
    salary_min = salary.get("from")
    salary_max = salary.get("to")
    currency = salary.get("currency", "RUR")

    exp_raw = (raw.get("experience") or {}).get("id", "noExperience")
    exp_map = {
        "noExperience": 0.0,
        "between1And3": 1.5,
        "between3And6": 4.0,
        "moreThan6": 7.0,
    }
    exp_years = exp_map.get(exp_raw, 0.0)

    sched = (raw.get("schedule") or {}).get("id", "fullDay")
    emp_map = {"fullDay": "full", "shift": "full", "flexible": "full",
               "remote": "full", "flyInFlyOut": "contract"}
    emp_type = emp_map.get(sched, "full")

    # Если scraper уже распарсил навыки (из Kaggle key_skills) — пропускаем LLM-вызов
    preloaded_skills = raw.get("_key_skills_parsed") or []
    if preloaded_skills and text:
        try:
            return Vacancy(
                id=vid, title=title, company=company,
                salary_min=salary_min, salary_max=salary_max, currency=currency,
                experience_years=exp_years, required_skills=preloaded_skills,
                employment_type=emp_type, raw_description=text,
            )
        except Exception:
            pass  # fall through to LLM extraction

    truncated = text[:3000]
    prompt = f"Вакансия: {title}\nКомпания: {company}\n\nОписание:\n{truncated}"

    try:
        result = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_model=Vacancy,
            max_retries=2,
            temperature=0.0,
        )
        result = result.model_copy(update={
            "id": vid,
            "title": title,
            "company": company,
            "raw_description": text,
            "salary_min": result.salary_min if result.salary_min is not None else salary_min,
            "salary_max": result.salary_max if result.salary_max is not None else salary_max,
            "currency": currency,
            "experience_years": result.experience_years if result.experience_years > 0 else exp_years,
            "employment_type": result.employment_type or emp_type,
        })
        return result
    except Exception as e:
        print(f"  [warn] {vid} ({title[:40]}): extraction error: {e}")
        try:
            return Vacancy(
                id=vid,
                title=title,
                company=company,
                salary_min=salary_min,
                salary_max=salary_max,
                currency=currency,
                experience_years=exp_years,
                required_skills=[],
                employment_type=emp_type,
                raw_description=text,
            )
        except Exception:
            return None


def run_extractor(raw_vacancies: list[dict] | None = None) -> list[Vacancy]:
    if raw_vacancies is None:
        if not INPUT.exists():
            raise FileNotFoundError(f"Нет файла {INPUT}. Сначала запусти scraper.py")
        raw_vacancies = json.loads(INPUT.read_text(encoding="utf-8"))

    client = make_client()
    results: list[Vacancy] = []
    errors = 0

    print(f"Извлекаю данные из {len(raw_vacancies)} вакансий...")
    for i, raw in enumerate(raw_vacancies, 1):
        vac = _extract_one(raw, client)
        if vac:
            results.append(vac)
            skills_preview = ", ".join(vac.required_skills[:4])
            print(f"  [{i:>3}/{len(raw_vacancies)}] {vac.title[:45]:<45} skills: {skills_preview}")
        else:
            errors += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps([v.model_dump() for v in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nИзвлечено: {len(results)}, ошибок: {errors} -> {OUTPUT}")
    return results


if __name__ == "__main__":
    run_extractor()
