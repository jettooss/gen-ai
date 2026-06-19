"""
LLM-as-judge: проверка галлюцинаций в extracted_skills.

Для каждой вакансии проверяем, действительно ли перечисленные навыки
встречаются в тексте описания. Навыки, которых нет в тексте — ghost-skills.

Запуск: python hallucination.py
Выход:  output/hallucination_report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from llm_client import get_model, make_client
from schemas import HallucinationReport, SkillVerification

EXTRACTED = Path(__file__).parent / "output" / "extracted_vacancies.json"
OUTPUT = Path(__file__).parent / "output" / "hallucination_report.json"

JUDGE_SYSTEM = """Ты проверяешь, есть ли конкретный навык в тексте описания вакансии.
Отвечай строго: present_in_text=true только если навык явно упомянут
(точное слово/аббревиатура, не просто связанная тема).
Приведи evidence — точную цитату из текста (или "не найдено")."""


def _check_skill(skill: str, description: str, client) -> SkillVerification:
    truncated = description[:2000]
    return client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Навык: {skill}\n\nТекст вакансии:\n{truncated}",
            },
        ],
        response_model=SkillVerification,
        temperature=0.7,
    )


def check_vacancy(vacancy: dict, client) -> HallucinationReport:
    skills = vacancy.get("required_skills", [])
    description = vacancy.get("raw_description", "")
    vid = vacancy.get("id", "?")

    if not skills or not description:
        return HallucinationReport(
            vacancy_id=vid,
            ghost_skills=[],
            verified_skills=skills,
            hallucination_rate=0.0,
        )

    ghost: list[str] = []
    verified: list[str] = []

    for skill in skills[:15]:
        try:
            result = _check_skill(skill, description, client)
            if result.present_in_text:
                verified.append(skill)
            else:
                ghost.append(skill)
        except Exception as e:
            print(f"    [warn] skill '{skill}': {e}")
            verified.append(skill)

    total = len(ghost) + len(verified)
    rate = len(ghost) / total if total > 0 else 0.0

    return HallucinationReport(
        vacancy_id=vid,
        ghost_skills=ghost,
        verified_skills=verified,
        hallucination_rate=round(rate, 3),
    )


def run_hallucination_check(
    vacancies: list[dict] | None = None,
    sample: int = 20,
) -> list[HallucinationReport]:
    if vacancies is None:
        if not EXTRACTED.exists():
            raise FileNotFoundError(f"Нет {EXTRACTED}. Запусти extractor.py")
        vacancies = json.loads(EXTRACTED.read_text(encoding="utf-8"))

    client = make_client()
    sample_vacs = [v for v in vacancies if v.get("required_skills") and v.get("raw_description")][:sample]

    print(f"Проверяю галлюцинации в {len(sample_vacs)} вакансиях...")
    reports: list[HallucinationReport] = []

    for i, vac in enumerate(sample_vacs, 1):
        print(f"  [{i:>2}/{len(sample_vacs)}] {vac['title'][:50]}", end=" ")
        report = check_vacancy(vac, client)
        reports.append(report)
        print(f"-> ghost: {len(report.ghost_skills)}/{len(report.ghost_skills)+len(report.verified_skills)} ({report.hallucination_rate:.0%})")

    total_skills = sum(len(r.ghost_skills) + len(r.verified_skills) for r in reports)
    total_ghost = sum(len(r.ghost_skills) for r in reports)
    overall_rate = total_ghost / total_skills if total_skills > 0 else 0.0

    print(f"\nИтого: ghost-skills {total_ghost}/{total_skills} = {overall_rate:.1%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "reports": [r.model_dump() for r in reports],
                "summary": {
                    "total_skills": total_skills,
                    "total_ghost": total_ghost,
                    "overall_hallucination_rate": round(overall_rate, 3),
                    "vacancies_checked": len(reports),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Результаты -> {OUTPUT}")
    return reports


if __name__ == "__main__":
    run_hallucination_check()
