"""
Загрузка вакансий из HuggingFace датасета trewwxsav/IT_vacancies_from_hh.ru.

Датасет: 68 584 реальных IT-вакансий с hh.ru (CSV, cp1251, разделитель ';')

Как скачать CSV:
  curl -L -o input/vacancies.csv \
    "https://huggingface.co/datasets/trewwxsav/IT_vacancies_from_hh.ru/resolve/main/all_vacancies.csv"

Запуск: python scraper.py
Результат: input/vacancies.json (100 вакансий)
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

INPUT_DIR = Path(__file__).parent / "input"
OUTPUT = INPUT_DIR / "vacancies.json"
CSV_PATH = INPUT_DIR / "vacancies.csv"

KAGGLE_DATASET = "antonbelyaevd/headhunter-vacancies-for-data-search"


def _download_kaggle() -> Path | None:
    """Скачать датасет через kaggle CLI. Вернуть путь к CSV или None."""
    try:
        import kaggle  # noqa: F401 — проверка установки
    except ImportError:
        print("[warn] kaggle не установлен: pip install kaggle")
        return None

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(
            "[warn] Не найден ~/.kaggle/kaggle.json\n"
            "  -> Скачай токен: kaggle.com -> Account -> API -> Create New Token\n"
            "  -> Положи в C:/Users/<user>/.kaggle/kaggle.json"
        )
        return None

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю датасет {KAGGLE_DATASET}...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", KAGGLE_DATASET, "-p", str(INPUT_DIR), "--unzip"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[error] kaggle download: {result.stderr[:300]}")
        return None

    csvs = list(INPUT_DIR.glob("*.csv"))
    if not csvs:
        print("[error] CSV не найден после скачивания")
        return None

    csv = sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)[0]
    print(f"Скачано: {csv.name} ({csv.stat().st_size // 1024} KB)")
    return csv


def _strip_html(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()
    except ImportError:
        import re
        return re.sub(r"<[^>]+>", " ", html or "").strip()


def _parse_skills(skills_raw) -> list[str]:
    """Парсим поле key_skills — может быть строкой или JSON-массивом."""
    if not skills_raw or (isinstance(skills_raw, float)):
        return []
    if isinstance(skills_raw, list):
        return [str(s).strip() for s in skills_raw if s]
    s = str(skills_raw).strip()
    if s.startswith("["):
        try:
            parsed = json.loads(s.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except Exception:
            pass
    return [sk.strip() for sk in s.split(",") if sk.strip()]


def _exp_to_years(exp_name: str) -> float:
    exp_name = str(exp_name).lower()
    if "без опыта" in exp_name or "no experience" in exp_name:
        return 0.0
    if "1" in exp_name and "3" in exp_name:
        return 1.5
    if "3" in exp_name and "6" in exp_name:
        return 4.0
    if "6" in exp_name or "more" in exp_name or "более" in exp_name:
        return 7.0
    return 0.0


def _load_csv(csv_path: Path, limit: int = 100) -> list[dict]:
    """
    Загружает CSV с HuggingFace датасета trewwxsav/IT_vacancies_from_hh.ru.
    Формат: cp1251 с BOM, разделитель ';', 7 колонок:
      название_вакансии ; название_компании ; опыт работы ;
      специализация ; профессиональные_требования ; зарплата_нижн ; зарплата_верхн
    """
    import csv as csv_mod

    results: list[dict] = []
    with open(csv_path, encoding="cp1251", errors="replace", newline="") as f:
        reader = csv_mod.reader(f, delimiter=";")
        raw_header = next(reader)
        # Убираем BOM из первой колонки
        header = [h.lstrip("﻿").strip() for h in raw_header]
        print(f"Колонки: {header}")

        # Маппинг по позиции (независимо от encoding-артефактов)
        # название_вакансии ; название_компании ; опыт работы ;
        # местоположение ; требования_обязанности ; зарплата_число ; зарплата_текст
        COL_TITLE = 0
        COL_COMPANY = 1
        COL_EXP = 2
        COL_CITY = 3
        COL_DESC = 4
        COL_SAL_NUM = 5    # одно числовое значение зарплаты
        COL_SAL_TEXT = 6   # текстовое (напр. "от 100000 до 200000")

        for i, row in enumerate(reader):
            if len(results) >= limit:
                break
            if len(row) < 5:
                continue

            title = row[COL_TITLE].strip()
            if not title:
                continue

            company = row[COL_COMPANY].strip() if len(row) > COL_COMPANY else ""

            try:
                sal_num_raw = row[COL_SAL_NUM].strip() if len(row) > COL_SAL_NUM else ""
                sal_num = int(float(sal_num_raw)) if sal_num_raw else None
            except (ValueError, TypeError):
                sal_num = None
            sal_from = sal_num
            sal_to = None

            exp_raw = row[COL_EXP].strip() if len(row) > COL_EXP else ""
            exp_years = _exp_to_years(exp_raw)

            desc = row[COL_DESC].strip() if len(row) > COL_DESC else ""

            results.append({
                "id": str(i),
                "name": title,
                "employer": {"name": company},
                "salary": {"from": sal_from, "to": sal_to, "currency": "RUR"},
                "experience": {"name": exp_raw},
                "schedule": {"id": "fullDay"},
                "description": desc[:5000],
                "_description_text": desc[:5000],
                "_key_skills_parsed": [],
                "_source": "hf_hh_ru",
            })

    return results


def run_scraper(target: int = 100) -> list[dict]:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = CSV_PATH
    if not csv_path.exists():
        csv_path = _download_kaggle()

    if csv_path and csv_path.exists():
        print(f"Читаю {csv_path.name}...")
        results = _load_csv(csv_path, limit=target)
        print(f"Загружено {len(results)} вакансий из HuggingFace датасета")
    else:
        print("CSV недоступен, генерирую mock-данные...")
        results = _generate_mock_vacancies(target)

    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Сохранено {len(results)} вакансий -> {OUTPUT}")
    return results


# ---------------------------------------------------------------------------
# Mock fallback (если CSV недоступен)
# ---------------------------------------------------------------------------

MOCK_TEMPLATES = [
    ("Data Scientist", ["Python", "scikit-learn", "pandas", "SQL", "ML", "statistics", "Jupyter"], 150000, 250000),
    ("ML Engineer", ["Python", "PyTorch", "Docker", "MLflow", "FastAPI", "Kubernetes", "CI/CD"], 180000, 300000),
    ("Python Backend Developer", ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker", "SQLAlchemy"], 150000, 280000),
    ("Data Engineer", ["Python", "Airflow", "Spark", "SQL", "Kafka", "dbt", "Hadoop"], 160000, 260000),
    ("Junior Data Scientist", ["Python", "pandas", "scikit-learn", "SQL", "Jupyter", "NumPy"], 70000, 120000),
    ("Senior Data Scientist", ["Python", "PyTorch", "ML", "A/B testing", "statistics", "SQL", "Spark"], 250000, 400000),
    ("NLP Engineer", ["Python", "transformers", "BERT", "spaCy", "NLP", "PyTorch", "HuggingFace"], 200000, 350000),
    ("Computer Vision Engineer", ["Python", "PyTorch", "OpenCV", "YOLO", "TensorFlow", "NumPy"], 180000, 300000),
    ("Data Analyst", ["SQL", "Python", "Tableau", "Excel", "Power BI", "statistics", "A/B testing"], 90000, 160000),
    ("Research Scientist", ["Python", "PyTorch", "CUDA", "RLHF", "transformers", "LLM", "research"], 300000, 500000),
    ("MLOps Engineer", ["Python", "Docker", "Kubernetes", "MLflow", "Airflow", "CI/CD", "AWS"], 200000, 350000),
    ("Recommender Systems Engineer", ["Python", "collaborative filtering", "Spark", "SQL", "ML", "A/B testing"], 180000, 280000),
    ("BI Developer", ["SQL", "Power BI", "Tableau", "Python", "ETL", "Excel"], 100000, 180000),
    ("AI Researcher", ["Python", "PyTorch", "LLM", "fine-tuning", "CUDA", "research", "NLP"], 280000, 450000),
    ("Data Science Lead", ["Python", "ML", "team lead", "SQL", "statistics", "PyTorch", "communication"], 350000, 550000),
]

COMPANIES = [
    "Яндекс", "Сбер", "VK", "Тинькофф", "Авито", "Wildberries", "МТС",
    "Ozon", "HeadHunter", "Kaspersky Lab", "1С", "СКБ Контур",
    "DataArt", "EPAM", "Рамблер", "Mail.ru", "Lamoda", "X5 Group",
]


def _generate_mock_vacancies(count: int = 70) -> list[dict]:
    import random
    random.seed(42)
    vacancies = []
    for i in range(count):
        title, skills, sal_min, sal_max = MOCK_TEMPLATES[i % len(MOCK_TEMPLATES)]
        company = COMPANIES[i % len(COMPANIES)]
        skill_sample = random.sample(skills, min(len(skills), random.randint(4, len(skills))))
        var = random.uniform(0.8, 1.2)
        desc = (
            f"Компания {company} ищет {title}. "
            f"Требования: {', '.join(skill_sample)}. "
            f"Зарплата: {int(sal_min*var)}–{int(sal_max*var)} RUR."
        )
        vacancies.append({
            "id": str(100000 + i),
            "name": title,
            "employer": {"name": company},
            "salary": {"from": int(sal_min * var), "to": int(sal_max * var), "currency": "RUR"},
            "experience": {"name": ""},
            "schedule": {"id": "fullDay"},
            "description": desc,
            "_description_text": desc,
            "_key_skills_parsed": skill_sample,
            "_source": "mock",
        })
    return vacancies


if __name__ == "__main__":
    run_scraper()
