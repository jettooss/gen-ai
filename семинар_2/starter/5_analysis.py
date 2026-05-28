"""
Финал семинара — расширенный анализ качества данных
====================================================
После раунда 5 у нас есть `personas.json` с 50 валидными персонами.
Раньше мы смотрели одну гистограмму возраста — и заканчивали. Здесь
копаем глубже: что именно пошло не так в распределении?

Что считаем:
  1. Гистограмма возрастов — была и раньше.
  2. Распределение по городам и профессиям (бары) — найдём mode collapse.
  3. Топ-N повторяющихся имён — другая грань collapse (модель любит «Анну»).
  4. Кросс-таблица город × профессия — есть ли нереалистичные комбинации?
  5. Boxplot доход × профессия — модель умеет связывать поля или просто
     генерит независимо?

На выходе:
  - ages.png         — гистограмма возрастов
  - cities.png       — распределение по городам
  - occupations.png  — распределение по профессиям
  - income_by_occupation.png — boxplot
  - report.md        — текстовая сводка для обсуждения

Запуск:
  python analysis.py [personas.json]
"""

import json
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load(path: str) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        sys.exit("Файл пустой — сначала прогоните persona_gen_solution.py")
    # Поддерживаем и плоскую (раунды 2-4), и вложенную (раунд 4.5) Persona.
    # Если внутри есть address: {city, district} — распаковываем наверх.
    flat = []
    for item in data:
        row = dict(item)
        if isinstance(row.get("address"), dict):
            addr = row.pop("address")
            row.setdefault("city", addr.get("city"))
            row.setdefault("district", addr.get("district"))
        flat.append(row)
    return pd.DataFrame(flat)

def plot_hist_ages(df: pd.DataFrame, out: str):
    plt.figure(figsize=(8, 4))
    plt.hist(df["age"], bins=12, color="#4A90D9", edgecolor="white")
    plt.xlabel("Возраст")
    plt.ylabel("Число заявок")
    plt.title(f"Распределение возраста ({len(df)} заявок)")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_bar(series: pd.Series, title: str, out: str, color="#4A90D9"):
    counts = series.value_counts()
    plt.figure(figsize=(9, 4))
    counts.plot.bar(color=color, edgecolor="white")
    plt.title(title)
    plt.ylabel("Число заявок")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return counts


def cross_table(df: pd.DataFrame) -> pd.DataFrame:
    if "city" not in df.columns or "speciality" not in df.columns:
        return pd.DataFrame()
    return pd.crosstab(df["city"], df["speciality"])


def write_report(df: pd.DataFrame, out: str):
    n = len(df)
    lines = [f"# Отчёт по {n} заявкам\n"]

    cities = df["city"].value_counts()
    top_city_pct = cities.iloc[0] / n * 100
    lines.append("## Города\n")
    lines.append(f"- Уникальных: {len(cities)}")
    lines.append(f"- Топ-1: **{cities.index[0]}** — {cities.iloc[0]} ({top_city_pct:.0f}%)")
    if top_city_pct > 40:
        lines.append("- ⚠ Превышен порог 40% → mode collapse по городам")
    else:
        lines.append("- Порог 40% не превышен")
    lines.append("")

    spec = df["speciality"].value_counts()
    top_spec_pct = spec.iloc[0] / n * 100
    lines.append("## Специальности\n")
    lines.append(f"- Уникальных: {len(spec)}")
    lines.append(f"- Топ-1: **{spec.index[0]}** — {spec.iloc[0]} ({top_spec_pct:.0f}%)")
    if top_spec_pct > 35:
        lines.append("- ⚠ Превышен порог 35% → mode collapse по специальностям")
    else:
        lines.append("- Порог 35% не превышен")
    lines.append("")

    names = df["full_name"].value_counts()
    dupes = names[names > 1]
    lines.append("## ФИО\n")
    lines.append(f"- Уникальных: {len(names)} из {n} ({len(names)/n*100:.0f}%)")
    if len(dupes):
        lines.append(f"- Повторы: {dict(dupes.head(5))}")
    else:
        lines.append("- Повторов нет")
    lines.append("")

    ct = cross_table(df)
    if not ct.empty:
        ct.to_csv("crosstab_city_speciality.csv", encoding="utf-8-sig")
        lines.append("## Кросс-таблица город × специальность\n")
        lines.append("```")
        lines.append(ct.to_string())
        lines.append("```")
        lines.append("")

    lines.append("## Спорные комбинации\n")
    lines.append(
        "- **Учитель + Финансовый учет и налогообложение** — возможно при смене "
        "карьеры, но без пояснения выглядит менее естественно, чем курс по методам обучения."
    )
    lines.append(
        "- **Медицинская сестра + Управление проектами** — реалистично для старшей "
        "медсестры, но для рядовой клинической роли нужен дополнительный контекст."
    )
    lines.append(
        "- **Инженер + Финансовый учет и налогообложение** — возможно для руководителя, "
        "но похоже на слабую связь между специальностью и курсом."
    )
    lines.append("")

    Path(out).write_text("\n".join(lines), encoding="utf-8")


def main(path: str = "personas.json"):
    df = load(path)
    print(f"Загружено: {len(df)} персон из {path}")

    plot_hist_ages(df, "ages.png")
    c = plot_bar(df["city"], "Распределение по городам", "cities.png", "#7AB66E")
    s = plot_bar(df["speciality"], "Распределение по специальностям", "specialities.png", "#D97A4A")
    write_report(df, "report.md")

    print("\nСохранено:")
    for f in ("ages.png", "cities.png", "specialities.png", "crosstab_city_speciality.csv", "report.md"):
        if Path(f).exists():
            print(f"  - {f}")

    print(f"\nТоп-город: {c.index[0]} ({c.iloc[0]}/{len(df)})")
    print(f"Топ-специальность: {s.index[0]} ({s.iloc[0]}/{len(df)})")
    print("\nДальше — открыть report.md и обсудить нереалистичные комбинации.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "applications.csv"
    main(path)
