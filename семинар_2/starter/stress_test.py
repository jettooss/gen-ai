import time
from pathlib import Path

from llm_client import get_model, make_client
from schema import Application, DesiredCourse


CONFLICT_PROMPT = f"""
Сгенерируй одну заявку на курс повышения квалификации.

Жесткое требование к полю desired_course:
- придумай новый оригинальный курс;
- НЕ используй ни один курс из этого списка: {", ".join(DesiredCourse.__args__)};
- значение desired_course должно отличаться от всех перечисленных вариантов.

Остальные поля сделай реалистичными:
- full_name: Иван Петрович Соколов;
- age: 35;
- address.city: Москва;
- address.district: Центральный округ;
- speciality: Учитель;
- years_of_experience: 12;
- graduation_year: 2012.
""".strip()


def run_once(max_retries: int) -> dict:
    client = make_client()
    model = get_model()
    started = time.time()

    try:
        application = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": CONFLICT_PROMPT}],
            response_model=Application,
            max_retries=max_retries,
            temperature=0.9,
        )
        return {
            "max_retries": max_retries,
            "status": "success",
            "seconds": time.time() - started,
            "detail": f"desired_course={application.desired_course!r}",
        }
    except Exception as error:
        return {
            "max_retries": max_retries,
            "status": "failed",
            "seconds": time.time() - started,
            "detail": f"{type(error).__name__}: {str(error)[:350]}",
        }


def write_report(results: list[dict]) -> None:
    lines = [
        "# Stress Test",
        "",
        "Поле конфликта: `desired_course`.",
        "",
        "Схема требует одно из фиксированных значений `Literal[...]`, а промпт требует придумать новый оригинальный курс и прямо запрещает использовать значения из схемы.",
        "",
        "## Результаты",
        "",
    ]

    for result in results:
        lines.append(
            f"- `max_retries={result['max_retries']}`: "
            f"**{result['status']}** за {result['seconds']:.1f} сек.; "
            f"{result['detail']}"
        )

    failed = [result for result in results if result["status"] == "failed"]
    successes = [result for result in results if result["status"] == "success"]

    lines.extend(["", "## Вывод", ""])
    if failed and not successes:
        last = results[-1]["max_retries"]
        lines.append(
            f"Модель сдалась даже при `max_retries={last}`: ретраи повторяли конфликт между промптом и схемой, но не могли сделать одновременно новый курс и значение из `Literal[...]`."
        )
    elif failed and successes:
        first_success = successes[0]["max_retries"]
        lines.append(
            f"При малом числе ретраев конфликт проявлялся как ошибка, но при `max_retries={first_success}` модель подчинилась JSON-схеме и выбрала разрешенный курс. Это показывает, что retry иногда спасает, но ценой лишних запросов и без гарантии."
        )
    else:
        lines.append(
            "Модель сразу подчинилась JSON-схеме и проигнорировала конфликтующую часть промпта. Это тоже важный результат: схема оказалась сильнее естественно-языковой инструкции, но полагаться на это рискованно."
        )

    Path("stress_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results = []
    for max_retries in (0, 1, 3):
        print(f"Running stress test with max_retries={max_retries}")
        result = run_once(max_retries)
        print(f"  {result['status']}: {result['detail']}")
        results.append(result)

    write_report(results)
    print("Saved stress_test_report.md")


if __name__ == "__main__":
    main()
