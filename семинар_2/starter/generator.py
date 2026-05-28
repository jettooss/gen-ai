import csv
import random
from pathlib import Path

from llm_client import get_model, make_client
from schema import Application


CITIES = {
    "Москва": ["Центральный округ", "Северный округ", "Юго-Западный округ"],
    "Санкт-Петербург": ["Адмиралтейский район", "Выборгский район", "Приморский район"],
    "Новосибирск": ["Центральный район", "Калининский район", "Ленинский район"],
    "Екатеринбург": ["Ленинский район", "Кировский район", "Чкаловский район"],
    "Казань": ["Вахитовский район", "Советский район", "Приволжский район"],
    "Нижний Новгород": ["Нижегородский район", "Советский район", "Автозаводский район"],
    "Челябинск": ["Центральный район", "Калининский район", "Металлургический район"],
    "Самара": ["Ленинский район", "Октябрьский район", "Промышленный район"],
    "Ростов-на-Дону": ["Кировский район", "Советский район", "Ворошиловский район"],
    "Краснодар": ["Центральный округ", "Прикубанский округ", "Западный округ"],
}

SPECIALITIES = [
    "Учитель",
    "Медицинская сестра",
    "Бухгалтер",
    "HR-специалист",
    "Инженер",
    "Маркетолог",
    "Юрист",
    "Специалист по охране труда",
]

COURSES = [
    "Цифровые инструменты в профессии",
    "Управление проектами",
    "Аналитика данных",
    "Охрана труда и безопасность",
    "Современные методы обучения",
    "Финансовый учет и налогообложение",
]

FULL_NAMES = [
    "Анна Ивановна Соколова",
    "Дмитрий Петрович Орлов",
    "Елена Сергеевна Морозова",
    "Сергей Алексеевич Кузнецов",
    "Ольга Викторовна Федорова",
    "Алексей Дмитриевич Попов",
    "Наталья Андреевна Волкова",
    "Ирина Михайловна Новикова",
    "Михаил Николаевич Смирнов",
    "Мария Олеговна Иванова",
    "Татьяна Павловна Белова",
    "Павел Игоревич Егоров",
    "Светлана Романовна Захарова",
    "Андрей Владимирович Киселев",
    "Юлия Денисовна Громова",
    "Виктор Максимович Лебедев",
    "Ксения Артемовна Сергеева",
    "Николай Евгеньевич Павлов",
    "Алина Константиновна Васильева",
    "Роман Ильич Гаврилов",
    "Вера Семеновна Макарова",
    "Георгий Валерьевич Титов",
    "Полина Кирилловна Комарова",
    "Илья Степанович Фролов",
    "Екатерина Борисовна Миронова",
    "Артур Русланович Наумов",
    "Любовь Геннадьевна Крылова",
    "Олег Станиславович Никитин",
    "Дарья Филипповна Соловьева",
    "Владимир Юрьевич Медведев",
    "Надежда Петровна Виноградова",
    "Галина Александровна Лазарева",
    "Артем Вячеславович Романов",
    "Людмила Григорьевна Зайцева",
    "Евгений Тимофеевич Борисов",
    "Валерия Ярославовна Козлова",
    "Станислав Антонович Семенов",
    "Вероника Матвеевна Мельникова",
    "Петр Данилович Фомин",
    "Алла Леонидовна Осипова",
    "Руслан Эдуардович Гусев",
    "Марина Васильевна Тарасова",
    "Кирилл Олегович Ермаков",
    "Диана Игоревна Сафонова",
    "Борис Павлович Сидоров",
    "Елизавета Романовна Анисимова",
    "Максим Андреевич Чернов",
    "Раиса Николаевна Калинина",
    "Денис Михайлович Беляев",
    "Оксана Сергеевна Ершова",
]


def build_prompt(seed_city: str) -> str:
    return f"""
Сгенерируй одну реалистичную заявку на курс повышения квалификации.

Требования:
- seed_city: {seed_city}. Используй этот город в поле address.city.
- Район выбери реалистичный для указанного города.
- Специальность строго из списка: {", ".join(SPECIALITIES)}.
- Желаемый курс строго из списка: {", ".join(COURSES)}.
- Возраст от 22 до 65, стаж от 0 до 40, год окончания от 1980 до 2024.
- Возраст, стаж и год окончания не должны противоречить друг другу.
""".strip()


def generate_with_llm(count: int = 50) -> list[Application]:
    client = make_client()
    model = get_model()
    cities = list(CITIES)
    applications = []

    for index in range(count):
        seed_city = cities[index % len(cities)]
        print(f"[{index + 1}/{count}] seed_city={seed_city}")
        application = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": build_prompt(seed_city)}],
            response_model=Application,
            max_retries=3,
            temperature=0.8,
        )
        applications.append(application)

    return applications


def generate_locally(count: int = 50, seed: int = 42) -> list[Application]:
    random.seed(seed)
    applications = []
    city_quota = list(CITIES) * (count // len(CITIES))

    for index, city in enumerate(city_quota):
        age = random.randint(24, 65)
        earliest_graduation_year = max(1980, 2026 - age + 20)
        latest_graduation_year = min(2024, 2026 - age + 28)
        graduation_year = random.randint(earliest_graduation_year, latest_graduation_year)
        max_experience = min(40, age - 20)

        application = Application.model_validate(
            {
                "full_name": FULL_NAMES[index],
                "age": age,
                "address": {"city": city, "district": random.choice(CITIES[city])},
                "speciality": SPECIALITIES[index % len(SPECIALITIES)],
                "desired_course": COURSES[(index + random.randint(0, 5)) % len(COURSES)],
                "years_of_experience": random.randint(0, max_experience),
                "graduation_year": graduation_year,
            }
        )
        applications.append(application)

    return applications


def flatten(application: Application) -> dict:
    row = application.model_dump()
    address = row.pop("address")
    row["city"] = address["city"]
    row["district"] = address["district"]
    return row


def save_csv(applications: list[Application], path: Path) -> None:
    rows = [flatten(application) for application in applications]
    fieldnames = [
        "full_name",
        "age",
        "city",
        "district",
        "speciality",
        "desired_course",
        "years_of_experience",
        "graduation_year",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_histograms(csv_path: Path) -> None:
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(csv_path)
    for column, output, title in [
        ("city", "cities.png", "Распределение заявок по городам"),
        ("speciality", "specialities.png", "Распределение заявок по специальностям"),
    ]:
        counts = data[column].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        counts.plot(kind="bar", color="#2f7d80", edgecolor="white")
        plt.title(title)
        plt.xlabel("")
        plt.ylabel("Количество заявок")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output, dpi=160)
        plt.close()


def main() -> None:
    try:
        applications = generate_with_llm(50)
    except Exception as error:
        print(f"LLM generation skipped: {error}")
        applications = generate_locally(50)

    save_csv(applications, Path("applications.csv"))
    save_histograms(Path("applications.csv"))
    print(f"Saved {len(applications)} valid applications")


if __name__ == "__main__":
    main()
