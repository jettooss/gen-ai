from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


City = Literal[
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Челябинск",
    "Самара",
    "Ростов-на-Дону",
    "Краснодар",
]

Speciality = Literal[
    "Учитель",
    "Медицинская сестра",
    "Бухгалтер",
    "HR-специалист",
    "Инженер",
    "Маркетолог",
    "Юрист",
    "Специалист по охране труда",
]

DesiredCourse = Literal[
    "Цифровые инструменты в профессии",
    "Управление проектами",
    "Аналитика данных",
    "Охрана труда и безопасность",
    "Современные методы обучения",
    "Финансовый учет и налогообложение",
]


class Address(BaseModel):
    city: City
    district: str = Field(min_length=2, max_length=80)


class Application(BaseModel):
    full_name: str = Field(min_length=5, max_length=120)
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_year_must_not_be_in_future(cls, value: int) -> int:
        current_year = date.today().year
        if value > current_year:
            raise ValueError("graduation_year cannot be in the future")
        return value

    @model_validator(mode="after")
    def dates_and_experience_must_match_age(self) -> "Application":
        current_year = date.today().year
        graduation_age = self.age - (current_year - self.graduation_year)
        if graduation_age < 20:
            raise ValueError("age and graduation_year contradict each other")
        if self.years_of_experience > max(0, self.age - 18):
            raise ValueError("years_of_experience is too high for age")
        return self

    @property
    def city(self) -> str:
        return self.address.city
