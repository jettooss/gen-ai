"""Pydantic-схемы для финального проекта: умный поиск IT-вакансий."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Vacancy(BaseModel):
    id: str
    title: str
    company: str
    salary_min: int | None = None
    salary_max: int | None = None
    currency: str = "RUR"
    experience_years: float = 0.0
    required_skills: list[str] = Field(default_factory=list)
    employment_type: Literal["full", "part", "contract", "internship"] = "full"
    raw_description: str = ""

    @field_validator("salary_max")
    @classmethod
    def max_not_less_than_min(cls, v, info):
        s_min = info.data.get("salary_min")
        if v is not None and s_min is not None and v < s_min:
            raise ValueError(f"salary_max ({v}) < salary_min ({s_min})")
        return v

    @field_validator("experience_years")
    @classmethod
    def experience_non_negative(cls, v):
        if v < 0:
            raise ValueError("experience_years must be >= 0")
        return v

    @field_validator("required_skills")
    @classmethod
    def deduplicate_skills(cls, v):
        seen: set[str] = set()
        result: list[str] = []
        for s in v:
            low = s.strip().lower()
            if low and low not in seen:
                seen.add(low)
                result.append(s.strip())
        return result


class VacancySummary(BaseModel):
    id: str
    title: str
    company: str
    salary_min: int | None = None
    salary_max: int | None = None
    currency: str = "RUR"
    experience_years: float = 0.0
    top_skills: list[str] = Field(default_factory=list)


class CandidateProfile(BaseModel):
    skills: list[str]
    experience_years: float
    desired_salary: int | None = None
    employment_type: list[Literal["full", "part", "contract", "internship"]] = Field(
        default_factory=lambda: ["full"]
    )

    @field_validator("experience_years")
    @classmethod
    def exp_non_negative(cls, v):
        if v < 0:
            raise ValueError("experience_years must be >= 0")
        return v


class MatchResult(BaseModel):
    vacancy_id: str
    score: int = Field(ge=1, le=10)
    matched_skills: list[str]
    missing_skills: list[str]
    reasoning: str


class HallucinationReport(BaseModel):
    vacancy_id: str
    ghost_skills: list[str]
    verified_skills: list[str]
    hallucination_rate: float

    @field_validator("hallucination_rate")
    @classmethod
    def rate_in_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hallucination_rate {v} not in [0, 1]")
        return v


class SkillVerification(BaseModel):
    skill: str
    present_in_text: bool
    evidence: str


class AnswerWithSources(BaseModel):
    answer: str
    source_ids: list[str]
    reasoning: str


class EvalCase(BaseModel):
    case_id: int
    query: str
    profile: CandidateProfile
    judge_score: int | None = None
    steps: int | None = None
    pass_: bool | None = None

    @field_validator("judge_score")
    @classmethod
    def score_in_range(cls, v):
        if v is not None and not (1 <= v <= 10):
            raise ValueError(f"judge_score {v} not in [1, 10]")
        return v


class AgentPlan(BaseModel):
    reasoning: str
    steps: list[AgentStep]


class AgentStep(BaseModel):
    id: int
    tool: str
    args: dict
    depends_on: list[int] = Field(default_factory=list)
    description: str


AgentPlan.model_rebuild()


class AgentVerdict(BaseModel):
    ok: bool
    score: int = Field(ge=1, le=10)
    reasoning: str
    issues: list[str] = Field(default_factory=list)
