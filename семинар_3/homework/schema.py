from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


IssueCategory = Literal[
    "performance",
    "design",
    "support",
    "price",
    "ads",
    "reliability",
]
Sentiment = Literal["positive", "negative", "neutral"]


class Issue(BaseModel):
    category: IssueCategory
    severity: int = Field(ge=1, le=5)
    quote: str = Field(min_length=3)


class Review(BaseModel):
    review_id: str
    app_name: str
    rating: int = Field(ge=1, le=5)
    review_date: Optional[date] = None
    app_version: Optional[str] = None
    issues: list[Issue] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def review_date_not_in_future(cls, value: Optional[date]) -> Optional[date]:
        if value is not None and value > date.today():
            raise ValueError("review_date cannot be in the future")
        return value


class AspectSentiment(BaseModel):
    aspect: IssueCategory
    sentiment: Sentiment
    quote: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    review_id: str
    app_name: str
    aspects: list[AspectSentiment] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    source: str
    key_points: list[str] = Field(min_length=1, max_length=8)
    sentiment: Literal["positive", "negative", "mixed"]


class GroupSummary(BaseModel):
    sources: list[str]
    themes: list[str] = Field(min_length=1, max_length=8)
    overall_sentiment: Literal["positive", "negative", "mixed"]


class ReviewSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=10)
    action_items: list[str] = Field(min_length=1, max_length=8)


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


class DiscoveredAspect(BaseModel):
    name: str = Field(min_length=2)
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class DynamicAspect(BaseModel):
    aspect: str
    sentiment: Sentiment
    quote: str = Field(min_length=3)
    confidence: float = Field(ge=0, le=1)


class DynamicReviewSentiment(BaseModel):
    review_id: str
    app_name: str
    aspects: list[DynamicAspect] = Field(default_factory=list)


class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=10)
    unique_per_app: dict[str, list[str]]
    overall_headline: str
