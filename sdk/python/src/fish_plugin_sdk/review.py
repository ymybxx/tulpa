"""Optional, read-only evaluation goals declared by a plugin."""

from pydantic import BaseModel, ConfigDict, Field


class ReviewCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(min_length=1, max_length=500)


class ReviewReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=180)
    url: str = Field(pattern=r"^https://", max_length=500)
    note: str = Field(default="", max_length=500)


class ReviewProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=120)
    goal: str = Field(min_length=1, max_length=1600)
    criteria: list[ReviewCriterion] = Field(min_length=1, max_length=8)
    references: list[ReviewReference] = Field(default_factory=list, max_length=8)
    mechanism: dict[str, str | float | int] = Field(default_factory=dict, max_length=20)
