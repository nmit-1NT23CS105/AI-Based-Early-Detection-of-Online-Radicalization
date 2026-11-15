from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class TextRequest(BaseModel):
    text: str = Field(..., min_length=4, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 4:
            raise ValueError("Text must contain at least four characters.")
        return value


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=25)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty text entry is required.")
        if len(cleaned) > 25:
            raise ValueError("Batch analysis is limited to 25 texts per request.")
        return cleaned

