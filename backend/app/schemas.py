from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CategoryId = Literal[101, 102, 103, 104, 105, 106]
MAX_DESCRIPTION_CHARS = 100000
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_DESCRIPTION_CHARS)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Category(StrictModel):
    id: CategoryId
    # The UI sends a name, but the prompt uses the server-side catalog only.
    name: str | None = Field(default=None, max_length=200)


class ClassificationRequest(StrictModel):
    title: str = Field(default='', max_length=100)
    description: Description
    mcId: CategoryId = 101


class ClassificationResponse(StrictModel):
    categories: list[Category]


class GenerateRequest(StrictModel):
    title: str = Field(default='', max_length=100)
    description: Description
    categories: list[Category] = Field(min_length=1, max_length=6)
    mcId: CategoryId | None = None

    @model_validator(mode='after')
    def validate_categories(self):
        ids = [c.id for c in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError('Категории не должны повторяться.')
        return self


class ModelDraft(StrictModel):
    # Ignore incidental extra fields from older model outputs; only the three
    # drafting fields matter. Categories were selected before this API call.
    model_config = ConfigDict(extra='ignore', strict=True)
    categoryId: CategoryId
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class ModelResponse(StrictModel):
    drafts: list[ModelDraft] = Field(min_length=1, max_length=6)


class Draft(StrictModel):
    categoryId: CategoryId
    title: str
    description: str


class GenerationWarning(StrictModel):
    categoryId: CategoryId
    reason: Literal['needs_details'] = 'needs_details'
    message: str


class GenerateResponse(StrictModel):
    drafts: list[Draft]
    warnings: list[GenerationWarning] = Field(default_factory=list)
