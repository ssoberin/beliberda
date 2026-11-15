from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CategoryId = Literal[101, 102, 103, 104, 105, 106]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Category(StrictModel):
    id: CategoryId
    # The UI sends a name, but the prompt uses the server-side catalog only.
    name: str | None = Field(default=None, max_length=200)


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
    model_config = ConfigDict(extra='forbid', strict=True)
    categoryId: CategoryId
    supported: bool
    title: Annotated[str, StringConstraints(min_length=1, max_length=50)] | None
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)] | None
    evidence: list[Annotated[str, StringConstraints(min_length=1, max_length=800)]] = Field(max_length=12)

    @model_validator(mode='after')
    def validate_support(self):
        if self.supported:
            if not self.title or not self.title.strip() or not self.text or not self.text.strip() or not self.evidence:
                raise ValueError('Подтверждённый черновик должен содержать заголовок, текст и цитаты.')
        elif self.evidence:
            raise ValueError('Для заготовки без подтверждения evidence должен быть пустым.')
        elif (self.title is None) != (self.text is None):
            raise ValueError('Заголовок и текст должны присутствовать вместе.')
        elif self.title is not None and (not self.title.strip() or not self.text.strip()):
            raise ValueError('Заголовок и текст заготовки не должны быть пустыми.')
        return self


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
