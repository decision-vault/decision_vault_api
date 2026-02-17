from pydantic import BaseModel, Field


class HFInferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    integrations: list[dict] | None = None
    context_length: int | None = Field(default=None, ge=512, le=32768)


class HFInferenceResponse(BaseModel):
    output: str
    tokens: int
    model: str
