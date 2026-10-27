from pydantic import BaseModel


class PredictionOut(BaseModel):
    flavor_id: str
    slug: str
    name: str
    confidence: float


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    low_confidence: bool
    references_used: int
    message: str | None = None


class ReferenceCreated(BaseModel):
    ids: list[str]
    flavor_id: str
    total_for_flavor: int


class FlavorStats(BaseModel):
    flavor_id: str
    slug: str
    name: str
    references: int


class ReferenceStatsResponse(BaseModel):
    flavors: list[FlavorStats]
    recommended_min: int = 8
