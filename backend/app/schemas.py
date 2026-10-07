from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Exactly the features the trained model uses. Unknown fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    country: str = Field(..., min_length=1, examples=["India"], description="Country (as in the training data)")
    year: int = Field(..., ge=1900, le=2100, examples=[2012],
                      description="Year (training data covers 1990-2013; other years trigger a warning)")
    rainfall_mm: float = Field(..., ge=0, le=6000, allow_inf_nan=False, examples=[1083],
                               description="Average annual rainfall (mm)")
    pesticides_tonnes: float = Field(..., ge=0, le=1_000_000, allow_inf_nan=False, examples=[50000],
                                     description="Pesticide use, national total (tonnes/year)")
    avg_temp: float = Field(..., ge=-10, le=50, allow_inf_nan=False, examples=[25.5], description="Average temperature (°C)")


class SuitabilityBreakdown(BaseModel):
    temperature_fit: int
    rainfall_fit: int


class PredictResponse(BaseModel):
    crop: str
    predicted_yield: float
    unit: str
    suitability_score: int
    suitability_breakdown: SuitabilityBreakdown
    status: str
    risk_level: str
    recommendations: list[str]
    warnings: list[str]
    disclaimer: str


class RecommendationResponse(BaseModel):
    """Rule-engine output (not ML). predicted_yield is included only because the rules
    compare it with the country's history; it comes from the same model call as /predict."""
    predicted_yield: float
    unit: str
    suitability_score: int
    suitability_breakdown: SuitabilityBreakdown
    status: str
    risk_level: str
    recommendations: list[str]
    warnings: list[str]
    disclaimer: str
