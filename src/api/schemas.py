from typing import Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    tenure_months: int = Field(..., ge=0)
    avg_monthly_spend_ngn: float = Field(..., ge=0)
    recharge_frequency_per_month: float = Field(..., ge=0)
    payment_consistency_score: float
    late_payment_count: int = Field(..., ge=0)
    operator: str
    account_type: str


class PredictionResponse(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: str
