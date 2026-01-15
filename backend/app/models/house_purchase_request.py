from pydantic import BaseModel, Field
from typing import Literal

class HousePurchaseRequest(BaseModel):
    income: float
    total_monthly_debt: float
    total_liquid_assets: float
    zip_code: str = Field(pattern=r"^\d{5}$")
    credit_score: int
    user_input: str = ""
    model: Literal["gpt-oss:20b"] = "gpt-oss:20b"