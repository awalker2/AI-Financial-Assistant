from pydantic import BaseModel
from typing import Literal

class RetirementRequest(BaseModel):
    current_age: int
    retirement_age: int
    current_savings: float
    current_investments: float
    supplemental_retirement_income: float
    annual_income: float
    desired_annual_income_in_retirement: float
    user_input: str = ""
    model: Literal["gemma3:27b"] = "gemma3:27b"