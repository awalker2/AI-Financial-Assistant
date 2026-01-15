from pydantic import BaseModel, Field
from typing import Literal

class MonthlyBudgetRequest(BaseModel):
    income: float
    total_monthly_debt: float
    household_size: int
    zip_code: str = Field(pattern=r"^\d{5}$")
    user_input: str = ""
    model: Literal["gpt-oss:20b"] = "gpt-oss:20b"