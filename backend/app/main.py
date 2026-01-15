from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
import logging
import datetime

from app.models.house_purchase_request import HousePurchaseRequest
from app.models.retirement_request import RetirementRequest
from app.models.monthly_budget_request import MonthlyBudgetRequest
from app.utils.ollama_utils import generate_ollama_stream, get_ollama_response_with_web

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/plan-home-purchase", response_class=PlainTextResponse)
def generate_house_purchase_plan(request: HousePurchaseRequest):
    try:
        prompt = f"""
        You are a financial assistant specializing in home purchases. Provide a detailed assessment of whether a potential home buyer can afford a home, and offer personalized recommendations.
        Please note that the current date is ${datetime.datetime.now().isoformat()}.
        You will also have access to the Internet, please use this to find relevant data like costs of living and housing trends in the zip code. As well as up to date interest rates.

        Here are the buyer's details:
        - Income: {request.income} per year
        - Total Monthly Debt: {request.total_monthly_debt}
        - Total Liquid Assets: {request.total_liquid_assets}
        - Zip Code: {request.zip_code} (for local property tax estimation)
        - Credit Score: {request.credit_score} (for interest rate, in addition to current trends)
        - User input: {request.user_input} (if applicable)

        Based on this information, please provide the following:

        1. **Affordability Assessment:**  Determine if the buyer can realistically afford a home and come up with a price range, considering their income, debts, and assets. Explain your reasoning.
        2. **Estimated Monthly Mortgage Payment:** Calculate the estimated monthly mortgage payment (principal and interest) for a home in their price range, using the provided interest rate.
        3. **Estimated Property Taxes:** Provide an estimate of annual property taxes for the given zip code. (Use a reasonable estimate if exact data isn't available.)
        4. **Estimated Homeowners Insurance:** Provide an estimate for annual homeowners insurance.
        5. **Total Estimated Monthly Housing Costs:** Calculate the total estimated monthly housing costs (mortgage, property taxes, insurance).
        6. **Recommendations:** Offer personalized recommendations to the buyer, such as:
            - Whether they should adjust their desired home price range.
            - Strategies for improving their financial situation (e.g., reducing debt, increasing savings).
            - Advice on securing a mortgage.

        Please provide a clear and concise assessment, using a professional tone, but limit wording as much as possible so users can get a quick response.
        """
        messages = [
            {
                'role': 'user',
                'content': prompt,
            }
        ]

        return get_ollama_response_with_web(logger, messages, request.model)

    except Exception as e:
        logger.exception("Exception processing:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan-monthly-budget", response_class=PlainTextResponse)
def generate_monthly_budget_plan(request: MonthlyBudgetRequest):
    try:
        prompt = f"""
        You are a financial assistant specializing in budgets. Please provide a brief sample monthly budget recommendation based on their income, debt, and personal preferences.
        Please note that the current date is ${datetime.datetime.now().isoformat()}.
        You will also have access to the Internet, please use this to find relevant data like costs of living in the zip code.

        Here are the buyer's details:
        - Income: {request.income} per year
        - Household Size: {request.household_size} people
        - Total Monthly Debt: {request.total_monthly_debt}
        - Zip Code: {request.zip_code} (for local property tax estimation)
        - User input: {request.user_input} (if applicable)

        Based on this information, please provide the following:

        1. **Monthly costs breakdown:**  Estimate how much of their income should go to important categories like housing, saving/investing, utilities, subscriptions, food, healthcare, ect.
            Feel free to include other categories especially if user input is provided. Explain your reasoning.
        2. **Estimated Monthly Housing Payment:** Give the client a range of how much they can spend on housing. You can leave out things like rate information, just focus on a payment range for now.
        3. **Estimated Savings and Investing:** Provide an estimate of what kinds of things the client can do by investing their income, for example if you save 10% for retirement or have some sort of custom goal.
        4. **Estimated Homeowners Insurance:** Provide an estimate for annual homeowners insurance.
        5. **Recommendations:** Offer personalized recommendations to the buyer, such as:
            - Whether their income is sufficient for the location and household size
            - Strategies for improving their financial situation (e.g., reducing debt, cutting certain costs).

        Please provide a clear and concise assessment, using a professional tone, but limit wording as much as possible so users can get a quick response.
        """
        messages = [
            {
                'role': 'user',
                'content': prompt,
            }
        ]

        return get_ollama_response_with_web(logger, messages, request.model)

    except Exception as e:
        logger.exception("Exception processing:")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan-retirement")
def generate_house_purchase_plan(request: RetirementRequest):
    try:
        prompt = f"""
        You are a financial assistant providing retirement planning advice. A client has provided the following information:

        - Current Age: {request.current_age} in years
        - Retirement Age: {request.retirement_age} in years
        - Current Cash Savings: ${request.current_savings}
        - Current Investments: ${request.current_investments}
        - Supplemental Income In Retirement: ${request.supplemental_retirement_income} (If applicable, does not include social security, please estimate this on your own in addition)
        - Annual Income: ${request.annual_income} per year
        - Desired Annual Income in Retirement: ${request.desired_annual_income_in_retirement} per year
        - User input: ${request.user_input} (if applicable)

        Based on this information, please provide a concise retirement plan addressing the following:

        1.  **Estimated Total Savings Needed at Retirement:** Calculate the total amount of savings the client will need at retirement to maintain their desired income, considering inflation.
        2.  **Annual Savings Required:** Determine the annual savings rate (as a percentage of income) the client needs to achieve their retirement goal.
        3.  **Investment Strategy:** Suggest a general investment strategy (e.g., diversified portfolio of stocks and bonds) that aligns with their risk tolerance and time horizon.
        4.  **Additional Considerations:** Offer any additional advice or recommendations, such as reducing debt or maximizing contributions to retirement accounts.
        5.  **If Feasible/On track:** Most importantly, we want the user to know if their retirement plan is realistic or needs additional suggestions to be feasible.

        Please present your response in a clear and easy-to-understand format. Limit wording as much as possible for quick understanding.
        """
        messages = [
            {
                'role': 'user',
                'content': prompt,
            },
        ]

        return StreamingResponse(
            generate_ollama_stream(messages, request.model),
            media_type="text/plain"
        )

    except Exception as e:
        logger.exception("Exception processing:")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)