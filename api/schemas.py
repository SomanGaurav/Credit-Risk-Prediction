"""Request and response models.

Field names mirror the raw dataset columns exactly, so a caller can post a row
straight from the source CSV. Aliases keep the wire format identical to the
dataset while giving Python-friendly attribute names.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    """One loan applicant. Bounds reject impossible values rather than letting the
    model silently extrapolate on them."""

    revolving_utilization: float = Field(
        alias="RevolvingUtilizationOfUnsecuredLines", ge=0,
        description="Balance on unsecured lines divided by total credit limit",
    )
    age: int = Field(alias="age", ge=18, le=120)
    times_30_59_days_late: int = Field(
        alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0, le=100)
    debt_ratio: float = Field(
        alias="DebtRatio", ge=0,
        description="Monthly debt payments divided by monthly gross income")
    monthly_income: float | None = Field(
        alias="MonthlyIncome", default=None, ge=0,
        description="Null is accepted -- ~20% of the training data had it missing")
    open_credit_lines: int = Field(
        alias="NumberOfOpenCreditLinesAndLoans", ge=0, le=100)
    times_90_days_late: int = Field(alias="NumberOfTimes90DaysLate", ge=0, le=100)
    real_estate_loans: int = Field(alias="NumberRealEstateLoansOrLines", ge=0, le=100)
    times_60_89_days_late: int = Field(
        alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0, le=100)
    dependents: float | None = Field(
        alias="NumberOfDependents", default=None, ge=0, le=50)

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.766,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.803,
                "MonthlyIncome": 9120,
                "NumberOfOpenCreditLinesAndLoans": 13,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 6,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        },
    )


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(min_length=1, max_length=10_000)


class Prediction(BaseModel):
    probability: float = Field(description="Predicted probability of serious delinquency")
    prediction: int = Field(description="1 if probability >= the model's threshold")
    risk_band: str = Field(description="low, medium or high")
    threshold: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_class: str
    source: str
    decision_threshold: float
    raw_features: list[str]
    engineered_features: list[str]
    n_training_rows: int
    training_positive_rate: float
    trained_at: str
    git_sha: str
    test_metrics: dict | None = None
