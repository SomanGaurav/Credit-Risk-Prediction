"""FastAPI service for credit-risk scoring.

    uvicorn api.main:app --reload

The model is loaded once at startup and held in module state -- deserialising a
pipeline per request would dominate latency. If the artefact is missing the app
still starts and /health reports degraded, so a container without a baked-in
model fails visibly rather than crash-looping.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    Applicant,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
)
from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("credit-risk-api")

API_VERSION = "1.0.0"

# Probability bands for human consumers. The medium/high boundary is the model's
# own decision threshold, so "high" means exactly "the model would flag this".
LOW_RISK_CEILING = 0.10

_state: dict = {"model": None, "meta": None, "test_metrics": None}


def _load_artifacts() -> None:
    if not config.MODEL_PATH.exists():
        logger.error("model artefact missing at %s -- serving degraded",
                     config.MODEL_PATH)
        return

    _state["model"] = joblib.load(config.MODEL_PATH)
    _state["meta"] = json.loads(config.MODEL_META_PATH.read_text())

    metrics_path = config.REPORTS_DIR / "test_metrics.json"
    if metrics_path.exists():
        _state["test_metrics"] = json.loads(metrics_path.read_text())

    logger.info("loaded %s (%s), threshold %.4f",
                _state["meta"]["model_name"], _state["meta"]["source"],
                _state["meta"]["decision_threshold"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield
    _state.clear()


app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts the probability that a loan applicant becomes "
                "seriously delinquent within two years.",
    version=API_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %d in %.1fms",
                request.method, request.url.path, response.status_code, elapsed_ms)
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


def _require_model():
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python -m src.build_model` and restart.",
        )
    return _state["model"], _state["meta"]


def _to_frame(applicants: list[Applicant]) -> pd.DataFrame:
    """Rebuild the exact raw-column frame the pipeline was fit on.

    by_alias gives back the original dataset column names; the three flag columns
    are derived the same way src.data.clean derives them, so training and serving
    see identical inputs.
    """
    rows = [a.model_dump(by_alias=True) for a in applicants]
    df = pd.DataFrame(rows)[config.FEATURES]
    # A batch of one all-null column comes back as dtype object (holding Python
    # None, not NaN), which breaks numeric ops like log1p downstream. Coercing
    # explicitly guarantees float64 + NaN regardless of batch size or nulls.
    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # The 96/98 sentinels only exist in the historical extract; a live applicant
    # cannot carry one, so the flag is always 0 here.
    df["HasDelinquencySentinel"] = 0
    df["MonthlyIncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependentsMissing"] = df["NumberOfDependents"].isna().astype(int)
    return df


def _risk_band(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= LOW_RISK_CEILING:
        return "medium"
    return "low"


def _predict(applicants: list[Applicant]) -> list[Prediction]:
    model, meta = _require_model()
    threshold = meta["decision_threshold"]

    try:
        probabilities = model.predict_proba(_to_frame(applicants))[:, 1]
    except Exception as exc:  # noqa: BLE001 - surface as 500 with context
        logger.exception("scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc

    return [
        Prediction(
            probability=round(float(p), 6),
            prediction=int(p >= threshold),
            risk_band=_risk_band(float(p), threshold),
            threshold=threshold,
        )
        for p in probabilities
    ]


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe. Returns 200 even when degraded so orchestrators can
    distinguish 'process up, model missing' from 'process down'."""
    loaded = _state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=API_VERSION,
    )


@app.get("/model-info", response_model=ModelInfo, tags=["ops"])
def model_info() -> ModelInfo:
    _, meta = _require_model()
    return ModelInfo(**meta, test_metrics=_state["test_metrics"])


@app.post("/predict", response_model=Prediction, tags=["scoring"])
def predict(applicant: Applicant) -> Prediction:
    """Score a single applicant."""
    return _predict([applicant])[0]


@app.post("/batch_predict", response_model=BatchResponse, tags=["scoring"])
def batch_predict(request: BatchRequest) -> BatchResponse:
    """Score up to 10,000 applicants in one call."""
    predictions = _predict(request.applicants)
    return BatchResponse(predictions=predictions, count=len(predictions))


@app.get("/", include_in_schema=False)
def root():
    return {"service": "credit-risk", "version": API_VERSION, "docs": "/docs"}
