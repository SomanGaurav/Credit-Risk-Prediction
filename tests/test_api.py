"""API contract tests.

A stub model is injected into the app's module state so these run without the
real artefact -- CI should test the interface, not a 1.4 MB binary.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app

THRESHOLD = 0.6


class StubModel:
    """Returns a probability derived from age, so responses vary predictably."""

    def predict_proba(self, X):
        p = np.clip(X["age"].to_numpy(dtype=float) / 100.0, 0.01, 0.99)
        return np.column_stack([1 - p, p])


STUB_META = {
    "model_name": "stub", "model_class": "StubModel", "source": "test",
    "model_params": {}, "decision_threshold": THRESHOLD,
    "raw_features": api_main.config.FEATURES, "engineered_features": ["a", "b"],
    "n_training_rows": 100, "training_positive_rate": 0.067,
    "trained_at": "2026-01-01T00:00:00+00:00", "git_sha": "testsha",
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        api_main._state["model"] = StubModel()
        api_main._state["meta"] = STUB_META
        api_main._state["test_metrics"] = {"roc_auc": 0.865}
        yield c
    api_main._state.clear()


@pytest.fixture
def no_model_client():
    with TestClient(app) as c:
        api_main._state["model"] = None
        api_main._state["meta"] = None
        yield c
    api_main._state.clear()


def test_health_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_reports_degraded_without_model(no_model_client):
    """Still 200 -- the process is alive, just unable to score."""
    response = no_model_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] == "stub"
    assert body["decision_threshold"] == THRESHOLD
    assert body["raw_features"] == api_main.config.FEATURES
    assert body["test_metrics"]["roc_auc"] == 0.865


def test_scoring_returns_503_without_model(no_model_client, raw_row):
    assert no_model_client.post("/predict", json=raw_row).status_code == 503


def test_predict_applies_threshold(client, raw_row):
    body = client.post("/predict", json={**raw_row, "age": 90}).json()
    assert body["probability"] == pytest.approx(0.90)
    assert body["prediction"] == 1          # 0.90 >= 0.6
    assert body["risk_band"] == "high"

    body = client.post("/predict", json={**raw_row, "age": 30}).json()
    assert body["prediction"] == 0          # 0.30 < 0.6
    assert body["risk_band"] == "medium"    # below threshold but >= 0.10


@pytest.mark.parametrize("probability,expected", [
    (0.99, "high"), (THRESHOLD, "high"),    # at the threshold, the model flags it
    (0.59, "medium"), (0.10, "medium"),
    (0.09, "low"), (0.0, "low"),
])
def test_risk_band_boundaries(probability, expected):
    """Banded directly: the stub's age-derived probability cannot reach the
    low band, since the schema floors age at 18."""
    assert api_main._risk_band(probability, THRESHOLD) == expected


def test_predict_accepts_null_income(client, raw_row):
    """~20% of training rows had no income; the API must not reject that."""
    response = client.post("/predict", json={**raw_row, "MonthlyIncome": None})
    assert response.status_code == 200


@pytest.mark.parametrize("field,bad", [
    ("age", 5),                                       # below adult minimum
    ("age", 200),                                     # implausible
    ("MonthlyIncome", -100),                          # negative
    ("RevolvingUtilizationOfUnsecuredLines", -0.5),   # negative ratio
    ("NumberOfTimes90DaysLate", -1),                  # negative count
])
def test_predict_rejects_out_of_range(client, raw_row, field, bad):
    response = client.post("/predict", json={**raw_row, field: bad})
    assert response.status_code == 422


def test_predict_rejects_unknown_field(client, raw_row):
    """extra='forbid' catches typo'd column names instead of silently ignoring them."""
    response = client.post("/predict", json={**raw_row, "TotallyMadeUp": 1})
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client, raw_row):
    del raw_row["age"]
    assert client.post("/predict", json=raw_row).status_code == 422


def test_batch_predict(client, raw_row):
    payload = {"applicants": [{**raw_row, "age": a} for a in (25, 50, 95)]}
    body = client.post("/batch_predict", json=payload).json()

    assert body["count"] == 3
    assert [p["prediction"] for p in body["predictions"]] == [0, 0, 1]
    assert [p["probability"] for p in body["predictions"]] == pytest.approx(
        [0.25, 0.50, 0.95])


def test_batch_predict_rejects_empty(client):
    assert client.post("/batch_predict", json={"applicants": []}).status_code == 422
