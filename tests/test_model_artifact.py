"""Contract tests against the serialised model.

Skipped when models/model.joblib is absent so a fresh clone still runs green;
run `python -m src.build_model` to exercise them.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from api.main import _to_frame
from api.schemas import Applicant
from src import config

pytestmark = pytest.mark.skipif(
    not config.MODEL_PATH.exists(), reason="no model artefact; run src.build_model"
)


@pytest.fixture(scope="module")
def model():
    return joblib.load(config.MODEL_PATH)


@pytest.fixture(scope="module")
def meta():
    return json.loads(config.MODEL_META_PATH.read_text())


def test_meta_is_complete(meta):
    for key in ["model_name", "decision_threshold", "raw_features",
                "engineered_features", "trained_at", "git_sha"]:
        assert key in meta
    assert 0.0 < meta["decision_threshold"] < 1.0
    assert meta["raw_features"] == config.FEATURES


def test_api_frame_is_accepted_by_the_pipeline(model, raw_row):
    """The exact frame the API builds must score without error -- this is the
    seam where training and serving most often drift apart."""
    frame = _to_frame([Applicant(**raw_row)])
    proba = model.predict_proba(frame)[:, 1]

    assert proba.shape == (1,)
    assert 0.0 <= proba[0] <= 1.0


def test_pipeline_handles_null_income(model, raw_row):
    frame = _to_frame([Applicant(**{**raw_row, "MonthlyIncome": None})])
    proba = model.predict_proba(frame)[:, 1]
    assert not np.isnan(proba).any()


def test_predictions_are_deterministic(model, raw_row):
    frame = _to_frame([Applicant(**raw_row)])
    assert model.predict_proba(frame)[0, 1] == model.predict_proba(frame)[0, 1]


def test_higher_delinquency_raises_risk(model, raw_row):
    """A sanity check on direction: more 90-day-late marks must not reduce risk."""
    clean_applicant = _to_frame([Applicant(**{
        **raw_row, "NumberOfTimes90DaysLate": 0,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0})])
    risky_applicant = _to_frame([Applicant(**{
        **raw_row, "NumberOfTimes90DaysLate": 5,
        "NumberOfTime30-59DaysPastDueNotWorse": 3,
        "NumberOfTime60-89DaysPastDueNotWorse": 3})])

    assert (model.predict_proba(risky_applicant)[0, 1]
            > model.predict_proba(clean_applicant)[0, 1])


def test_batch_matches_individual_scoring(model, raw_row):
    """Batching must be a pure optimisation, not a different code path."""
    rows = [{**raw_row, "age": a} for a in (25, 45, 65)]
    batched = model.predict_proba(_to_frame([Applicant(**r) for r in rows]))[:, 1]
    individual = [model.predict_proba(_to_frame([Applicant(**r)]))[0, 1] for r in rows]

    np.testing.assert_allclose(batched, individual, rtol=1e-9)
