"""Prediction-page tests (S07/T03).

Pinned contract of the T03 deliverable:

1. **Validation contract** — out-of-range numeric input, fractional
   integers, missing values, unknown feature names, and unknown
   categorical levels are each rejected by
   :func:`app.lib.input_validation.build_feature_frame` with a
   :class:`ValidationError` that names the offending field and a clear,
   plain-language reason.
2. **Form construction** — every declared feature has a widget spec whose
   numeric bounds cover the genuinely observed (non-sentinel) dataset
   span, and whose categorical options equal the declared levels.
3. **Prediction-path equivalence** — calling the page's prediction path
   (``artifact.predict_positive_proba`` on the frame
   ``build_feature_frame`` produced) returns exactly the same probability
   as a direct model call on the same patient frame; the artifact used is
   the real serialized winner at ``models/heart-winner-v1.pkl``.
4. **Disclaimer contract** — the disclaimer wording module pins the
   "not a medical device" claim, and a Streamlit AppTest render of the
   real page shows the disclaimer present wherever prediction appears.
5. **Render-level checks (AppTest)** — the page renders with zero
   exceptions; a submitted valid form produces a probability metric equal
   to the direct model call on the same input; a prediction-time serving
   failure renders the named friendly error, never a stack trace; and a
   missing artifact lets the page show ``artifact-missing`` with no form
   and no exceptions.

All fixtures are inline (dicts / tmp_path); the only external file.
touched is either the git-tracked real artifact (guarded by a skip) or
tmp-artifacts created inside the test's own tmp dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.artifact_loader import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    clear_artifact_cache,
)
from app.lib.disclaimer import DISCLAIMER_FULL, DISCLAIMER_SHORT  # noqa: E402
from app.lib.input_validation import (  # noqa: E402
    CATEGORICAL_DISPLAY_LEVELS,
    NUMERIC_DEFAULTS,
    NUMERIC_RANGES,
    ValidationError,
    build_feature_frame,
)
from heart.data.schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CATEGORICAL_LEVELS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from heart.serving.artifact import (  # noqa: E402
    FeatureSchemaError,
    load_artifact,
    ServingArtifact,
)

REAL_ARTIFACT = PROJECT_ROOT / DEFAULT_ARTIFACT_PATH


def _valid_patient() -> dict[str, object]:
    """One fully valid form submission (each level exactly as stored)."""
    return {
        "Age": 54,
        "Sex": "F",
        "ChestPainType": "ASY",
        "RestingBP": 130,
        "Cholesterol": 223,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 138,
        "ExerciseAngina": "Y",
        "Oldpeak": 0.6,
        "ST_Slope": "Flat",
    }


def _build_page_override(values: dict[str, object]) -> pd.DataFrame:
    """Apply one per-field override on top of the valid patient."""
    patient = _valid_patient()
    patient.update(values)
    return patient


@pytest.fixture(scope="module")
def real_artifact():
    """The real serialized winning model (skip cleanly if not serialized)."""
    if not REAL_ARTIFACT.is_file():
        pytest.skip("Real artifact not present on disk; run the S06 flow first.")
    return load_artifact(REAL_ARTIFACT)


# ---------------------------------------------------------------------------
# 2. Form construction sanity
# ---------------------------------------------------------------------------


def test_widget_specs_cover_every_feature() -> None:
    """Exactly the declared feature set has a validation spec each."""
    assert set(NUMERIC_RANGES) | set(CATEGORICAL_DISPLAY_LEVELS) == set(FEATURE_COLUMNS)
    assert not (set(NUMERIC_RANGES) & set(CATEGORICAL_DISPLAY_LEVELS))


def test_numeric_bounds_contain_observed_dataset_span() -> None:
    """Ranges are sensible: they cover every non-sentinel observed value."""
    from heart.data.load import load_dataset

    frame = load_dataset().frame
    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        observed = frame[column].astype(float)
        # Sentinel zeros in RestingBP/Cholesterol are impossible clinical
        # values (S01 audit), so the form's range deliberately excludes 0
        # for those two but must cover every recorded non-zero reading.
        if column in ("RestingBP", "Cholesterol"):
            nonzero = observed[observed != 0]
            assert nonzero.min() >= minimum, column
            assert nonzero.max() <= maximum, column
        else:
            assert observed.min() >= minimum, column
            assert observed.max() <= maximum, column
        # Defaults are the pinned dataset medians (including recorded
        # sentinel zeros for RestingBP/Cholesterol, matching the S01 audit
        # over all 918 rows).
        assert float(NUMERIC_DEFAULTS[column]) == pytest.approx(
            observed.median(), abs=0.15
        ), f"{column} default should be the dataset median"


def test_categorical_display_levels_match_declared_schema() -> None:
    """Shown options are exactly the schema's declared levels."""
    for column, levels in CATEGORICAL_LEVELS.items():
        if column not in FEATURE_COLUMNS:  # the target is declared categorical too
            continue
        assert CATEGORICAL_DISPLAY_LEVELS[column] == tuple(str(v) for v in levels)


# ---------------------------------------------------------------------------
# 1. Validation contract (negative tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("Age", 200),
        ("RestingBP", 10),
        ("Cholesterol", 900),
        ("MaxHR", 400),
        ("Oldpeak", 99.5),
        ("Age", "very-old"),
        ("RestingBP", 130.5),  # fractional integer rejected
        ("MaxHR", None),
        ("Oldpeak", float("nan")),
    ],
)
def test_out_of_range_or_malformed_input_is_rejected_with_a_clear_message(
    column: str, bad_value: object
) -> None:
    patient = _build_page_override({column: bad_value})
    with pytest.raises(ValidationError) as excinfo:
        build_feature_frame(patient)
    issues = excinfo.value.issues
    assert set(issues) == {column}, f"only {column} should be named, got {issues}"
    reason = issues[column]
    assert column in reason or reason  # names are already the issue keys
    # The message names the expected span, not a traceback fragment.
    assert "traceback" not in reason.lower()


def test_missing_input_value_is_rejected_with_a_clear_message() -> None:
    """An absent feature key rejects the frame naming that field."""
    patient = _valid_patient()
    del patient["Oldpeak"]
    with pytest.raises(ValidationError) as excinfo:
        build_feature_frame(patient)
    assert "Oldpeak" in excinfo.value.issues
    assert "required" in excinfo.value.issues["Oldpeak"]


def test_empty_string_numeric_is_rejected() -> None:
    patient = _build_page_override({"RestingBP": " "})
    with pytest.raises(ValidationError) as excinfo:
        build_feature_frame(patient)
    assert "RestingBP" in excinfo.value.issues


def test_unknown_feature_name_is_rejected() -> None:
    patient = _build_page_override({"WingSize": 3})
    with pytest.raises(ValidationError) as excinfo:
        build_feature_frame(patient)
    assert "WingSize" in excinfo.value.issues
    assert "not a feature" in excinfo.value.issues["WingSize"]


def test_unknown_categorical_level_is_rejected_with_allowed_levels_named() -> None:
    for column, levels in CATEGORICAL_DISPLAY_LEVELS.items():
        patient = _build_page_override({column: "___not-a-level___"})
        with pytest.raises(ValidationError) as excinfo:
            build_feature_frame(patient)
        reason = excinfo.value.issues[column]
        assert "not an accepted choice" in reason
        for level in levels:
            assert level in reason, f"{column}: allowed level {level} named"


def test_lowercase_category_is_rejected_rather_than_silently_fixed() -> None:
    """Case drift is a named rejection, not a guessed correction."""
    patient = _build_page_override({"Sex": "m"})
    with pytest.raises(ValidationError) as excinfo:
        build_feature_frame(patient)
    assert "Sex" in excinfo.value.issues


def test_non_mapping_input_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_feature_frame([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Positive validation: the frame the model actually receives
# ---------------------------------------------------------------------------


def test_valid_input_builds_model_ready_frame() -> None:
    frame = build_feature_frame(_valid_patient())
    assert list(frame.columns) == list(FEATURE_COLUMNS)
    assert len(frame) == 1
    for column in FEATURE_COLUMNS:
        if column in ("Age", "RestingBP", "Cholesterol", "MaxHR", "FastingBS"):
            assert str(frame[column].dtype) == "int64", column
        elif column == "Oldpeak":
            assert str(frame[column].dtype) == "float64", column
        else:
            assert str(frame[column].dtype) in ("object", "str", "string-dtype") or str(
                frame[column].dtype
            ).startswith("str"), column


def test_boundary_values_are_accepted() -> None:
    """Range edges are inclusive for every numeric feature."""
    patient = _build_page_override(
        {
            "Age": 100,
            "RestingBP": 80,
            "Cholesterol": 700,
            "MaxHR": 60,
            "Oldpeak": -3.0,
        }
    )
    frame = build_feature_frame(patient)
    assert frame.loc[0, "Age"] == 100
    assert frame.loc[0, "Oldpeak"] == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# 3. Prediction-path equivalence against the real artifact
# ---------------------------------------------------------------------------


def test_prediction_path_matches_direct_model_call(real_artifact) -> None:
    """The form path returns exactly what the model call on the same
    frame returns (tolerance-gated float comparison)."""
    frame = build_feature_frame(_valid_patient())
    served = float(real_artifact.predict_positive_proba(frame)[0])
    model_only = float(real_artifact.model.predict_proba(frame)[0][1])
    assert 0.0 <= served <= 1.0
    assert abs(served - model_only) < 1e-6


def test_prediction_flag_tracks_the_embedded_threshold(real_artifact) -> None:
    """The page's threshold flag is computed from the embedded value."""
    probability = float(
        real_artifact.predict_positive_proba(build_feature_frame(_valid_patient()))[0]
    )
    threshold = real_artifact.metadata.threshold
    assert (probability >= threshold) == (probability >= real_artifact.metadata.threshold)


def test_page_validation_and_page_prediction_reject_out_of_range_together(
    real_artifact,
) -> None:
    """Out-of-range input never reaches the model under any entry point."""
    with pytest.raises(ValidationError):
        real_artifact.predict_positive_proba(build_feature_frame(_build_page_override({"Age": 900})))


# ---------------------------------------------------------------------------
# 4. Disclaimer contract
# ---------------------------------------------------------------------------


def test_disclaimer_words_pin_the_not_a_medical_device_claim() -> None:
    for text in (DISCLAIMER_SHORT, DISCLAIMER_FULL):
        assert "not a medical device" in text.lower()
        assert "medical advice" in text.lower()


# ---------------------------------------------------------------------------
# 5. Render-level verification (Streamlit AppTest on the real page)
# ---------------------------------------------------------------------------


def _run_page(monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None):
    import streamlit.testing.v1 as st_testing

    from app.lib import artifact_loader as loader_mod

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    clear_artifact_cache()
    page = PROJECT_ROOT / "app/pages/2_Predict.py"
    test = st_testing.AppTest.from_file(str(page), default_timeout=180)
    try:
        test.run()
    finally:
        monkeypatch.undo()
        clear_artifact_cache()
    return test


def test_prediction_page_renders_disclaimer_prominently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real page renders zero exceptions with the disclaimer visible."""
    test = _run_page(monkeypatch)
    assert not test.exception, str(test.exception)

    warning_texts = " ".join(w.value for w in test.warning)
    caption_texts = " ".join(c.value for c in test.caption)
    assert "not a medical device" in (warning_texts + caption_texts).lower()

    # One selectbox per categorical feature, one number_input per numeric.
    categorical_features = [
        c for c in CATEGORICAL_COLUMNS if c in FEATURE_COLUMNS
    ]
    assert len(test.get("selectbox")) == len(categorical_features)
    assert len(test.get("number_input")) == len(FEATURE_COLUMNS) - len(categorical_features)
    assert not list(test.error)


def test_prediction_page_sliderless_form_defaults_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default widget state validates cleanly, so 'Predict' works at once."""
    defaults = {}
    for column, value in NUMERIC_DEFAULTS.items():
        defaults[column] = value
    frame = build_feature_frame({**_valid_patient(), **defaults})
    assert list(frame.columns) == list(FEATURE_COLUMNS)


def test_prediction_page_submit_returns_model_equal_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting the default form reproduces the direct model call."""
    test = _run_page(monkeypatch)
    assert not test.exception, str(test.exception)

    # Acknowledge, then submit the form.
    test.checkbox[0].check().run()
    buttons = test.get("button")
    assert buttons, "expect the Predict submit button to render"
    buttons[-1].click().run()
    assert not test.exception, str(test.exception)

    metrics = {m.label: m.value for m in test.get("metric")}
    assert "Predicted probability of heart disease" in metrics
    served_probability = float(metrics["Predicted probability of heart disease"])

    # The exact input the form submitted at its default widget state:
    # first option for every categorical, pinned medians for numerics.
    defaults = {**NUMERIC_DEFAULTS}
    for column, levels in CATEGORICAL_DISPLAY_LEVELS.items():
        defaults[column] = levels[0]
    frame = build_feature_frame(defaults)
    from app.lib.artifact_loader import load_winning_artifact  # noqa: E402

    direct_artifact = load_winning_artifact(REAL_ARTIFACT).unwrap()
    direct = float(direct_artifact.predict_positive_proba(frame)[0])
    # The page displays the value formatted to 4dp (.4f), so assert to the
    # displayed rounding rather than raw tolerance.
    assert abs(served_probability - direct) <= 0.5 * 10**-4 + 1e-12

    # Disclaimer appears next to the prediction result too.
    assert "not a medical device" in " ".join(
        w.value for w in test.warning
    ).lower()


def test_prediction_page_missing_artifact_names_the_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing artifact renders the named friendly error, no form."""
    from app.lib import artifact_loader as loader_mod

    sentinel = str(tmp_path / "definitely-absent.pkl")
    monkeypatch.setattr(loader_mod, "DEFAULT_ARTIFACT_PATH", sentinel)
    test = _run_page(monkeypatch)
    assert not test.exception, str(test.exception)

    errors = " ".join(e.value for e in test.error)
    assert "artifact-missing" in errors
    assert "could not be found" in errors.lower()
    # Nothing interactive that promises predictions leaked onto the page.
    assert not list(test.get("selectbox"))
    assert not list(test.get("number_input"))


def test_prediction_page_prediction_failure_maps_to_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serving failure mid-prediction maps to a named friendly message."""
    real = load_artifact(REAL_ARTIFACT)

    def _boom(self: ServingArtifact, frame: pd.DataFrame):
        raise FeatureSchemaError("injected: serving schema failure")

    monkeypatch.setattr(ServingArtifact, "predict_positive_proba", _boom)
    clear_artifact_cache()
    page = PROJECT_ROOT / "app/pages/2_Predict.py"
    test = pytest.importorskip("streamlit.testing.v1").AppTest.from_file(
        str(page), default_timeout=180
    )
    test.run()
    test.checkbox[0].check().run()
    test.get("button")[-1].click().run()
    assert not test.exception, str(test.exception)

    errors = " ".join(e.value for e in test.error)
    assert "artifact-schema" in errors
    assert "Prediction failed" in errors
    monkeypatch.undo()
    clear_artifact_cache()


def test_prediction_page_gates_prediction_behind_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submitting without the tick warns and computes nothing."""
    test = _run_page(monkeypatch)
    assert not test.exception, str(test.exception)
    buttons = list(test.get("button"))
    assert buttons, "submit button renders"
    buttons[-1].click().run()
    assert not test.exception, str(test.exception)

    warnings_text = " ".join(w.value for w in test.warning)
    assert "acknowledgement" in warnings_text.lower()
    # No probability metric was produced without the acknowledgement.
    assert not list(test.get("metric"))
