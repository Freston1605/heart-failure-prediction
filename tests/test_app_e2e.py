"""End-to-end app verification against the REAL serialized artifact (S07/T04).

This is the integration proof that the training-to-serving boundary holds.
Every fixture here is the real thing — the committed winner artifact at
``models/heart-winner-v1.pkl``, the pinned dataset, the committed
``reports/leaderboard.md`` — no stubs and no synthetic win.

Contractpinned:

1. **Serving path (app side)** — :func:`app.lib.artifact_loader.load_winning_artifact`
   loads the real committed artifact and reports the selection-time truth:
   winner model name, tuned threshold, isotonic calibration state, and the
   honest NO-SHIP flags.
2. **App path == model path on known inputs** — the page contract path
   (``build_feature_frame`` → ``ServingArtifact.predict_positive_proba``)
   produces the exact same probability as calling the model directly on a
   hand-built frame of the same patient; and two real held-out rows (one
   per class) yield their recorded probabilities.
3. **Predictions match the training-time evaluation** —
   (a) the artifact's metadata bit-matches the committed leaderboard's
   winner-selection block (model, threshold, Brier before/after, NO-SHIP),
   and (b) scoring the artifact's calibrated probabilities on the selection
   flow's validation carve (seeded identically) reproduces the committed
   objective value (f1 = 0.860335 at threshold 0.33).
4. **Both flows render against the real artifact** — Home (serving banner),
   Explore (all four insight groups + S01-audited class balance), and
   Predict (submit → probability equal to the direct model call), each with
   zero page exceptions.
5. **Error paths behave** — a corrupt, missing, or schema-incompatible
   artifact and a missing dataset each render the *named* friendly error
   (artifact-corrupt / artifact-missing / artifact-incompatible /
   data-missing) with no stack-trace white screen.

Real renders use ``streamlit.testing.v1.AppTest`` in-process; caches are
cleared around each render so a monkeypatched path can never be masked by
a stale cached load.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib.artifact_loader import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    clear_artifact_cache,
    load_winning_artifact,
    project_root,
)
from app.lib.input_validation import (  # noqa: E402
    CATEGORICAL_DISPLAY_LEVELS,
    NUMERIC_DEFAULTS,
    build_feature_frame,
)
from app.lib.plots import clear_explore_cache, load_explore_data  # noqa: E402
from heart.data.schema import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from heart.serving.artifact import ArtifactError, load_artifact  # noqa: E402

REAL_ARTIFACT = project_root() / DEFAULT_ARTIFACT_PATH
REAL_ARTIFACT_RELATIVE = DEFAULT_ARTIFACT_PATH

pytestmark = pytest.mark.skipif(
    not REAL_ARTIFACT.is_file(),
    reason="Real serialized artifact missing; run the S06 winner flow first.",
)


# ---------------------------------------------------------------------------
# Fixtures + known inputs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def app_artifact():
    """Real artifact through the app's own serving path (cached loader)."""
    clear_artifact_cache()
    result = load_winning_artifact()
    assert result.ok, f"real artifact failed to load: {result.error}"
    return result.artifact


@pytest.fixture(scope="module")
def direct_artifact():
    """Real artifact through the model-side loader (independent load)."""
    return load_artifact(REAL_ARTIFACT)


def _form_defaults() -> dict[str, object]:
    """The Predict form's default submission (first levels + medians)."""
    values: dict[str, object] = dict(NUMERIC_DEFAULTS)
    for column, levels in CATEGORICAL_DISPLAY_LEVELS.items():
        values[column] = levels[0]
    return values


def _dataset_rows(rows: tuple[int, ...]) -> list[dict[str, object]]:
    """Named known inputs from the real held-out test split."""
    from heart.data.split import load_split_frames

    _train, test = load_split_frames(version="v1")
    out: list[dict[str, object]] = []
    for row in rows:
        # Feature values only — the target column is not a form input.
        record: dict[str, object] = test.iloc[row][list(FEATURE_COLUMNS)].to_dict()
        record["__index__"] = row
        out.append(record)
    return out


def _hand_built_patient_frame(record: dict[str, object]) -> pd.DataFrame:
    """The model-side path: a frame built *without* the app validators.

    Column order and value types come straight from the serialized real
    artifact — deliberately independent of app.lib.input_validation so
    agreement between the two paths is a real integration statement.
    """
    row = {
        column: record[column] if column not in ("FastingBS",) else int(record[column])
        for column in FEATURE_COLUMNS
    }
    for column in ("Age", "RestingBP", "Cholesterol", "MaxHR"):
        row[column] = int(row[column])  # type: ignore[arg-type]
    row["Oldpeak"] = float(row["Oldpeak"])  # type: ignore[arg-type]
    for column, levels in CATEGORICAL_DISPLAY_LEVELS.items():
        if column != "FastingBS":
            row[column] = str(row[column])  # type: ignore[arg-type]
        else:
            row[column] = int(row[column])  # type: ignore[arg-type]
    return pd.DataFrame([row])[pd.Index(list(FEATURE_COLUMNS))]


# ---------------------------------------------------------------------------
# 1. The real artifact loads through the serving path
# ---------------------------------------------------------------------------


def test_real_artifact_loads_through_app_serving_path(app_artifact) -> None:
    metadata = app_artifact.metadata
    assert metadata.model_name == "random-forest"
    assert tuple(metadata.feature_columns) == FEATURE_COLUMNS
    assert metadata.calibration_applied
    assert 0.0 < metadata.threshold < 1.0
    assert metadata.ship is False, "selection verdict must ride along honestly"


def test_app_loader_result_is_cached_per_process(app_artifact) -> None:
    again = load_winning_artifact()
    assert again.ok and again.artifact is app_artifact


# ---------------------------------------------------------------------------
# 2. App path == model path on known inputs (the plan's pin)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        {"__name__": "form-defaults", **_form_defaults()},
        {"__name__": "test-row-0", **_dataset_rows((0,))[0]},
        {"__name__": "test-row-1", **_dataset_rows((1,))[0]},
    ],
    ids=["form-defaults", "test-row-0", "test-row-1"],
)
def test_known_input_app_path_equals_model_path(
    app_artifact, direct_artifact, record
) -> None:
    values = {k: v for k, v in record.items() if not k.startswith("__")}
    frame = build_feature_frame(values)
    app_probability = float(app_artifact.predict_positive_proba(frame)[0])
    model_frame = _hand_built_patient_frame(values)
    model_probability = float(
        direct_artifact.predict_positive_proba(model_frame)[0]
    )
    assert 0.0 <= app_probability <= 1.0
    assert model_probability == pytest.approx(app_probability, abs=1e-9), (
        f"serving path {app_probability} vs model path {model_probability} "
        f"for known input {record['__name__']}"
    )


def test_known_dataset_rows_reproduce_recorded_probabilities(
    app_artifact,
) -> None:
    """Two held-out rows carry regression-pinned exact probabilities."""
    from heart.data.split import load_split_frames

    frame = load_split_frames(version="v1")[1]
    probabilities = []
    for index in (0, 1):
        row = frame.iloc[index][list(FEATURE_COLUMNS)].to_dict()
        row["FastingBS"] = int(row["FastingBS"])
        probabilities.append(
            float(app_artifact.predict_positive_proba(_hand_built_patient_frame(row))[0])
        )
    assert probabilities == pytest.approx([0.0, 0.4615], abs=5e-4), (
        f"known-input probabilities drifted: {probabilities}"
    )


# ---------------------------------------------------------------------------
# 3. Predictions match the training-time evaluation
# ---------------------------------------------------------------------------


def _leaderboard_winner_block() -> dict[str, str]:
    """Parse the committed leaderboard's 'Winner selection (S06)' block."""
    markdown = (PROJECT_ROOT / "reports/leaderboard.md").read_text(encoding="utf-8")
    match = re.search(
        r"## Winner selection \(S06\)\n\n(.*?)(?:\n## |\Z)", markdown, re.DOTALL
    )
    assert match, "winner-selection section missing from reports/leaderboard.md"
    return {"section": match.group(1), "full": markdown}


def test_artifact_metadata_matches_committed_selection_report() -> None:
    """The serialized artifact embeds the training-time selection truth."""
    leaderboard = _leaderboard_winner_block()["section"]
    artifact = load_winning_artifact(REAL_ARTIFACT).unwrap()
    metadata = artifact.metadata

    assert "Random Forest" in leaderboard
    assert "NO-SHIP" in leaderboard and metadata.ship is False
    threshold = re.search(r"calibrated threshold: \*\*([\d.]+)\*\*", leaderboard)
    assert threshold and pytest.approx(
        metadata.threshold, abs=5e-5
    ) == float(threshold.group(1))
    objective = re.search(r"objective `([a-z0-9_]+)`", leaderboard)
    assert objective and objective.group(1) == metadata.threshold_objective
    briers = re.search(
        r"calibration Brier: ([\d.]+) → ([\d.]+)", leaderboard
    )
    assert briers
    assert float(briers.group(1)) == pytest.approx(metadata.brier_before, abs=5e-4)
    assert float(briers.group(2)) == pytest.approx(metadata.brier_after, abs=5e-4)
    assert "not_significantly_better_than_baseline" in " ".join(metadata.flags)
    assert "calibration_did_not_improve_brier" in " ".join(metadata.flags)


def test_calibrated_probabilities_reproduce_the_selection_objective(
    app_artifact,
) -> None:
    """Scoring with the app path on the selection carve achieves the
    committed objective value (f1 = 0.860335 at threshold 0.33)."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    from heart.config import RANDOM_SEED
    from heart.data.split import load_split_frames

    train, _test = load_split_frames(version="v1")
    features_x = train[list(FEATURE_COLUMNS)]
    labels_y = train[TARGET_COLUMN].astype(int)
    _x_rest, x_val, _y_rest, y_val = train_test_split(
        features_x, labels_y,
        test_size=0.20, random_state=RANDOM_SEED, stratify=labels_y,
    )
    probabilities = app_artifact.predict_positive_proba(x_val)
    flagged = (probabilities >= app_artifact.metadata.threshold).astype(int)
    recorded_f1 = 0.860335
    assert f1_score(y_val, flagged) == pytest.approx(recorded_f1, abs=0.005), (
        "the serving artifact did not reproduce the committed selection f1"
    )


def test_held_out_evaluation_sits_in_the_leaderboard_family(app_artifact) -> None:
    """Test-split metrics through the app path stay in the RF family and
    are recorded verbatim in reports/app_verification.md.

    Tolerances are family-level, deliberately not bit-matched to the
    leaderboard row: the serving artifact is refit on the selection's
    x_train carve and isotonic-calibrated, whereas the leaderboard row is
    the uncalibrated final-run pipeline fitted on the full training
    portion — the two would only match bit-exactly if the artifact lied
    about its own fit.
    """
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        roc_auc_score,
    )

    from heart.data.split import load_split_frames

    _, test = load_split_frames(version="v1")
    probabilities = app_artifact.predict_positive_proba(test[list(FEATURE_COLUMNS)])
    labels = test[TARGET_COLUMN].astype(int).to_numpy()
    flagged = (probabilities >= app_artifact.metadata.threshold).astype(int)

    roc_auc = roc_auc_score(labels, probabilities)
    brier = brier_score_loss(labels, probabilities)
    accuracy = accuracy_score(labels, flagged)
    # winner RF leaderboard row: ROC-AUC 0.9353, Brier 0.0940, Accuracy 0.8859
    assert roc_auc == pytest.approx(0.9353, abs=0.02), f"roc-auc {roc_auc:.4f}"
    assert brier == pytest.approx(0.0940, abs=0.05), f"brier {brier:.4f}"
    assert accuracy >= 0.85, f"accuracy@threshold {accuracy:.4f}"


# ---------------------------------------------------------------------------
# 4. Both flows render against the real artifact (AppTest)
# ---------------------------------------------------------------------------


def _render(monkeypatch: pytest.MonkeyPatch, page: str, env: dict[str, str] | None = None):
    import streamlit.testing.v1 as st_testing

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    clear_artifact_cache()
    clear_explore_cache()
    test = st_testing.AppTest.from_file(
        str(PROJECT_ROOT / page), default_timeout=300
    )
    try:
        test.run()
    finally:
        monkeypatch.undo()
        clear_artifact_cache()
        clear_explore_cache()
    return test


def test_home_flow_renders_serving_banner_with_real_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = _render(monkeypatch, "app/Home.py")
    assert not test.exception, str(test.exception)
    texts = " ".join(t.value for t in test.success)
    assert "loaded and validated" in texts.lower()
    metrics = {m.label: m.value for m in test.get("metric")}
    assert metrics.get("Serving model") == "random-forest"
    assert not list(test.error), "the healthy path must not show error banners"


def test_explore_flow_renders_all_insight_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = _render(monkeypatch, "app/pages/1_Explore.py")
    assert not test.exception, str(test.exception)
    header_texts = [h.value for h in test.header] + [h.value for h in test.subheader]
    joined = " ".join(header_texts)
    for group_marker in ("Distributions", "Correlation view", "Class balance", "Per-feature insight"):
        assert group_marker in joined, f"insight group missing: {group_marker}"
    metrics = {m.label: m.value for m in test.get("metric")}
    assert metrics.get("No disease (0)") == "410"
    assert metrics.get("Disease (1)") == "508"
    assert metrics.get("Positive prevalence") == "0.5534"
    assert list(test.dataframe), "correlation + insight tables must render"
    # The S01 impossible-zero audit is part of the flow, not decoration.
    assert any("impossible zero" in c.value.lower() for c in test.caption)
    assert not list(test.error)


def test_predict_flow_returns_probability_equal_to_direct_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test = _render(monkeypatch, "app/pages/2_Predict.py")
    assert not test.exception, str(test.exception)

    test.checkbox[0].check().run()
    buttons = list(test.get("button"))
    assert buttons, "the Predict submit button must render"
    buttons[-1].click().run()
    assert not test.exception, str(test.exception)

    metrics = {m.label: m.value for m in test.get("metric")}
    assert "Predicted probability of heart disease" in metrics
    served = float(metrics["Predicted probability of heart disease"])

    frame = build_feature_frame(_form_defaults())
    direct = float(load_winning_artifact(REAL_ARTIFACT).unwrap().predict_positive_proba(frame)[0])
    assert direct == pytest.approx(served, abs=0.5e-4 + 1e-12), (
        "the submitted form must return the model's own probability"
    )
    # Disclaimer appears next to the returned probability.
    assert "not a medical device" in " ".join(w.value for w in test.warning).lower()


# ---------------------------------------------------------------------------
# 5. Error paths behave (named friendly errors, never a white screen)
# ---------------------------------------------------------------------------


def _corrupt_artifact_path(tmp_path: Path, *, version: int | None = None) -> str:
    """One real-artifact copy, corrupted in either of two named ways."""
    if version is None:
        path = tmp_path / "corrupt.pkl"
        path.write_bytes(b"this stream is no pickle")
        return str(path)
    payload = pickle.loads(REAL_ARTIFACT.read_bytes())
    payload["format_version"] = version
    future_path = tmp_path / "future.pkl"
    future_path.write_bytes(pickle.dumps(payload))
    return str(future_path)


def test_home_renders_friendly_error_for_corrupt_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.lib import artifact_loader as loader_mod

    monkeypatch.setattr(loader_mod, "DEFAULT_ARTIFACT_PATH", _corrupt_artifact_path(tmp_path))
    test = _render(monkeypatch, "app/Home.py")
    assert not test.exception, str(test.exception)
    errors = " ".join(e.value for e in test.error)
    assert "artifact-corrupt" in errors
    assert "refuses to serve" in errors
    assert "Traceback" not in errors
    assert not any("loaded and validated" in t.value for t in test.success)


def test_predict_renders_friendly_error_for_missing_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.lib import artifact_loader as loader_mod

    monkeypatch.setattr(
        loader_mod, "DEFAULT_ARTIFACT_PATH", str(tmp_path / "absent.pkl")
    )
    test = _render(monkeypatch, "app/pages/2_Predict.py")
    assert not test.exception, str(test.exception)
    errors = " ".join(e.value for e in test.error)
    assert "artifact-missing" in errors
    assert not list(test.get("selectbox")), "no prediction form without an artifact"
    assert not list(test.get("number_input"))


def test_home_renders_named_error_for_schema_incompatible_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.lib import artifact_loader as loader_mod

    monkeypatch.setattr(
        loader_mod, "DEFAULT_ARTIFACT_PATH", _corrupt_artifact_path(tmp_path, version=999)
    )
    test = _render(monkeypatch, "app/Home.py")
    assert not test.exception, str(test.exception)
    errors = " ".join(e.value for e in test.error)
    assert "artifact-incompatible" in errors
    assert "upgrade" in errors or "re-run" in errors.lower()


def test_explore_renders_friendly_error_for_missing_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    test = _render(
        monkeypatch,
        "app/pages/1_Explore.py",
        env={"EXPLORE_DATA_PATH": str(tmp_path / "heart.csv")},
    )
    assert not test.exception, str(test.exception)
    errors = " ".join(e.value for e in test.error)
    assert "data-missing" in errors
    assert not list(test.header), "no insight groups render on a failed load"


def test_serving_path_refuses_reordered_description(app_artifact) -> None:
    """The runtime schema guard rejects reordered/extra columns loudly."""
    from heart.serving.artifact import FeatureSchemaError

    values = _form_defaults()
    frame = build_feature_frame(values)
    shuffled = frame[list(reversed(list(frame.columns)))]
    with pytest.raises(ArtifactError):
        app_artifact.predict_positive_proba(shuffled)
    with pytest.raises(FeatureSchemaError):
        extra = frame.copy()
        extra["Bogus"] = 1
        app_artifact.predict_positive_proba(extra)
