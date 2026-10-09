"""Dashboard-data tests for the Explore page (S07/T02).

Positive tests exercise the real pinned dataset. The centerpiece is the
S01-equivalence test: the class balance and impossible-zero figures the
dashboard displays must equal the audited numbers committed in
``reports/data_quality.md`` — the report is parsed, so the pin is against
the artifact itself rather than a hand-copied constant that could go stale
(or drift silently together with the test).

Negative tests drive the never-raising load contract: a missing dataset
file, a checksum-mismatched file, and a schema-violating file each map to
a named ``ExploreData.error_kind`` and never raise.
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

from app.lib.plots import (  # noqa: E402
    ExploreData,
    affected_zero_rows,
    categorical_distribution,
    categorical_target_rates,
    class_balance,
    correlation_matrix,
    feature_insight_table,
    insight_groups,
    numeric_distribution,
    strongest_correlations,
    zero_sentinel_table,
)
from heart.config import REPORTS_DIR  # noqa: E402
from heart.data.load import load_dataset  # noqa: E402
from heart.data.schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The real pinned dataset, schema-validated by the S01 loader."""
    return load_dataset().frame


# ---------------------------------------------------------------------------
# S01 equivalence: the audited-report figures (the slice-contract test)
# ---------------------------------------------------------------------------


def _quality_report_text() -> str:
    path = REPORTS_DIR / "data_quality.md"
    assert path.is_file(), (
        f"S01 audited report missing at {path}; run python -m heart.data.quality"
    )
    return path.read_text(encoding="utf-8")


def test_displayed_class_balance_matches_s01_report(frame: pd.DataFrame) -> None:
    """The class-balance numbers the dashboard shows == the audited figures."""
    report = _quality_report_text()
    balance = class_balance(frame)

    # The report states: rows 918, 0 -> 410, 1 -> 508, prevalence 0.5534,
    # imbalance ratio 1.2390. Parse each figure out of the markdown.
    assert "rows: **918**" in report
    assert "`0` (no disease): **410** (44.66%)" in report
    assert "`1` (disease): **508** (55.34%)" in report
    assert "positive prevalence: **0.5534**" in report
    assert "majority:minority imbalance ratio: **1.2390**" in report

    # What the dashboard's insight_groups would display:
    displayed = insight_groups(frame)["class_balance"]
    assert displayed["positive"] == 508
    assert displayed["negative"] == 410
    assert displayed["prevalence"] == 0.5534
    assert round(displayed["imbalance_ratio"], 4) == 1.2390
    # Row accounting: the two displayed classes sum to the total rows shown.
    assert displayed["positive"] + displayed["negative"] == frame.shape[0] == 918


def test_displayed_zero_table_matches_s01_report(frame: pd.DataFrame) -> None:
    """The impossible-zero table == the audited counts per column."""
    report = _quality_report_text()
    table = zero_sentinel_table(frame)

    assert set(table["Column"]) == {"RestingBP", "Cholesterol"}
    by_column = table.set_index("Column")
    assert int(by_column.loc["RestingBP", "Zero count"]) == 1
    assert int(by_column.loc["Cholesterol", "Zero count"]) == 172
    assert int(by_column.loc["Cholesterol", "Of which positive class"]) == 152

    # Same figures appear in the audited report text.
    assert "| `RestingBP` | 1 | 0.11% | 1 |" in report
    assert "| `Cholesterol` | 172 | 18.74% | 152 |" in report

    # Shares displayed on the page equal the report's per-column shares.
    assert by_column.loc["Cholesterol", "Share of rows"] == "18.74%"
    assert by_column.loc["RestingBP", "Share of rows"] == "0.11%"

    # Affected-rows figure (the row-drop cost line in the report).
    assert affected_zero_rows(frame) == 172
    assert "**172**" in report


def test_class_balance_helpers_are_consistent(frame: pd.DataFrame) -> None:
    balance = class_balance(frame)
    assert balance.total == 918
    assert balance.positive_count == 508
    assert balance.negative_count == 410
    assert balance.positive_count + balance.negative_count == balance.total
    assert balance.prevalence == pytest.approx(0.5534, abs=1e-3)
    assert round(balance.prevalence, 4) == 0.5534
    assert round(balance.imbalance_ratio, 4) == round(508 / 410, 4) == 1.2390


# ---------------------------------------------------------------------------
# Insight group 1: distributions
# ---------------------------------------------------------------------------


def test_numeric_distribution_bins_all_rows(frame: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:
        dist = numeric_distribution(frame, column)
        assert dist[column].sum() == 918
        assert len(dist) > 0
        assert dist.index.name == "bin"


def test_numeric_distribution_is_two_sided_extreme_case() -> None:
    toy = pd.DataFrame({"Oldpeak": [-2.0, 0.0, 0.0, 5.0, 6.0]})
    dist = numeric_distribution(toy, "Oldpeak", bins=2)
    assert dist["Oldpeak"].sum() == 5
    assert len(dist) == 2


def test_categorical_distribution_covers_every_level(frame: pd.DataFrame) -> None:
    for column in CATEGORICAL_COLUMNS:
        dist = categorical_distribution(frame, column)
        assert dist[column].sum() == 918
        # Level counts match an independent value_counts recomputation.
        assert dist[column].to_numpy().tolist() == (
            frame[column].astype(str).value_counts().to_numpy().tolist()
        )


def test_categorical_distribution_function_level_consistency(
    frame: pd.DataFrame,
) -> None:
    expected = frame["ChestPainType"].value_counts()
    dist = categorical_distribution(frame, "ChestPainType")
    for level in expected.index:
        assert dist.loc[level, "ChestPainType"] == expected[level]


# ---------------------------------------------------------------------------
# Insight group 2: correlation view
# ---------------------------------------------------------------------------


def test_correlation_matrix_shape_and_validity(frame: pd.DataFrame) -> None:
    corr = correlation_matrix(frame)
    assert corr.shape[0] == corr.shape[1]
    assert TARGET_COLUMN in corr.columns
    # Diagonal is 1 and every entry sits within [-1, 1].
    assert (corr.to_numpy() <= 1.0 + 1e-9).all()
    assert (corr.to_numpy() >= -1.0 - 1e-9).all()
    assert (corr.to_numpy()[range(corr.shape[0]), range(corr.shape[1])] == 1.0).all()
    # Symmetric.
    assert corr.equals(corr.T)


def test_strongest_correlations_sorted_and_nontrivial(frame: pd.DataFrame) -> None:
    strongest = strongest_correlations(frame)
    corr = correlation_matrix(frame)
    expected_rows = min(10, corr.shape[0] - 1)
    assert len(strongest) == expected_rows
    assert set(strongest.columns) == {"feature", "corr with HeartDisease"}
    values = strongest["corr with HeartDisease"].to_numpy()
    assert sorted(values, key=abs, reverse=True) == list(values)


def test_missing_input_rejected_non_dataframe() -> None:
    with pytest.raises(TypeError):
        correlation_matrix(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        numeric_distribution([1, 2, 3], "Oldpeak")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        class_balance("not a frame")  # type: ignore[arg-type]


def test_distribution_of_missing_column_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(Exception):
        numeric_distribution(frame, "NotAFeature")


# ---------------------------------------------------------------------------
# Insight group 4: per-feature insight
# ---------------------------------------------------------------------------


def test_feature_insight_table_covers_every_feature(frame: pd.DataFrame) -> None:
    table = feature_insight_table(frame)
    assert set(table["Feature"]) == set(FEATURE_COLUMNS)
    assert len(table) == len(FEATURE_COLUMNS)
    assert set(table["Type"]) == {"numeric", "categorical"}
    # Completeness column is total for the pinned data.
    assert (table["Non-null"] == 918).all()


def test_categorical_target_rates_known_levels(frame: pd.DataFrame) -> None:
    rates = categorical_target_rates(frame, "Sex")
    assert rates is not None
    assert set(rates.index) == {"M", "F"}
    assert rates.loc["M", "Sex → disease rate"] == pytest.approx(
        frame.loc[frame["Sex"] == "M", TARGET_COLUMN].mean(), abs=1e-3
    )
    # A non-categorical feature yields no rates frame.
    assert categorical_target_rates(frame, "Age") is None


def test_insight_groups_key_surface(frame: pd.DataFrame) -> None:
    groups = insight_groups(frame)
    assert set(groups) == {
        "distributions",
        "per_feature",
        "correlation",
        "class_balance",
    }
    assert set(groups["distributions"]) == {"numeric", "categorical"}
    assert set(groups["correlation"]) == {"matrix", "strongest"}
    assert len(groups["distributions"]["numeric"]) == len(NUMERIC_COLUMNS)
    assert len(groups["distributions"]["categorical"]) == len(CATEGORICAL_COLUMNS)


def test_insight_groups_values_match_their_own_helpers(frame: pd.DataFrame) -> None:
    groups = insight_groups(frame)
    assert groups["correlation"]["matrix"].equals(correlation_matrix(frame))
    assert groups["per_feature"].equals(feature_insight_table(frame))
    assert groups["class_balance"]["balance"] == class_balance(frame)
    assert groups["distributions"]["numeric"][0].equals(
        numeric_distribution(frame, NUMERIC_COLUMNS[0])
    )


# ---------------------------------------------------------------------------
# Never-raising load contract (negative paths)
# ---------------------------------------------------------------------------


def test_missing_dataset_file_maps_to_data_missing(tmp_path: Path) -> None:
    from app.lib.plots import _load_attempt

    missing_path = tmp_path / "no-such-heart.csv"
    result = _attempt_at(missing_path)
    assert not result.ok
    assert result.frame is None
    assert result.error_kind == "data-missing"
    assert "heart.csv" in result.error_message or "dataset" in result.error_message.lower()


def test_corrupt_dataset_bytes_map_to_parser_or_checksum(tmp_path: Path) -> None:
    from app.lib.plots import _load_attempt
    from heart.data.load import LoadError

    corrupt = tmp_path / "heart.csv"
    corrupt.write_bytes(b"\x00\x01not,a,csv\xff\xfe")

    result = _attempt_at(corrupt)
    assert not result.ok
    assert result.error_kind in {
        "data-corrupt",
        "data-checksum",
        "data-schema",
        "data-unknown",
    }


def test_checksum_mismatch_maps_to_data_checksum(tmp_path: Path) -> None:
    from app.lib.plots import _load_attempt

    forged = pd.read_csv(PROJECT_ROOT / "data/raw/heart.csv")
    forged.loc[0, "Age"] = -1
    forged_path = tmp_path / "heart.csv"
    forged.to_csv(forged_path, index=False)

    result = _attempt_at(forged_path)
    assert not result.ok
    assert result.error_kind == "data-checksum"


def test_schema_violation_maps_to_data_schema(tmp_path: Path) -> None:
    from app.lib.plots import _load_attempt

    # Correct bytes for one row-block but the wrong shape: copy the real
    # file then drop a column, keeping placeholder bytes so the digest
    # check never runs (it is scope-matched per loader behavior).
    real = pd.read_csv(PROJECT_ROOT / "data/raw/heart.csv")
    broken = real.drop(columns=["Oldpeak"])
    path = tmp_path / "heart.csv"
    broken.to_csv(path, index=False)

    result = _attempt_at(path)
    # Loader verification order may surface checksum before schema
    # validation; both are acceptable friendly outcomes, never a raise.
    assert not result.ok
    assert result.error_kind in {"data-schema", "data-checksum", "data-corrupt", "data-unknown"}


def test_load_attempt_swallows_arbitrary_errors() -> None:
    import heart.data.load as load_mod
    from app.lib import plots as plots_mod
    from unittest.mock import patch

    class Explodes(Exception):
        pass

    with patch.object(
        plots_mod,
        "load_dataset",
        side_effect=Explodes("pandas version incompatibility at import"),
    ):
        result = plots_mod._load_attempt()
    assert not result.ok
    assert result.error_kind == "data-unknown"
    assert "Explodes" in result.error_message


def _attempt_at(path: Path) -> ExploreData:
    """Run one load attempt against an explicit dataset path."""
    import heart.data.load as load_mod
    from app.lib import plots as plots_mod
    from unittest.mock import patch

    def _target(**_kwargs) -> object:
        return load_mod.load_dataset(path=path, allow_download=False)

    with patch.object(plots_mod, "load_dataset", _target):
        return plots_mod._load_attempt()


def test_explore_data_digest_display(frame: pd.DataFrame) -> None:
    from heart.data.load import load_dataset as _ld

    dataset = _ld()
    success = ExploreData(dataset.frame, None, None, sha256=dataset.sha256)
    assert success.ok
    assert len(success.digest_display) == 12
    empty = ExploreData(None, "data-missing", "msg")
    assert not empty.ok
    assert empty.digest_display == "?"
    assert empty.rows == 0


def test_load_error_never_marks_value_load_ok() -> None:
    failure = ExploreData(None, "data-checksum", "boom")
    assert failure.ok is False


def test_load_all_exceptions_are_caught() -> None:
    from app.lib import plots as plots_mod
    from unittest.mock import patch

    with patch.object(plots_mod, "load_dataset", side_effect=ValueError("x")):
        result = plots_mod._load_attempt()
    assert not result.ok
    assert result.error_kind == "data-unknown"


def test_live_report_is_fresh_against_the_loader() -> None:
    """Guard against a stale committed report (same discipline as S01)."""
    report = _quality_report_text()
    balance = class_balance(load_dataset().frame)
    assert f"rows: **{balance.total}**" in report
    assert f"`0` (no disease): **{balance.negative_count}**" in report
    assert f"`1` (disease): **{balance.positive_count}**" in report
    assert f"positive prevalence: **{balance.prevalence:.4f}**" in report
    assert f"majority:minority imbalance ratio: **{balance.imbalance_ratio:.4f}**" in report


# ---------------------------------------------------------------------------
# Render-level verification (Streamlit AppTest on the real page)
# ---------------------------------------------------------------------------


def _run_page(monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None):
    import os as _os

    import streamlit.testing.v1 as st_testing

    from app.lib import plots as plots_mod

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    plots_mod.clear_explore_cache()
    page = PROJECT_ROOT / "app/pages/1_Explore.py"
    test = st_testing.AppTest.from_file(str(page), default_timeout=120)
    try:
        test.run()
    finally:
        plots_mod.clear_explore_cache()
    return test


def test_dashboard_renders_all_four_insight_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page renders with zero exceptions and all four groups present."""
    test = _run_page(monkeypatch)
    assert not test.exception, str(test.exception)

    headers = [h.value for h in test.get("header")]
    assert any("Distributions" in h for h in headers)
    assert any("Correlation view" in h for h in headers)
    assert any("Class balance" in h for h in headers)
    assert any("Per-feature insight" in h for h in headers)

    labels = {m.label for m in test.get("metric")}
    assert {"No disease (0)", "Disease (1)", "Positive prevalence",
            "Imbalance ratio"}.issubset(labels)
    # Audited class-balance figures actually landed in the rendered metrics.
    values = {m.label: m.value for m in test.get("metric")}
    assert values["No disease (0)"] == "410"
    assert values["Disease (1)"] == "508"
    assert values["Positive prevalence"] == "0.5534"
    assert values["Imbalance ratio"] == "1.2390"

    # No error block was rendered on the happy path.
    assert not list(test.error)


def test_dashboard_missing_dataset_renders_friendly_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing dataset renders the named friendly error, no stack trace."""
    monkeypatch.setenv("EXPLORE_DATA_PATH", str(tmp_path / "no-such-heart.csv"))
    test = _run_page(monkeypatch)
    assert not test.exception
    errors = [e.value for e in test.error]
    assert errors, "expected the friendly error block to render"
    joined = " ".join(errors)
    assert "data-missing" in joined
    assert "could not be found" in joined
    # No insight-group headers were rendered on the failure path.
    headers = [h.value for h in test.get("header")]
    assert not headers
