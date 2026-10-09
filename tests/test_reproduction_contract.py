"""Reproduction contract tests (S08/T04).

These tests protect the project's central claim — "a stranger with a clean
environment can regenerate the published leaderboard with one documented
command" — as an executable contract rather than a prose assertion:

1. ``reports/reproduction.md`` exists and records the evidence of the actual
   clean-environment reproduction run (isolated venv, pinned stack, exit 0).
2. Every leaderboard model's published numbers in ``reports/leaderboard.md``
   match the numbers recorded beside them in ``reports/reproduction.md``, and
   the recorded deviation for each metric is zero (any drift would fail here).
3. ``scripts/reproduce.sh`` structurally guarantees a fresh environment: it
   creates its own virtualenv and never runs the pipeline with the developer
   interpreter.
4. The dataset integrity gate runs *before* any training step, so a corrupt
   or substituted dataset aborts the pipeline before models are fit.
5. ``make reproduce`` / the README document the one-command entry point.
6. The dataset pin still holds in the developer environment right now
   (``scripts/verify_dataset.py`` exits 0 on the tracked data).

These are contract checks over tracked artifacts; the heavyweight
negative-path behavior of the gate itself (tampered data, missing raw file,
split drift) is covered by ``tests/test_reproduction_gate.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_PATH = REPO_ROOT / "reports" / "leaderboard.md"
REPRODUCTION_PATH = REPO_ROOT / "reports" / "reproduction.md"
REPRODUCE_SH_PATH = REPO_ROOT / "scripts" / "reproduce.sh"
README_PATH = REPO_ROOT / "README.md"
MAKEFILE_PATH = REPO_ROOT / "Makefile"
GATE_TEST_PATH = REPO_ROOT / "tests" / "test_reproduction_gate.py"

# Leaderboard columns (after Rank/Model/Type/Family/Device) whose equality the
# contract checks. N is an int; everything else is a 4-decimal float.
METRIC_COLUMNS = (
    "ROC-AUC",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "PR-AUC",
    "Specificity",
    "NPV",
    "Prevalence",
    "Brier",
    "ECE",
)


def _table_rows(path: Path, section_header: str) -> list[list[str]]:
    """Return the cell lists of the first markdown table under a section."""
    text = path.read_text(encoding="utf-8")
    section = text.split(section_header, 1)[1]
    lines = section.splitlines()
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break  # table ended
            continue  # still inside prose between header and table
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue  # separator row
        rows.append(cells)
    return rows


def parse_leaderboard(path: Path) -> dict[str, dict[str, str]]:
    """Parse the leaderboard metric table into {model: {column: value}}."""
    rows = _table_rows(path, "## Leaderboard")
    assert len(rows) >= 2, f"leaderboard table missing in {path}"
    header = rows[0]
    index = {name: i for i, name in enumerate(header)}
    parsed: dict[str, dict[str, str]] = {}
    for cells in rows[1:]:
        model = cells[index["Model"]]
        parsed[model] = {name: cells[i] for name, i in index.items()}
    return parsed


def parse_comparison_table(path: Path) -> list[dict[str, str]]:
    """Parse the Published-vs-Reproduced comparison table in reproduction.md."""
    rows = _table_rows(path, "## Published vs reproduced numbers")
    assert len(rows) >= 2, f"comparison table missing in {path}"
    header = rows[0]
    return [dict(zip(header, cells)) for cells in rows[1:]]


# ---------------------------------------------------------------------------
# 1. The evidence report exists and records the actual clean run
# ---------------------------------------------------------------------------
def test_reproduction_report_exists_and_records_clean_run() -> None:
    text = REPRODUCTION_PATH.read_text(encoding="utf-8")
    for needle in (
        "## Environment (fresh, not the development one)",
        "## Published vs reproduced numbers",
        "## Deviations",
        "exit code 0",
    ):
        assert needle in text, f"reproduction.md lacks required section: {needle}"
    # Fresh-environment proof: an isolated venv path that is NOT the repo root.
    assert re.search(r"venv[^\n]*\.repro-venv|REPRO_VENV", text), (
        "reproduction.md must record the isolated reproduction venv"
    )


# ---------------------------------------------------------------------------
# 2. Published vs reproduced numbers agree, deviation recorded as zero
# ---------------------------------------------------------------------------
def test_published_and_reproduced_numbers_agree_for_every_model() -> None:
    leaderboard = parse_leaderboard(LEADERBOARD_PATH)
    comparison = parse_comparison_table(REPRODUCTION_PATH)

    by_model = {row["Model"]: row for row in comparison}
    missing = set(leaderboard) - set(by_model)
    assert not missing, f"reproduction.md is missing comparison rows for: {sorted(missing)}"
    extra = set(by_model) - set(leaderboard)
    assert not extra, f"reproduction.md lists models absent from the leaderboard: {sorted(extra)}"

    for model, board_row in sorted(leaderboard.items()):
        row = by_model[model]
        for metric in METRIC_COLUMNS:
            published = float(row[f"Published {metric}"])
            reproduced = float(row[f"Reproduced {metric}"])
            recorded_deviation = float(row[f"Deviation ({metric})"])
            board_value = float(board_row[metric])
            # The published report must agree with the current leaderboard...
            assert published == pytest.approx(board_value, abs=1e-9), (
                f"{model}/{metric}: reproduction.md published value {published} "
                f"!= leaderboard value {board_value}"
            )
            # ...the reproduced value must match the published one...
            assert reproduced == pytest.approx(published, abs=1e-9), (
                f"{model}/{metric}: reproduced {reproduced} != published {published}"
            )
            # ...and the recorded deviation must be the observed difference.
            assert recorded_deviation == pytest.approx(reproduced - published, abs=1e-9), (
                f"{model}/{metric}: recorded deviation {recorded_deviation} "
                f"is not the observed difference"
            )
        assert int(row["Published N"]) == int(board_row["N"]) == int(row["Reproduced N"])


def test_all_recorded_deviations_are_zero() -> None:
    comparison = parse_comparison_table(REPRODUCTION_PATH)
    nonzero = []
    for row in comparison:
        for metric in METRIC_COLUMNS:
            if float(row[f"Deviation ({metric})"]) != 0.0:
                nonzero.append((row["Model"], metric, row[f"Deviation ({metric})"]))
    assert not nonzero, f"unexpected nonzero recorded deviations: {nonzero}"


# ---------------------------------------------------------------------------
# 3. The reproduction cannot silently reuse the development environment
# ---------------------------------------------------------------------------
def test_reproduce_script_creates_isolated_venv() -> None:
    script = REPRODUCE_SH_PATH.read_text(encoding="utf-8")
    # A dedicated venv is created and every pipeline stage runs with it.
    assert 'python3 -m venv "$VENV_PATH"' in script
    assert 'PYTHON="$VENV_PATH/bin/python"' in script
    for stage in ("heart.models.run_battery", "heart.eval.selection",
                  "heart.reporting.leaderboard", "scripts/verify_dataset.py"):
        stage_line = next(
            line for line in script.splitlines()
            if stage in line and not line.strip().startswith("#")
        )
        assert '"$PYTHON"' in stage_line, f"stage {stage} does not run on the isolated venv: {stage_line}"
    # No unpinned interpreter escape hatch: no bare `pip install`/`python -m`.
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('echo'):
            continue
        assert not re.match(r"^(pip install|python3? -m pip install)\b", stripped), (
            f"reproduce.sh runs pip outside the isolated venv: {line}"
        )


# ---------------------------------------------------------------------------
# 4. The integrity gate precedes any training step
# ---------------------------------------------------------------------------
def test_integrity_gate_runs_before_training() -> None:
    script = REPRODUCE_SH_PATH.read_text(encoding="utf-8")
    gate_index = script.index("scripts/verify_dataset.py")
    battery_index = script.index("heart.models.run_battery")
    assert gate_index < battery_index, (
        "reproduce.sh must verify dataset integrity before running the battery"
    )


# ---------------------------------------------------------------------------
# 5. The one-command entry point is documented
# ---------------------------------------------------------------------------
def test_makefile_and_readme_document_the_entry_point() -> None:
    makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    assert "reproduce:" in makefile and "./$(REPRO_SCRIPT)" in makefile, (
        "Makefile must expose `make reproduce` delegating to scripts/reproduce.sh"
    )
    readme = README_PATH.read_text(encoding="utf-8")
    assert "make reproduce" in readme, "README must document the one-command reproduction"


def test_negative_path_suite_is_still_present() -> None:
    # The gate's negative paths (tampered/missing raw, split drift) are owned
    # by the T03 suite; the contract depends on them staying in place.
    text = GATE_TEST_PATH.read_text(encoding="utf-8")
    for needle in ("rc == 2", "rc == 3"):
        assert needle in text, f"test_reproduction_gate.py lost the {needle} assertion"


# ---------------------------------------------------------------------------
# 6. Live integrity check: the pin still holds in the developer environment
# ---------------------------------------------------------------------------
def test_dataset_integrity_gate_passes_on_tracked_data() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_dataset.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "src", "HOME": str(Path.home())},
    )
    assert result.returncode == 0, (
        f"integrity gate failed on tracked data (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
