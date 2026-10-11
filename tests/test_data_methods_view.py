"""Data & Methods view contract tests (M005/S02/T01).

Mirroring ``tests/test_runs_page.py`` section 1, these tests pin the
never-raise boundary of :mod:`app.lib.data_methods_view` — the contract
module between the dashboard's Data & Methods page and the S01-generated
artifacts under ``reports/data-methods/``:

1. **Populated artifacts** — a tmp artifact directory built from the real
   :func:`heart.reporting.data_methods.generate_report` (small fixture
   dataset) yields ``ok`` with the ``meta.json`` payload, all four stage
   payloads, the existing PNG figure paths, the loaded manifest and the
   clean verified digest mapping.
2. **Named error kinds** — a missing directory, a missing
   ``manifest.json``, a missing ``meta.json``, invalid JSON in a stage
   file, a missing expected stage file, a missing referenced PNG figure,
   and a tampered artifact byte (``verify_manifest`` raising
   :class:`ProvenanceDigestMismatchError`) each yield the *named* error
   kind with a plain-language message naming the documented regeneration
   command ``PYTHONPATH=src python -m heart.reporting.data_methods`` —
   and no exception ever escapes the loader.
3. **Environment seam** — ``DATA_METHODS_DIR`` points the loader at a tmp
   artifact directory without an explicit argument; an explicit argument
   wins over the env override (mirroring ``RUNS_TRACKING_DIR``).
4. **Module hygiene** — the module imports without streamlit (MEM047).

Matplotlib is forced to the headless Agg backend before the engine imports
it (same guard as ``tests/test_data_methods_report.py``), and every write
in this module lands under ``tmp_path`` — never the committed ``reports/``
tree.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.lib import data_methods_view  # noqa: E402
from app.lib.data_methods_view import (  # noqa: E402
    DATA_METHODS_DIR_ENV,
    ERROR_KIND_ARTIFACTS_CORRUPT,
    ERROR_KIND_ARTIFACTS_MISSING,
    ERROR_KIND_ARTIFACTS_UNKNOWN,
    REGENERATION_COMMAND,
    DataMethodsView,
    existing_figure_paths,
    load_data_methods_view,
)
from heart.reporting.data_methods import (  # noqa: E402
    verify_manifest,
)
from heart.reporting.data_methods.report import STAGE_ARTIFACTS  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture frame + populated artifact dir (mirrors test_data_methods_report.py)
# ---------------------------------------------------------------------------


def make_frame(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Deterministic fixture frame in the raw dataset's column shape."""
    rng = np.random.default_rng(seed)
    severity = np.linspace(0.0, 1.0, n)
    cholesterol = rng.integers(120, 400, size=n)
    cholesterol[:4] = 0  # declared sentinels (zero-as-missing policy)
    restingbp = rng.integers(90, 190, size=n)
    restingbp[2] = 0  # declared sentinel
    return pd.DataFrame(
        {
            "Age": rng.integers(30, 81, size=n),
            "Sex": rng.choice(["M", "F"], size=n),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], size=n),
            "RestingBP": restingbp,
            "Cholesterol": cholesterol,
            "FastingBS": rng.integers(0, 2, size=n),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], size=n),
            "MaxHR": rng.integers(90, 202, size=n),
            "ExerciseAngina": np.where(rng.random(n) < severity, "Y", "N"),
            "Oldpeak": np.round(severity * 4.0, 2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], size=n),
            "HeartDisease": np.where(rng.random(n) < severity, 1, 0),
        }
    )


@pytest.fixture()
def generated(tmp_path):
    """Generate the real report artifacts from the fixture into tmp_path."""
    artifact_dir = tmp_path / "data-methods"
    generate = pytest.importorskip(
        "heart.reporting.data_methods", reason="report machinery unavailable"
    )
    generate.generate_report(
        make_frame(),
        output_dir=artifact_dir,
        report_path=tmp_path / "data-methods.md",
    )
    return {"artifact_dir": artifact_dir, "tmp_path": tmp_path}


# ---------------------------------------------------------------------------
# 1. Populated artifacts: the full view
# ---------------------------------------------------------------------------


class TestPopulatedArtifacts:
    """A generated tmp artifact directory yields the full view, never raises."""

    def test_ok_with_meta_stage_and_figure_data(self, generated) -> None:
        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert isinstance(view, DataMethodsView)
        assert view.ok is True
        assert view.error_kind is None
        assert view.error_message is None
        # meta.json payload: dataset provenance the page's provenance line needs.
        assert isinstance(view.meta, dict)
        assert view.meta["row_count"] == 120
        assert view.meta["seed"] == 42
        assert view.meta["target_column"] == "HeartDisease"
        assert "RestingBP" in view.meta["declared_zero_as_missing_columns"]
        assert "Cholesterol" in view.meta["declared_zero_as_missing_columns"]
        # All four stage payloads loaded.
        assert set(view.stages) == set(STAGE_ARTIFACTS) - {"meta"}
        for stage_key in view.stages:
            assert isinstance(view.stages[stage_key], dict)
        # stage1 carries the hygiene tables the page renders.
        assert "missingness" in view.stages["stage1_hygiene"]
        assert "impossible_values" in view.stages["stage1_hygiene"]
        # The attempted dir is the resolved artifact directory.
        assert view.attempted_dir == str(generated["artifact_dir"])

    def test_figures_exist_and_are_pngs(self, generated) -> None:
        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is True
        assert len(view.figure_paths) > 0
        for path in view.figure_paths:
            assert path.suffix == ".png"
            assert path.is_file()
            assert path.parent == generated["artifact_dir"]
        names = {path.name for path in view.figure_paths}
        assert "stage1_missingness_patterns.png" in names
        assert "stage3_correlation_heatmap.png" in names

    def test_manifest_loaded_and_verification_clean(self, generated) -> None:
        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is True
        assert view.manifest is not None
        assert view.manifest["algorithm"] == "sha256"
        assert view.manifest["artifact_count"] > 0
        # Verified digests come from verify_manifest and match the manifest.
        assert view.verified_digests == view.manifest["artifacts"]
        # The S01 engine itself re-verifies the same directory cleanly.
        assert verify_manifest(generated["artifact_dir"]) == view.verified_digests


# ---------------------------------------------------------------------------
# 2. Missing artifacts: named kinds, friendly messages
# ---------------------------------------------------------------------------


class TestArtifactsMissing:
    """A missing directory / manifest / meta maps to ``artifacts-missing``."""

    def test_directory_does_not_exist(self, tmp_path) -> None:
        view = load_data_methods_view(artifact_dir=tmp_path / "no-such-dir")

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_MISSING
        assert REGENERATION_COMMAND in view.error_message
        assert str(tmp_path / "no-such-dir") in view.error_message

    def test_path_is_a_file_not_a_directory(self, tmp_path) -> None:
        file_path = tmp_path / "not-a-dir"
        file_path.write_text("plainly not an artifact dir", encoding="utf-8")

        view = load_data_methods_view(artifact_dir=file_path)

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_MISSING
        assert REGENERATION_COMMAND in view.error_message

    def test_missing_manifest_json(self, generated) -> None:
        (generated["artifact_dir"] / "manifest.json").unlink()

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_MISSING
        assert "manifest.json" in view.error_message
        assert REGENERATION_COMMAND in view.error_message

    def test_missing_meta_json(self, generated) -> None:
        (generated["artifact_dir"] / "meta.json").unlink()

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_MISSING
        assert "meta.json" in view.error_message
        assert REGENERATION_COMMAND in view.error_message


# ---------------------------------------------------------------------------
# 3. Corrupt artifacts: named kinds, friendly messages
# ---------------------------------------------------------------------------


class TestArtifactsCorrupt:
    """Invalid JSON, missing stage files and digest mismatch map to
    ``artifacts-corrupt`` — the tamper signal, never a partial render."""

    def test_invalid_json_in_a_stage_file(self, generated) -> None:
        stage_path = generated["artifact_dir"] / "stage2_univariate.json"
        stage_path.write_text("{not json", encoding="utf-8")

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_CORRUPT
        assert "stage2_univariate.json" in view.error_message
        assert REGENERATION_COMMAND in view.error_message

    def test_missing_expected_stage_file(self, generated) -> None:
        (generated["artifact_dir"] / "stage4_assumptions.json").unlink()

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_CORRUPT
        assert "stage4_assumptions.json" in view.error_message
        assert REGENERATION_COMMAND in view.error_message

    def test_tampered_artifact_byte_is_a_digest_mismatch(self, generated) -> None:
        """The tamper signal: verify_manifest raises
        ProvenanceDigestMismatchError, and the view maps it to corrupt."""
        png_path = generated["artifact_dir"] / "stage3_correlation_heatmap.png"
        data = bytearray(png_path.read_bytes())
        data[0] ^= 0xFF
        png_path.write_bytes(bytes(data))
        with pytest.raises(Exception):  # noqa: B017, PT011 - the engine raises
            verify_manifest(generated["artifact_dir"])

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_CORRUPT
        assert REGENERATION_COMMAND in view.error_message
        assert view.verified_digests is None

    def test_missing_referenced_figure_is_corrupt(self, generated, tmp_path) -> None:
        """Defense-in-depth: a stage payload referencing a PNG that does not
        exist on disk is corrupt even if the manifest itself would verify
        (manifest here hand-built to list only the JSONs)."""
        import hashlib

        artifact_dir = generated["artifact_dir"]
        # Hand-build a manifest covering only the JSON artifacts so the
        # provenance check passes, then point a stage payload at a missing PNG.
        entries = {}
        for name in ("meta.json", "stage1_hygiene.json",
                     "stage2_univariate.json", "stage3_bivariate.json",
                     "stage4_assumptions.json"):
            data = (artifact_dir / name).read_bytes()
            entries[name] = hashlib.sha256(data).hexdigest()
        payload_dir = generated["tmp_path"] / "hand-built"
        payload_dir.mkdir()
        for name in ("meta.json", "stage1_hygiene.json", "stage2_univariate.json",
                     "stage3_bivariate.json", "stage4_assumptions.json"):
            (payload_dir / name).write_bytes((artifact_dir / name).read_bytes())
        payload_dir.joinpath("manifest.json").write_text(
            json.dumps({"algorithm": "sha256", "artifact_count": len(entries),
                        "artifacts": entries}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        view = load_data_methods_view(artifact_dir=payload_dir)

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_CORRUPT
        assert "missing from the artifact directory" in view.error_message
        assert REGENERATION_COMMAND in view.error_message


# ---------------------------------------------------------------------------
# 4. Never-raise boundary
# ---------------------------------------------------------------------------


class TestNeverRaise:
    """Every outcome is a returned kind; no input shape raises."""

    def test_load_never_raises_on_any_directory_shape(self, tmp_path) -> None:
        file_path = tmp_path / "a-file"
        file_path.write_text("x", encoding="utf-8")
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        targets = [tmp_path / "missing-dir", file_path, empty_dir]
        for target in targets:
            view = load_data_methods_view(artifact_dir=target)
            assert isinstance(view, DataMethodsView)
            assert isinstance(view.ok, bool)
            assert REGENERATION_COMMAND in view.error_message

    def test_unexpected_failure_maps_to_artifacts_unknown(
        self, generated, monkeypatch
    ) -> None:
        """Anything unexpected from the provenance layer lands in
        ``artifacts-unknown`` rather than escaping the loader."""

        def explode(artifact_dir):
            raise RuntimeError("provenance backend exploded")

        monkeypatch.setattr(data_methods_view.provenance, "verify_manifest", explode)

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_UNKNOWN
        assert "RuntimeError" in view.error_message
        assert REGENERATION_COMMAND in view.error_message


# ---------------------------------------------------------------------------
# 5. Environment seam (DATA_METHODS_DIR)
# ---------------------------------------------------------------------------


class TestDataMethodsDirEnvSeam:
    """``DATA_METHODS_DIR`` points the loader at a tmp dir without an arg."""

    def test_env_override_yields_populated_artifacts(self, generated, monkeypatch) -> None:
        monkeypatch.setenv(DATA_METHODS_DIR_ENV, str(generated["artifact_dir"]))

        view = load_data_methods_view()

        assert view.ok is True
        assert view.meta["row_count"] == 120
        assert view.attempted_dir == str(generated["artifact_dir"])

    def test_env_override_missing_dir_yields_artifacts_missing(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(DATA_METHODS_DIR_ENV, str(tmp_path / "absent"))

        view = load_data_methods_view()

        assert view.ok is False
        assert view.error_kind == ERROR_KIND_ARTIFACTS_MISSING

    def test_explicit_arg_wins_over_env_override(
        self, generated, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv(DATA_METHODS_DIR_ENV, str(tmp_path / "absent"))

        view = load_data_methods_view(artifact_dir=generated["artifact_dir"])

        assert view.ok is True


# ---------------------------------------------------------------------------
# 6. Figure helper: only existing PNG paths
# ---------------------------------------------------------------------------


class TestExistingFigurePaths:
    """The helper exposes PNG paths only for files that exist."""

    def test_filters_nonexistent_names(self, generated) -> None:
        real = "stage1_missingness_patterns.png"
        names = [real, "no-such-figure.png", "stage3_correlation_heatmap.png"]

        paths = existing_figure_paths(generated["artifact_dir"], names)

        assert [path.name for path in paths] == sorted(
            [real, "stage3_correlation_heatmap.png"]
        )

    def test_enumerates_all_existing_pngs_sorted(self, generated) -> None:
        paths = existing_figure_paths(generated["artifact_dir"])

        assert all(path.is_file() and path.suffix == ".png" for path in paths)
        assert [p.name for p in paths] == sorted(p.name for p in paths)
        assert len(paths) > 0

    def test_missing_directory_yields_no_paths(self, tmp_path) -> None:
        assert existing_figure_paths(tmp_path / "absent") == ()


# ---------------------------------------------------------------------------
# 7. Module hygiene: plain Python, no streamlit dependency
# ---------------------------------------------------------------------------


def test_data_methods_view_module_imports_without_streamlit() -> None:
    """The contract module must be pytest-testable without Streamlit (MEM047)."""
    code = (
        "import sys\n"
        "import app.lib.data_methods_view\n"
        "assert 'streamlit' not in sys.modules, 'streamlit leaked into "
        "data_methods_view'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PROJECT_ROOT),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
