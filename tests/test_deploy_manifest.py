"""Guards for the Streamlit Community Cloud deploy manifest and its CI gate.

M007/S02/T03: the Predict page now imports shap at request time, so
app/requirements.txt must pin the exact shap version declared in
pyproject.toml (no drift), stay direct-deps-only (no matplotlib/seaborn —
the app charts natively), and .github/workflows/ci.yml must install + smoke
that manifest on every PR so a broken pin fails in CI, not at Cloud boot
(MEM117).

M007/S04 bugfix (2026-10-13): the deployed /Predict page crashed at boot with
``ModuleNotFoundError: mlflow`` because the app-scoped manifest never carried
``mlflow``, which the Predict page's import closure (``heart.tracking.mlflow_store``
via ``app/lib/runs_view.py``) imports at module load. ``mlflow`` is now pinned to
the pyproject version and this module guards the whole bug class: every
third-party top-level import under ``app/`` must be carried by the manifest.
"""

import ast
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
MANIFEST_PATH = REPO_ROOT / "app" / "requirements.txt"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# First-party packages under app/ (their imports are not manifest deps).
_FIRST_PARTY = {"app", "heart"}
# Import name overrides: a manifest pin whose import name differs from the pin.
_IMPORT_NAME_OVERRIDES = {"scikit-learn": "sklearn"}


def _manifest_pins() -> dict[str, str]:
    """Parse `name==version` pins from the deploy manifest (comments stripped)."""
    pins: dict[str, str] = {}
    for raw in MANIFEST_PATH.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        assert "==" in line, f"every manifest dep must be pinned, got: {raw!r}"
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()
    return pins


def _pyproject_dep_version(name: str) -> str:
    """Return the ``==`` version declared for *name* in pyproject.toml.

    Searches both the main ``dependencies`` list and the ``ml`` optional extra
    (shap lives in the extra, mlflow in the main deps).
    """
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    candidate_lists = [data["project"].get("dependencies", [])]
    candidate_lists += list(data["project"].get("optional-dependencies", {}).values())
    for deps in candidate_lists:
        matches = [d for d in deps if d.split("==")[0] == name]
        if matches:
            return matches[0].split("==", 1)[1]
    raise AssertionError(f"pyproject.toml must declare {name}==<version>")


def _pyproject_shap_version() -> str:
    return _pyproject_dep_version("shap")


def _manifest_covered_import_names() -> set[str]:
    """Set of top-level import names the manifest's pins provide."""
    covered: set[str] = set()
    for pin in _manifest_pins():
        covered.add(_IMPORT_NAME_OVERRIDES.get(pin, pin))
    return covered


def _third_party_app_imports() -> set[str]:
    """Top-level third-party module names imported anywhere under ``app/``."""
    found: set[str] = set()
    for path in APP_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                top = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 or not node.module:  # relative import within app/
                    continue
                top = node.module.split(".")[0]
            else:
                continue
            if top in sys.stdlib_module_names or top in _FIRST_PARTY:
                continue
            found.add(top)
    return found


def test_manifest_pins_shap_matching_pyproject():
    """Manifest shap pin equals pyproject [ml] pin — same version, no drift."""
    pins = _manifest_pins()
    assert "shap" in pins, "app/requirements.txt must pin shap for request-time SHAP"
    assert pins["shap"] == _pyproject_dep_version("shap")


def test_manifest_pins_mlflow_matching_pyproject():
    """Manifest mlflow pin equals pyproject pin — the deployed-boot bug guard.

    Regression for the 2026-10-13 M007 redeploy crash: the Predict page's
    import closure imports mlflow at module load, so a manifest without the
    exact pyproject mlflow pin crashes the deployed app at boot.
    """
    pins = _manifest_pins()
    assert "mlflow" in pins, (
        "app/requirements.txt must pin mlflow — the deployed Predict page "
        "imports it at module load and cannot boot without it"
    )
    assert pins["mlflow"] == _pyproject_dep_version("mlflow")


def test_manifest_pins_optuna_matching_pyproject():
    """Manifest optuna pin equals pyproject [ml] pin — deployed-boot bug guard.

    Regression for the 2026-10-14 deployed crash observed during M008/S01 T02's
    live browser session: every deployed page rendered the redacted ImportError
    from ``src/heart/tuning/runner.py:63`` because ``heart.tuning.runner``
    imports ``optuna`` at module load and the app-scoped manifest never carried
    it (the same bug class as the 2026-10-13 mlflow crash above).
    """
    pins = _manifest_pins()
    assert "optuna" in pins, (
        "app/requirements.txt must pin optuna — the deployed app's import "
        "closure reaches heart.tuning.runner, which imports it at module load"
    )
    assert pins["optuna"] == _pyproject_dep_version("optuna")


def test_app_third_party_imports_all_carried_by_manifest():
    """Every third-party import under app/ is provided by the deploy manifest.

    This is the class-level guard against the boot-crash bug: if any app module
    starts importing a package that app/requirements.txt does not carry, the
    deployed app would crash at import time — this test fails first, in CI.
    """
    covered = _manifest_covered_import_names()
    needed = _third_party_app_imports()
    missing = needed - covered
    assert not missing, (
        "app/ imports third-party packages missing from app/requirements.txt "
        f"(the deployed app would crash at boot): {sorted(missing)}"
    )


def test_manifest_stays_direct_deps_only():
    """No matplotlib/seaborn in the deploy manifest — the app charts natively."""
    pins = _manifest_pins()
    assert "matplotlib" not in pins
    assert "seaborn" not in pins
    # numba/slicer/tqdm are shap transitives — pip resolves them, never listed.
    assert "numba" not in pins
    assert "slicer" not in pins
    assert "tqdm" not in pins


def test_ci_has_deploy_manifest_gate_job():
    """CI installs the deploy manifest and asserts the pinned shap version."""
    ci = yaml.safe_load(CI_PATH.read_text())  # malformed YAML fails the suite
    jobs = ci["jobs"]
    assert "test" in jobs and "build" in jobs, "existing jobs must remain"
    job = jobs["deploy-manifest"]
    assert job["runs-on"] == "ubuntu-latest"
    setup = next(s for s in job["steps"] if "setup-python" in str(s.get("uses", "")))
    assert setup["with"]["cache-dependency-path"] == "app/requirements.txt"
    runs = " ".join(str(s.get("run", "")) for s in job["steps"])
    assert "pip install -r app/requirements.txt" in runs
    assert "import shap, streamlit, sklearn, joblib, pandas, numpy, scipy, mlflow" in runs
    assert f"shap.__version__ == '{_pyproject_dep_version('shap')}'" in runs
    # mlflow guard: the manifest must smoke-import + pin-assert mlflow too
    # (regression for the 2026-10-13 deployed boot crash).
    assert f"mlflow.__version__ == '{_pyproject_dep_version('mlflow')}'" in runs


def test_ci_triggers_on_every_pr_and_push():
    """The manifest gate must run on the same triggers as the test job."""
    ci = yaml.safe_load(CI_PATH.read_text())
    # PyYAML (YAML 1.1) coerces the bare `on` key to boolean True.
    on = ci.get("on", ci.get(True))
    assert "pull_request" in on and "push" in on  # both, like `test`
