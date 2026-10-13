"""Guards for the Streamlit Community Cloud deploy manifest and its CI gate.

M007/S02/T03: the Predict page now imports shap at request time, so
app/requirements.txt must pin the exact shap version declared in
pyproject.toml (no drift), stay direct-deps-only (no matplotlib/seaborn —
the app charts natively), and .github/workflows/ci.yml must install + smoke
that manifest on every PR so a broken pin fails in CI, not at Cloud boot
(MEM117).
"""

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "app" / "requirements.txt"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


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


def _pyproject_shap_version() -> str:
    data = tomllib.loads(PYPROJECT_PATH.read_text())
    ml_deps = data["project"]["optional-dependencies"]["ml"]
    matches = [d for d in ml_deps if d.split("==")[0] == "shap"]
    assert matches, "pyproject.toml [ml] extras must declare shap"
    return matches[0].split("==", 1)[1]


def test_manifest_pins_shap_matching_pyproject():
    """Manifest shap pin equals pyproject [ml] pin — same version, no drift."""
    pins = _manifest_pins()
    assert "shap" in pins, "app/requirements.txt must pin shap for request-time SHAP"
    assert pins["shap"] == _pyproject_shap_version()


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
    assert "import shap, streamlit, sklearn, joblib, pandas, numpy, scipy" in runs
    assert f"shap.__version__ == '{_pyproject_shap_version()}'" in runs


def test_ci_triggers_on_every_pr_and_push():
    """The manifest gate must run on the same triggers as the test job."""
    ci = yaml.safe_load(CI_PATH.read_text())
    # PyYAML (YAML 1.1) coerces the bare `on` key to boolean True.
    on = ci.get("on", ci.get(True))
    assert "pull_request" in on and "push" in on  # both, like `test`
