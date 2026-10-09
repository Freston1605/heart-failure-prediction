"""Tests for the classical-battery registry (S03/T01).

Two contracts are under test:

1. **Battery membership** — :mod:`heart.models.registry` declares exactly the
   fixed classical battery: three logistic-regression variants (L2, L1,
   ElasticNet), Naive Bayes, LDA, QDA, kNN, SVM, Random Forest, and XGBoost.
   Every member resolves by ``model_type`` and constructs a real (unfitted)
   estimator from its spec.

2. **Search-space contract** — each :class:`~heart.models.spaces.SearchSpace`
   declares exactly the hyperparameters its spec tunes: the space keys, the
   spec's ``declared_hyperparameters``, the sampled-trial keys, and the keys
   the estimator's ``get_params()`` accepts are all identical. That is the
   guarantee the runner (S03/T02) relies on to stay generic.

Negative tests pin the named failure paths: unknown/blank model types, missing
or unknown hyperparameters (fixed params may not be overridden), malformed
:class:`~heart.models.spaces.ParamSpec` entries, duplicate parameter names,
and the XGBoost missing-dependency path.

No test writes to the repository's ``experiments/`` tree or the ``.gsd``
state; Optuna studies here run in memory.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import numpy as np
import optuna
import pytest
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from heart.data.schema import FEATURE_COLUMNS
from heart.models.registry import (
    BATTERY_MODEL_TYPES,
    BATTERY_MODELS,
    BATTERY_SIZE,
    BATTERY_SPECS,
    MissingHyperparameterError,
    ModelDependencyError,
    ModelSpec,
    RegistryError,
    UnknownHyperparameterError,
    UnknownModelTypeError,
    build_estimator,
    build_pipeline,
    registry_report,
    resolve_spec,
    validate_params,
)
from heart.models.spaces import (
    CATEGORICAL_KIND,
    FLOAT_KIND,
    INT_KIND,
    DuplicateParameterError,
    InvalidParameterSpecError,
    ParamSpec,
    SearchSpace,
    SearchSpaceError,
    UnknownParameterError,
    default_params,
    sample_params,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: The fixed battery, exactly as declared (leaderboard order).
EXPECTED_MODEL_TYPES: tuple[str, ...] = (
    "logistic-regression-l2",
    "logistic-regression-l1",
    "logistic-regression-elasticnet",
    "naive-bayes",
    "lda",
    "qda",
    "knn",
    "svm",
    "random-forest",
    "xgboost",
)

#: Mapping from model_type to its expected estimator class.
EXPECTED_ESTIMATOR: dict[str, type] = {
    "logistic-regression-l2": LogisticRegression,
    "logistic-regression-l1": LogisticRegression,
    "logistic-regression-elasticnet": LogisticRegression,
    "naive-bayes": GaussianNB,
    "lda": LinearDiscriminantAnalysis,
    "qda": QuadraticDiscriminantAnalysis,
    "knn": KNeighborsClassifier,
    "svm": SVC,
    "random-forest": RandomForestClassifier,
    "xgboost": __import__("xgboost", fromlist=["XGBClassifier"]).XGBClassifier,
}

ALL_BATTERY = list(BATTERY_MODELS)


# ---------------------------------------------------------------------------
# Battery membership
# ---------------------------------------------------------------------------


def test_battery_size_and_exact_membership():
    assert BATTERY_SIZE == len(EXPECTED_MODEL_TYPES)
    assert tuple(BATTERY_MODEL_TYPES) == EXPECTED_MODEL_TYPES
    assert set(BATTERY_SPECS) == set(EXPECTED_MODEL_TYPES)


@pytest.mark.parametrize("model_type", EXPECTED_MODEL_TYPES)
def test_every_battery_member_resolves(model_type):
    spec = resolve_spec(model_type)
    assert spec.model_type == model_type
    assert spec is BATTERY_SPECS[model_type]
    assert spec.model_name
    assert spec.family
    assert len(spec.space) >= 1


def test_logistic_regression_variants_cover_three_penalties():
    l2 = resolve_spec("logistic-regression-l2")
    l1 = resolve_spec("logistic-regression-l1")
    en = resolve_spec("logistic-regression-elasticnet")
    # sklearn >= 1.8 folds ``penalty`` into ``l1_ratio``: 0=L2, 1=L1, (0,1)=EN.
    assert l2.fixed_params["l1_ratio"] == 0.0
    assert l1.fixed_params["l1_ratio"] == 1.0
    assert "l1_ratio" not in en.fixed_params  # tuned for elastic-net
    assert "l1_ratio" in en.space.keys
    assert l2.fixed_params["solver"] == "lbfgs"
    assert l1.fixed_params["solver"] == "liblinear"
    assert en.fixed_params["solver"] == "saga"
    # C/class_weight are tuned in each variant; solver is never tuned.
    for spec in (l2, l1, en):
        assert "C" in spec.space.keys
        assert "solver" in spec.fixed_params
        assert "solver" not in spec.space.keys


def test_xgboost_present_in_registry_with_lazy_dependency():
    spec = resolve_spec("xgboost")
    assert spec.family == "gradient-boosting"
    # Construction imports the package only inside the factory, so the module
    # itself imports without xgboost on sys.path.
    estimator = build_estimator(spec)
    assert hasattr(estimator, "fit") and hasattr(estimator, "predict_proba")


# ---------------------------------------------------------------------------
# Search-space contract: keys == declared hyperparameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_space_keys_match_declared_hyperparameters(spec):
    """The core guarantee: spec declares exactly the tuned keys, and each key
    is a real parameter of the estimator it constructs."""
    assert frozenset(spec.space.keys) == spec.declared_hyperparameters
    estimator = build_estimator(spec)
    estimator_params = set(estimator.get_params())
    for key in spec.space.keys:
        assert key in estimator_params, (
            f"{spec.model_type}: tuned key {key!r} is not accepted by "
            f"{type(estimator).__name__}."
        )


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_default_params_match_space_keys(spec):
    params = default_params(spec.space)
    assert set(params) == set(spec.space.keys)
    # Every default lies inside its declared region.
    for param_spec in spec.space:
        value = params[param_spec.name]
        if param_spec.kind == CATEGORICAL_KIND:
            assert value in param_spec.choices
        else:
            assert param_spec.low <= float(value) <= param_spec.high


@dataclass
class _RecordingTrial:
    """Stand-in Optuna trial that records which keys were suggested.

    Exposes exactly the Optuna suggest protocol; never depends on Optuna, so
    the spaces module's sampling surface can be tested in isolation.
    """

    requested: list[str] = field(default_factory=list)
    counter: dict[str, int] = field(default_factory=dict)

    def _record(self, name: str) -> None:
        self.requested.append(name)
        self.counter[name] = self.counter.get(name, 0) + 1

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self._record(name)
        return float(low)

    def suggest_int(self, name, low, high, *, step=None, log=False):
        self._record(name)
        return int(low)

    def suggest_categorical(self, name, choices):
        self._record(name)
        return choices[0]


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_recording_trial_samples_exactly_the_declared_keys(spec):
    """A protocol-level fake trial asks for precisely the declared keys."""
    trial = _RecordingTrial()
    sampled = sample_params(spec.space, trial)
    assert set(trial.requested) == set(spec.space.keys)
    assert set(sampled) == set(spec.space.keys)
    assert sample_params(spec.space, trial) == spec.space.sample(trial)  # parity


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_optuna_trial_samples_exactly_the_declared_keys(spec):
    """A real Optuna trial samples precisely the spec's search-space keys."""
    captured: dict[str, object] = {}
    space = spec.space

    def objective(trial):
        captured.update(space.sample(trial))
        return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    assert set(captured) == set(space.keys)


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_construct_each_estimator_from_sampled_params(spec):
    """Each battery member constructs from the params the runner would produce."""
    estimator = build_estimator(spec, default_params(spec.space))
    assert isinstance(estimator, ClassifierMixin)
    assert hasattr(estimator, "predict")
    assert hasattr(estimator, "predict_proba")
    assert isinstance(estimator, EXPECTED_ESTIMATOR[spec.model_type])


def test_pipeline_appends_shared_preprocessing():
    spec = resolve_spec("logistic-regression-l2")
    pipeline = build_pipeline(spec)
    assert [name for name, _ in pipeline.steps] == ["preprocess", "clf"]
    assert pipeline.named_steps["clf"].C == 1.0
    assert pipeline.named_steps["preprocess"].named_steps["features"]


def test_build_pipeline_honours_the_preprocessing_flag():
    spec = resolve_spec("knn")
    bare = ModelSpec(
        model_type="knn-bare",
        model_name="kNN (bare)",
        family="distance",
        space=spec.space,
        estimator_factory=spec.estimator_factory,
        fixed_params=spec.fixed_params,
        preprocessing=False,
        description="bare kNN for tests",
    )
    resolved = build_pipeline(bare)
    assert isinstance(resolved, KNeighborsClassifier)  # no preprocessing chain


# ---------------------------------------------------------------------------
# Sampled defaults stay within reach of every estimator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ALL_BATTERY, ids=lambda s: s.model_type)
def test_fixed_params_survive_construction(spec):
    estimator = build_estimator(spec)
    for key, expected in spec.fixed_params.items():
        assert getattr(estimator, key) == expected, (
            f"{spec.model_type}: fixed param {key} expected {expected!r} but "
            f"got {getattr(estimator, key)!r}."
        )


def test_lda_drops_shrinkage_for_svd_solver():
    spec = resolve_spec("lda")
    svd_estimator = build_estimator(spec, {"solver": "svd", "shrinkage": 0.5})
    assert isinstance(svd_estimator, LinearDiscriminantAnalysis)
    # sklearn forbids combining svd with shrinkage; the factory drops it.
    assert svd_estimator.get_params()["shrinkage"] is None
    lsqr_estimator = build_estimator(spec, {"solver": "lsqr", "shrinkage": 0.5})
    assert lsqr_estimator.get_params()["shrinkage"] == 0.5


def test_svm_drops_gamma_for_linear_kernel():
    spec = resolve_spec("svm")
    linear = build_estimator(
        spec, {"C": 1.0, "kernel": "linear", "gamma": 0.1, "class_weight": None}
    )
    rbf = build_estimator(
        spec, {"C": 1.0, "kernel": "rbf", "gamma": 0.1, "class_weight": None}
    )
    assert linear.get_params()["gamma"] == "scale"  # sklearn default, unresolved
    assert rbf.get_params()["gamma"] == 0.1
    assert linear.probability is True  # probabilities are pinned for SVC


# ---------------------------------------------------------------------------
# Negative tests: resolution and parameter validation
# ---------------------------------------------------------------------------


def test_resolve_spec_rejects_blank_model_type():
    with pytest.raises(UnknownModelTypeError):
        resolve_spec("   ")


def test_resolve_spec_rejects_unknown_model_type():
    with pytest.raises(UnknownModelTypeError, match="kmeans"):
        resolve_spec("kmeans")


def test_validate_params_rejects_partial_tuned_dict():
    spec = resolve_spec("logistic-regression-l2")
    with pytest.raises(MissingHyperparameterError, match="C"):
        validate_params(spec, {"class_weight": "balanced"})


def test_validate_params_rejects_unknown_hyperparameter():
    spec = resolve_spec("random-forest")
    with pytest.raises(UnknownHyperparameterError):
        validate_params(spec, dict(default_params(spec.space), nonsense=1))


def test_fixed_params_cannot_be_overridden():
    """A caller may only supply tuned keys; fixed params are pinned by spec."""
    spec = resolve_spec("logistic-regression-l2")
    params = dict(default_params(spec.space), solver="newton-cg")
    with pytest.raises(UnknownHyperparameterError, match="solver"):
        build_estimator(spec, params)


def test_build_estimator_rejects_non_mapping_params():
    spec = resolve_spec("qda")
    with pytest.raises(RegistryError):
        build_estimator(spec, [0.5])


# ---------------------------------------------------------------------------
# Negative tests: search-space definition
# ---------------------------------------------------------------------------


def test_param_spec_rejects_unknown_kind():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="x", kind="logistic", low=0.0, high=1.0)


def test_param_spec_rejects_non_numeric_bounds():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="x", kind=FLOAT_KIND, low="low", high="high")


def test_param_spec_rejects_inverted_bounds():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="x", kind=FLOAT_KIND, low=2.0, high=1.0)


def test_param_spec_rejects_non_positive_step():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="x", kind=FLOAT_KIND, low=0.0, high=1.0, step=0.0)


def test_param_spec_rejects_fractional_step_for_int():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="k", kind=INT_KIND, low=1, high=10, step=0.5)


def test_param_spec_rejects_categorical_without_choices():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="kernel", kind=CATEGORICAL_KIND)


def test_param_spec_rejects_default_outside_region():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="C", kind=FLOAT_KIND, low=1e-4, high=1e2, default=999.0)


def test_param_spec_rejects_int_default_outside_region():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(name="k", kind=INT_KIND, low=1, high=10, default=20)


def test_param_spec_rejects_categorical_default_not_in_choices():
    with pytest.raises(InvalidParameterSpecError):
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=("balanced",),
            default="auto",
        )


def test_search_space_rejects_duplicate_names():
    with pytest.raises(DuplicateParameterError):
        SearchSpace(
            params=(
                ParamSpec(name="C", kind=FLOAT_KIND, low=1e-4, high=1e2),
                ParamSpec(name="C", kind=FLOAT_KIND, low=1e-3, high=1e1),
            )
        )


def test_search_space_get_unknown_parameter():
    space = resolve_spec("naive-bayes").space
    with pytest.raises(UnknownParameterError, match="nope"):
        space.get("nope")


def test_sample_rejects_none_trial():
    spec = resolve_spec("knn")
    with pytest.raises(SearchSpaceError):
        spec.space.sample(None)


# ---------------------------------------------------------------------------
# Negative tests: ModelSpec validation + dependency path
# ---------------------------------------------------------------------------


def test_model_spec_rejects_empty_search_space():
    with pytest.raises(RegistryError, match="at least one"):
        ModelSpec(
            model_type="ghost",
            model_name="Ghost",
            family="linear",
            space=SearchSpace(params=()),
            estimator_factory=lambda params: LogisticRegression(**dict(params)),
        )


def test_model_spec_rejects_unknown_family():
    # ``neural`` is a declared family since S05/T03 (the MLP sweep reuses the
    # ModelSpec machinery); a genuinely unknown family is still rejected.
    with pytest.raises(RegistryError, match="family"):
        ModelSpec(
            model_type="ghost",
            model_name="Ghost",
            family="transformer",
            space=resolve_spec("qda").space,
            estimator_factory=lambda params: LogisticRegression(**dict(params)),
        )


def test_model_spec_rejects_whitespace_in_model_type():
    with pytest.raises(RegistryError, match="whitespace"):
        ModelSpec(
            model_type="ghost model",
            model_name="Ghost",
            family="linear",
            space=resolve_spec("qda").space,
            estimator_factory=lambda params: LogisticRegression(**dict(params)),
        )


def test_xgboost_dependency_failure_is_named(monkeypatch):
    """When xgboost cannot be imported, the error names the package and fix."""
    spec = resolve_spec("xgboost")
    # ``import xgboost`` with None in sys.modules raises ImportError, exactly
    # as if the package were uninstalled.
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(ModelDependencyError, match="pip install"):
        build_estimator(spec)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_registry_report_lists_every_battery_member():
    report = registry_report()
    assert f"classical battery: {BATTERY_SIZE} members" in report
    for model_type in EXPECTED_MODEL_TYPES:
        assert model_type in report
    assert "logistic-regression-l2" in report
    assert "xgboost" in report


# ---------------------------------------------------------------------------
# A tiny end-to-end smoke so the battery is not just metadata
# ---------------------------------------------------------------------------


def _schema_valid_frame(rows: int, seed: int) -> "pd.DataFrame":
    """A small frame matching the declared schema, labels included."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, rows)
    return pd.DataFrame(
        {
            "Age": rng.integers(29, 78, rows),
            "Sex": rng.choice(["F", "M"], rows),
            "ChestPainType": rng.choice(["ASY", "ATA", "NAP", "TA"], rows),
            "RestingBP": rng.integers(95, 190, rows),
            "Cholesterol": np.where(
                rng.random(rows) < 0.2,
                0,
                rng.integers(120, 340, rows),
            ),
            "FastingBS": rng.integers(0, 2, rows),
            "RestingECG": rng.choice(["Normal", "ST", "LVH"], rows),
            "MaxHR": rng.integers(70, 200, rows),
            "ExerciseAngina": rng.choice(["N", "Y"], rows),
            "Oldpeak": rng.uniform(0.0, 4.0, rows).round(2),
            "ST_Slope": rng.choice(["Up", "Flat", "Down"], rows),
            "HeartDisease": labels,
        }
    )


def test_battery_members_fit_and_pred_prob_array_with_preprocessing():
    """Smallest meaningful integration proof: fit+score a subset of members
    through the shared preprocessing pipeline on synthetic-schema data."""
    rows = 60
    frame = _schema_valid_frame(rows, seed=42)
    features = frame[list(FEATURE_COLUMNS)]
    labels = frame["HeartDisease"]
    for model_type in ("logistic-regression-l2", "knn", "naive-bayes"):
        spec = resolve_spec(model_type)
        tuned = {key: spec.space.get(key).default for key in spec.space.keys}
        pipeline = build_pipeline(spec, tuned)
        pipeline.fit(features, labels)
        probabilities = pipeline.predict_proba(features)
        assert probabilities.shape == (rows, 2)
        assert np.all(np.isfinite(probabilities))


def test_reduced_forest_configuration_fits():
    """A deliberately small forest configuration (used by smoke runs) fits."""
    rows = 80
    frame = _schema_valid_frame(rows, seed=7)
    features = frame[list(FEATURE_COLUMNS)]
    labels = frame["HeartDisease"]
    spec = resolve_spec("random-forest")
    tuned = {
        "n_estimators": 20,
        "max_depth": 4,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": None,
    }
    pipeline = build_pipeline(spec, tuned)
    pipeline.fit(features, labels)
    probabilities = pipeline.predict_proba(features)
    assert probabilities.shape == (rows, 2)
    assert np.all(np.isfinite(probabilities))