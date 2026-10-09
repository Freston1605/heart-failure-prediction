"""The fixed classical battery as a registry of model specs (S03/T01).

This module is the single source of truth for **what** the classical battery
contains. One :class:`ModelSpec` per member declares:

* a stable ``model_type`` key and a human-readable ``model_name`` (the latter
  is slugified into the MLflow run name by ``heart.tracking.run``);
* its :class:`~heart.models.spaces.SearchSpace` — the Optuna region the tuning
  runner will explore;
* an ``estimator_factory`` — a plain ``params -> estimator`` function so the
  runner never switches on model identity;
* ``fixed_params`` that are pinned (solver/penalty choices, ``probability``,
  ``random_state``) and never tuned.

Important design property: **the runner stays generic**. It imports
:func:`build_pipeline` here, samples ``spec.space``, and calls
``spec.estimator_factory`` — there is no ``if model_type == ...`` anywhere
downstream. Adding a model to the battery is a one-entry registry change.

An estimator dependency that is optional at import time (XGBoost) is constructed
through :func:`_make_xgboost_classifier`, which imports the package lazily and
raises the named :class:`ModelDependencyError` with install guidance when the
package is absent. The registry itself therefore imports without XGBoost
installed.

Observability
-------------
:func:`registry_report` renders the full battery (model type, family, tuned
keys, fixed params, description) for human inspection.
:func:`tests/test_registry.py` asserts every *declared* battery member exists,
that constructing each estimator from its sampled defaults succeeds, and that
each spec's search-space keys and declared hyperparameters are identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Mapping

from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from heart.config import RANDOM_SEED
from heart.data.pipeline import build_preprocessing_pipeline
from heart.models.spaces import (
    CATEGORICAL_KIND,
    SearchSpace,
    default_params,
    knn_space,
    lda_space,
    logistic_regression_elasticnet_space,
    logistic_regression_l1_space,
    logistic_regression_l2_space,
    naive_bayes_space,
    qda_space,
    random_forest_space,
    svm_space,
    xgboost_space,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class RegistryError(Exception):
    """Base class for every registry, spec, or construction failure."""


class UnknownModelTypeError(RegistryError):
    """The requested model type is not a member of the declared battery."""


class UnknownHyperparameterError(RegistryError):
    """A parameter dict contains a key the spec does not declare."""


class MissingHyperparameterError(RegistryError):
    """A parameter dict omits a tuned hyperparameter the spec declares."""


class ModelDependencyError(RegistryError):
    """An estimator's optional third-party package is not installed."""


# ---------------------------------------------------------------------------
# The model spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One member of the fixed classical battery.

    Parameters
    ----------
    model_type:
        The registry key (also used for CLI ``--model-type`` values).
    model_name:
        Human-readable name; slugified into MLflow run names.
    family:
        Coarse estimator family used in reports and tags.
    space:
        The :class:`SearchSpace` the tuning runner explores. Declares exactly
        the tuned hyperparameters.
    estimator_factory:
        ``params -> estimator``. Receives the merged fixed+tuned parameters
        and returns an *unfitted* classifier exposing ``predict`` and
        ``predict_proba`` after fitting.
    fixed_params:
        Hyperparameters pinned by the spec and never tuned (for example
        ``solver='lbfgs'`` or ``probability=True``).
    preprocessing:
        Whether the spec's pipeline prepends the shared leakage-safe
        preprocessing chain (imputation, scaling, one-hot encoding). Every
        battery member uses it; the flag lets tests exercise a bare estimator.
    description:
        One-sentence human summary of the member.
    """

    model_type: str
    model_name: str
    family: str
    space: SearchSpace
    estimator_factory: Callable[[Mapping[str, object]], ClassifierMixin]
    fixed_params: Mapping[str, object] = field(default_factory=dict)
    preprocessing: bool = True
    description: str = ""

    _FAMILIES: ClassVar[frozenset[str]] = frozenset(
        {
            "linear",
            "bayes",
            "discriminant",
            "distance",
            "kernel",
            "ensemble",
            "gradient-boosting",
            # Neural family added in S05/T03: the tuned MLP sweep reuses the
            # classical ModelSpec machinery (family tag, pipeline contract), so
            # its specs need a declared family that is not a classical one.
            "neural",
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise RegistryError(
                f"model_type must be a non-empty string, got {self.model_type!r}."
            )
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise RegistryError(
                f"model_name must be a non-empty string, got {self.model_name!r}."
            )
        if self.model_type.strip() != self.model_type or any(
            character.isspace() for character in self.model_type
        ):
            raise RegistryError(
                f"model_type {self.model_type!r} may not contain whitespace; it "
                "is used as a registry key and CLI value."
            )
        if self.family not in self._FAMILIES:
            raise RegistryError(
                f"model_type {self.model_type!r} has unknown family "
                f"{self.family!r}; expected one of {sorted(self._FAMILIES)}."
            )
        if not self.space.params:
            raise RegistryError(
                f"model_type {self.model_type!r} must declare at least one tuned "
                "hyperparameter in its search space."
            )
        if not callable(self.estimator_factory):
            raise RegistryError(
                f"model_type {self.model_type!r} has a non-callable "
                "estimator_factory."
            )

    # -- declared hyperparameters ------------------------------------------

    @property
    def declared_hyperparameters(self) -> frozenset[str]:
        """The tuned hyperparameter names this spec declares.

        By contract this equals ``frozenset(space.keys)`` — a change on either
        side is a test failure, which is exactly how the "search space keys
        match the declared hyperparameters" guarantee is enforced. Fixed
        parameters are pinned by the spec and deliberately *not* part of this
        set, so a caller-supplied dict may never override them.
        """
        return frozenset(self.space.keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "family": self.family,
            "preprocessing": self.preprocessing,
            "fixed_params": dict(self.fixed_params),
            "tuned_hyperparameters": sorted(self.declared_hyperparameters),
            "search_space": self.space.to_dict(),
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Estimator factories
# ---------------------------------------------------------------------------


def _make_logistic(params: Mapping[str, object]) -> LogisticRegression:
    return LogisticRegression(**dict(params))


def _make_naive_bayes(params: Mapping[str, object]) -> GaussianNB:
    return GaussianNB(**dict(params))


def _make_lda(params: Mapping[str, object]) -> LinearDiscriminantAnalysis:
    # ``shrinkage`` is only meaningful for the lsqr/eigen solvers; drop it for
    # svd so the sampled space never trips sklearn's solver/shinkage check.
    resolved = dict(params)
    if resolved.get("solver") == "svd":
        resolved.pop("shrinkage", None)
    return LinearDiscriminantAnalysis(**resolved)


def _make_qda(params: Mapping[str, object]) -> QuadraticDiscriminantAnalysis:
    return QuadraticDiscriminantAnalysis(**dict(params))


def _make_knn(params: Mapping[str, object]) -> KNeighborsClassifier:
    return KNeighborsClassifier(**dict(params))


def _make_svm(params: Mapping[str, object]) -> SVC:
    # ``gamma`` is only meaningful for kernel functions that use it; drop it
    # for the linear kernel so sampled params never trip sklearn's check.
    resolved = dict(params)
    if resolved.get("kernel") == "linear":
        resolved.pop("gamma", None)
    return SVC(**resolved)


def _make_random_forest(params: Mapping[str, object]) -> RandomForestClassifier:
    return RandomForestClassifier(**dict(params))


def _import_xgboost_classifier():
    """Import and return the XGBClassifier class, with a named failure path."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised when uninstalled
        raise ModelDependencyError(
            "The xgboost battery member requires the 'xgboost' package, which "
            'is not installed. Install the ML extra with: pip install -e ".[ml]"'
        ) from exc
    return XGBClassifier


def _make_xgboost(params: Mapping[str, object]) -> ClassifierMixin:
    """Construct ``XGBClassifier`` from ``params`` (package imported lazily)."""
    classifier_type = _import_xgboost_classifier()
    return classifier_type(**dict(params))


# ---------------------------------------------------------------------------
# The fixed battery
# ---------------------------------------------------------------------------

#: Shared logistic-regression fixed parameters (baseline parity).
_LR_FIXED: dict[str, object] = {"max_iter": 2000, "random_state": RANDOM_SEED}

#: Solver/l1_ratio pairing for each logistic variant. sklearn >= 1.8 folds the
#: ``penalty`` kwarg into ``l1_ratio`` (l2 -> 0, l1 -> 1, elasticnet -> (0,1)),
#: so the variants are expressed in the forward-compatible API.
_LR_VARIANTS: tuple[tuple[str, float, str, str], ...] = (
    # (model_type slug, l1_ratio, solver, human name)
    ("logistic-regression-l2", 0.0, "lbfgs", "Logistic Regression (L2)"),
    ("logistic-regression-l1", 1.0, "liblinear", "Logistic Regression (L1)"),
    (
        "logistic-regression-elasticnet",
        0.5,
        "saga",
        "Logistic Regression (ElasticNet)",
    ),
)


def _logistic_specs() -> tuple[ModelSpec, ...]:
    spaces = {
        "logistic-regression-l2": logistic_regression_l2_space(),
        "logistic-regression-l1": logistic_regression_l1_space(),
        "logistic-regression-elasticnet": logistic_regression_elasticnet_space(),
    }
    specs: list[ModelSpec] = []
    for model_type, l1_ratio, solver, name in _LR_VARIANTS:
        if model_type == "logistic-regression-elasticnet":
            # l1_ratio is tuned for the elastic-net variant; the other two
            # variants pin it to 0 (L2) or 1 (L1) as part of their identity.
            fixed_l1_ratio: dict[str, object] = {}
            description = (
                f"logistic regression with elastic-net mixing weight tuned in "
                f"(0, 1) (solver saga); the tuned strength is C and the "
                "class-weight policy."
            )
        else:
            fixed_l1_ratio = {"l1_ratio": l1_ratio}
            description = (
                f"logistic regression with l1_ratio={l1_ratio:g} "
                f"(solver {solver}); the tuned strength is C and the "
                "class-weight policy."
            )
        specs.append(
            ModelSpec(
                model_type=model_type,
                model_name=name,
                family="linear",
                space=spaces[model_type],
                estimator_factory=_make_logistic,
                fixed_params={
                    **_LR_FIXED,
                    "solver": solver,
                    **fixed_l1_ratio,
                },
                description=description,
            )
        )
    return tuple(specs)


def _battery_specs() -> tuple[ModelSpec, ...]:
    """The full battery, in leaderboard order (linear -> boosting)."""
    return tuple(
        [
            *_logistic_specs(),
            ModelSpec(
                model_type="naive-bayes",
                model_name="Naive Bayes",
                family="bayes",
                space=naive_bayes_space(),
                estimator_factory=_make_naive_bayes,
                description=(
                    "Gaussian Naive Bayes; only the variance-smoothing floor "
                    "is tuned."
                ),
            ),
            ModelSpec(
                model_type="lda",
                model_name="LDA",
                family="discriminant",
                space=lda_space(),
                estimator_factory=_make_lda,
                description=(
                    "Linear discriminant analysis; solver and shrinkage are "
                    "tuned (shrinkage only applies to the lsqr solver)."
                ),
            ),
            ModelSpec(
                model_type="qda",
                model_name="QDA",
                family="discriminant",
                space=qda_space(),
                estimator_factory=_make_qda,
                description=(
                    "Quadratic discriminant analysis; covariance regularisation "
                    "is tuned."
                ),
            ),
            ModelSpec(
                model_type="knn",
                model_name="k-Nearest Neighbours",
                family="distance",
                space=knn_space(),
                estimator_factory=_make_knn,
                description=(
                    "k-nearest neighbours on the scaled feature space; "
                    "neighbourhood size, weighting, and Minkowski power are tuned."
                ),
            ),
            ModelSpec(
                model_type="svm",
                model_name="Support Vector Machine",
                family="kernel",
                space=svm_space(),
                estimator_factory=_make_svm,
                fixed_params={"probability": True, "random_state": RANDOM_SEED},
                description=(
                    "Support-vector classifier with calibrated probabilities; "
                    "C, kernel, and gamma are tuned."
                ),
            ),
            ModelSpec(
                model_type="random-forest",
                model_name="Random Forest",
                family="ensemble",
                space=random_forest_space(),
                estimator_factory=_make_random_forest,
                fixed_params={"random_state": RANDOM_SEED},
                description=(
                    "Random forest of pruned trees; ensemble size, tree depth, "
                    "split counts, feature sampling, and class weighting are tuned."
                ),
            ),
            ModelSpec(
                model_type="xgboost",
                model_name="XGBoost",
                family="gradient-boosting",
                space=xgboost_space(),
                estimator_factory=_make_xgboost,
                fixed_params={
                    "random_state": RANDOM_SEED,
                    "eval_metric": "logloss",
                },
                description=(
                    "Gradient-boosted trees (XGBoost); boosting rounds, depth, "
                    "learning rate, subsampling, and L2 regularisation are tuned."
                ),
            ),
        ]
    )


#: The complete battery keyed by ``model_type``.
BATTERY_SPECS: dict[str, ModelSpec] = {
    spec.model_type: spec for spec in _battery_specs()
}

#: Battery member keys, in leaderboard (registration) order.
BATTERY_MODEL_TYPES: tuple[str, ...] = tuple(BATTERY_SPECS)

#: Battery members, in leaderboard (registration) order.
BATTERY_MODELS: tuple[ModelSpec, ...] = tuple(BATTERY_SPECS.values())

#: Number of fixed battery members (asserted by ``test_registry.py``).
BATTERY_SIZE: int = len(BATTERY_MODELS)


# ---------------------------------------------------------------------------
# Resolution and construction
# ---------------------------------------------------------------------------


def resolve_spec(model_type: str) -> ModelSpec:
    """Return the declared spec for ``model_type`` or raise loudly."""
    if not isinstance(model_type, str) or not model_type.strip():
        raise UnknownModelTypeError(
            f"model_type must be a non-empty string, got {model_type!r}."
        )
    try:
        return BATTERY_SPECS[model_type.strip()]
    except KeyError as exc:
        raise UnknownModelTypeError(
            f"Unknown battery model type {model_type.strip()!r}; declared "
            f"members are {list(BATTERY_MODEL_TYPES)}."
        ) from exc


def validate_params(spec: ModelSpec, params: Mapping[str, object]) -> dict[str, object]:
    """Check a parameter dict against the spec's declared hyperparameters.

    Raises
    ------
    MissingHyperparameterError
        A tuned hyperparameter declared by the spec is absent.
    UnknownHyperparameterError
        The dict carries a key the spec does not accept (neither tuned nor
        fixed).
    """
    if not isinstance(params, Mapping):
        raise RegistryError(
            f"params must be a mapping, got {type(params).__name__}."
        )
    missing = sorted(spec.declared_hyperparameters - set(params))
    if missing:
        raise MissingHyperparameterError(
            f"Model {spec.model_type!r} is missing tuned hyperparameter(s) "
            f"{missing}; supply one value per declared search-space key "
            f"{sorted(spec.declared_hyperparameters)}."
        )
    unknown = sorted(set(params) - spec.declared_hyperparameters)
    if unknown:
        raise UnknownHyperparameterError(
            f"Model {spec.model_type!r} does not accept hyperparameter(s) "
            f"{unknown}; only the tuned search-space keys may be supplied "
            f"{sorted(spec.declared_hyperparameters)}. Fixed parameters are "
            f"pinned by the spec and cannot be overridden."
        )
    return dict(params)


def build_estimator(
    spec: ModelSpec, params: Mapping[str, object] | None = None
) -> ClassifierMixin:
    """Construct the *unfitted* estimator for ``spec`` with fixed+tuned params.

    ``params`` supplies the tuned hyperparameters (one per search-space key);
    when omitted, every parameter falls back to its declared default. The
    spec's ``fixed_params`` are always applied and cannot be overridden here —
    the fixed set is what keeps battery members comparable.
    """
    tuned = (
        default_params(spec.space) if params is None else validate_params(spec, params)
    )
    merged = dict(spec.fixed_params)
    merged.update(tuned)
    estimator = spec.estimator_factory(merged)
    if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
        raise RegistryError(
            f"estimator_factory of {spec.model_type!r} returned a "
            f"{type(estimator).__name__} without fit()/predict(); battery "
            "members must be scikit-learn-style classifiers."
        )
    logger.debug(
        "Built estimator for %s: fixed=%s tuned=%s",
        spec.model_type,
        dict(spec.fixed_params),
        tuned,
    )
    return estimator


def build_pipeline(
    spec: ModelSpec, params: Mapping[str, object] | None = None
) -> Pipeline | ClassifierMixin:
    """Build the model's full training object: preprocessing + estimator.

    Every battery member uses the shared leakage-safe preprocessing chain
    (zero-as-missing imputation, standard scaling, one-hot encoding) fitted on
    training rows only — the same chain as the S02 baseline. Specs with
    ``preprocessing=False`` return the bare estimator.
    """
    estimator = build_estimator(spec, params)
    if not spec.preprocessing:
        return estimator
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline()),
            ("clf", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def registry_report() -> str:
    """Render the fixed battery as a human-readable table."""
    lines = [
        f"classical battery: {BATTERY_SIZE} members",
        "",
        "| # | model_type | family | tuned params | fixed params |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, spec in enumerate(BATTERY_MODELS, start=1):
        lines.append(
            f"| {index} | `{spec.model_type}` | {spec.family} | "
            f"{', '.join(spec.space.keys)} | {dict(spec.fixed_params)} |"
        )
    lines.append("")
    lines.append("Search-space detail:")
    for spec in BATTERY_MODELS:
        lines.append("")
        lines.append(f"### {spec.model_type}")
        lines.append(f"_{spec.description}_")
        lines.append("")
        lines.append("| parameter | kind | region | default |")
        lines.append("| --- | --- | --- | --- |")
        for param in spec.space:
            if param.kind == CATEGORICAL_KIND:
                region = str(list(param.choices))
            else:
                region = (
                    f"[{param.low}, {param.high}]"
                    + (" log" if param.log else "")
                    + (f" step {param.step}" if param.step is not None else "")
                )
            lines.append(
                f"| `{param.name}` | {param.kind} | {region} | {param.default!r} |"
            )
    return "\n".join(lines)
